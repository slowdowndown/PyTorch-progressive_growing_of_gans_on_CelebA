
# -*- coding: utf-8 -*-
import os
from glob import glob
import numpy as np 
import h5py
import imageio              # 修正: 导入 imageio 用于保存图片
from PIL import Image       # 修正: 导入 Pillow(PIL) 用于读取和缩放图片


prefix = './datasets/'

# 这个函数虽然在当前训练流程中未被激活的CelebA类使用，
# 但我们还是为它修正了兼容性问题，以防未来使用。
def get_img(img_path, is_crop=True, crop_h=256, resize_h=64, normalize=False):
    # 修正: 使用 Pillow 读取图片
    img = Image.open(img_path).convert('RGB')
    
    resize_w = resize_h
    if is_crop:
        crop_w = crop_h
        w, h = img.size
        j = int(round((h - crop_h) / 2.))
        i = int(round((w - crop_w) / 2.))
        img = img.crop((i, j, i + crop_w, j + crop_h))
    
    # 修正: 使用 Pillow 缩放图片，LANCZOS 是高质量的下采样滤镜
    img = img.resize((resize_w, resize_h), Image.LANCZOS)
    
    cropped_image = np.array(img).astype(np.float32)

    if normalize:
        cropped_image = cropped_image / 127.5 - 1.0
    
    return np.transpose(cropped_image, [2, 0, 1])


class CelebA():
    def __init__(self):
        # .h5 文件名在这里被硬编码了
        # 请确保用 h5tool.py 创建的文件名与此一致
        datapath = 'celeba-128x128.h5'
        
        self._base_key = 'data'
        h5_path = os.path.join(prefix, datapath)
        
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"数据集文件未找到: {h5_path}。请先运行 h5tool.py 生成数据集。")

        self.dataset = h5py.File(h5_path, 'r')
        self._len = {k:len(self.dataset[k]) for k in self.dataset.keys() if k.startswith('data')}
        
        # 检查所需的分辨率是否存在
        # 例如，对于128x128的目标，我们需要4x4到128x128的所有层级
        max_resol = 0
        for k in self._len.keys():
            res = int(k.replace('data','').split('x')[0])
            if res > max_resol:
                max_resol = res
        
        required_resolutions = [f'data{2**r}x{2**r}' for r in range(2, int(np.log2(max_resol)) + 1)]
        for r_str in required_resolutions:
             if r_str not in self.dataset:
                 raise KeyError(f"所需的分辨率 '{r_str}' 在HDF5文件中未找到。可用分辨率: {list(self.dataset.keys())}")


    def __call__(self, batch_size, size, level=None):
        key = self._base_key + '{}x{}'.format(size, size)
        idx = np.random.randint(self._len[key], size=batch_size)
        batch_x = np.array([self.dataset[key][i] for i in idx], dtype=np.float32)
        batch_x = batch_x / 127.5 - 1.0 # 归一化到 [-1, 1]
        
        if level is not None and level != int(level):
            # 这是PGGAN核心的 "fade-in" 逻辑
            alpha = level - int(level)
            
            # 从下一个更低的分辨率进行上采样
            if size > 4: # 最小分辨率是 4x4
                lr_key = self._base_key + '{}x{}'.format(size//2, size//2)
                low_resol_batch_x_lr = np.array([self.dataset[lr_key][i] for i in idx], dtype=np.float32)
                low_resol_batch_x_lr = low_resol_batch_x_lr / 127.5 - 1.0
                
                # 使用 repeat 进行简单的最近邻上采样
                low_resol_batch_x = low_resol_batch_x_lr.repeat(2, axis=2).repeat(2, axis=3)
                
                # 按alpha值混合高分辨率和上采样后的低分辨率图像
                batch_x = batch_x * alpha + low_resol_batch_x * (1 - alpha)
                
        return batch_x

    def save_imgs(self, samples, file_name):
        N_samples, channel, height, width = samples.shape
        N_row = N_col = int(np.ceil(N_samples**0.5))
        
        # 将样本从 [-1, 1] 范围转换回 [0, 255] 的图片范围
        samples = ((samples + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        
        combined_imgs = np.ones((channel, N_row*height, N_col*width), dtype=np.uint8)
        for i in range(N_row):
            for j in range(N_col):
                if i*N_col+j < samples.shape[0]:
                    combined_imgs[:,i*height:(i+1)*height, j*width:(j+1)*width] = samples[i*N_col+j]
        
        combined_imgs = np.transpose(combined_imgs, [1, 2, 0])
        
        # 修正: 使用 imageio.imwrite 来替代过时的 imsave
        imageio.imwrite(file_name + '.png', combined_imgs)


class RandomNoiseGenerator():
    def __init__(self, size, noise_type='gaussian'):
        self.size = size
        self.noise_type = noise_type.lower()
        assert self.noise_type in ['gaussian', 'uniform']
        if self.noise_type == 'gaussian':
            self.generator = lambda s: np.random.randn(*s)
        elif self.noise_type == 'uniform':
            self.generator = lambda s: np.random.uniform(-1, 1, size=s)

    def __call__(self, batch_size):
        return self.generator([batch_size, self.size]).astype(np.float32)

