
# -*- coding: utf-8 -*-
# A new, modern logger for PyTorch using the standard TensorBoard library.
# This version removes the unnecessary TensorFlow 1.x dependency.

import numpy as np
from torch.utils.tensorboard import SummaryWriter

# 不再需要 tensorflow 或 scipy.misc

class Logger(object):
    def __init__(self, log_dir):
        # Create a summary writer logging to log_dir.
        # 使用 PyTorch 自带的 SummaryWriter
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        # Log a scalar variable.
        # 使用新的 API: add_scalar
        self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, images, step):
        
        # Log a list of images.
        # The PyTorch SummaryWriter add_images method expects a tensor 
        # in the format (N, C, H, W).
        
        # (此函数在当前 train.py 中未被调用，但我们提供一个正确的实现)
        if isinstance(images, list):
            images = np.array(images)

        # 如果图像格式是 (N, H, W, C)，转换为 (N, C, H, W)
        if images.ndim == 4 and images.shape[3] in [1, 3]:
            images = images.transpose(0, 3, 1, 2)
        
        # PyTorch 的 add_images 可以直接处理 NumPy 数组或 PyTorch Tensor
        self.writer.add_images(tag, images, step, dataformats='NCHW')

    def histo_summary(self, tag, values, step, bins='auto'):
        # Log a histogram of the tensor of values.
        # 使用新的 API: add_histogram
        self.writer.add_histogram(tag, values, step, bins=bins)
        self.writer.flush() # 保持和原版行为一致，在记录后立即刷新

    def close(self):
        # Close the writer.
        self.writer.close()

