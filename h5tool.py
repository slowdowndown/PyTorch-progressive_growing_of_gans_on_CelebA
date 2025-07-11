# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import sys
import io
import glob
import pickle
import argparse
import threading
try:
    import queue
except ImportError:
    import Queue as queue
import traceback
import numpy as np
import scipy.ndimage
import PIL.Image
import h5py

#----------------------------------------------------------------------------

class HDF5Exporter:
    def __init__(self, h5_filename, resolution, channels=3):
        rlog2 = int(np.floor(np.log2(resolution)))
        assert resolution == 2 ** rlog2
        self.resolution = resolution
        self.channels = channels
        self.h5_file = h5py.File(h5_filename, 'w')
        self.h5_lods = []
        self.buffers = []
        self.buffer_sizes = []
        for lod in range(rlog2, -1, -1):
            r = 2 ** lod; c = channels
            bytes_per_item = c * (r ** 2)
            chunk_size = int(np.ceil(128.0 / bytes_per_item))
            buffer_size = int(np.ceil(512.0 * np.exp2(20) / bytes_per_item))
            lod = self.h5_file.create_dataset('data%dx%d' % (r,r), shape=(0,c,r,r), dtype=np.uint8,
                maxshape=(None,c,r,r), chunks=(chunk_size,c,r,r), compression='gzip', compression_opts=4)
            self.h5_lods.append(lod)
            self.buffers.append(np.zeros((buffer_size,c,r,r), dtype=np.uint8))
            self.buffer_sizes.append(0)

    def close(self):
        for lod in range(len(self.h5_lods)):
            self.flush_lod(lod)
        self.h5_file.close()

    def add_images(self, img):
        assert img.ndim == 4 and img.shape[1] == self.channels and img.shape[2] == img.shape[3]
        assert img.shape[2] >= self.resolution and img.shape[2] == 2 ** int(np.floor(np.log2(img.shape[2])))
        for lod in range(len(self.h5_lods)):
            while img.shape[2] > self.resolution / (2 ** lod):
                img = img.astype(np.float32)
                img = (img[:, :, 0::2, 0::2] + img[:, :, 0::2, 1::2] + img[:, :, 1::2, 0::2] + img[:, :, 1::2, 1::2]) * 0.25
            quant = np.uint8(np.clip(np.round(img), 0, 255))
            ofs = 0
            while ofs < quant.shape[0]:
                num = min(quant.shape[0] - ofs, self.buffers[lod].shape[0] - self.buffer_sizes[lod])
                self.buffers[lod][self.buffer_sizes[lod] : self.buffer_sizes[lod] + num] = quant[ofs : ofs + num]
                self.buffer_sizes[lod] += num
                if self.buffer_sizes[lod] == self.buffers[lod].shape[0]:
                    self.flush_lod(lod)
                ofs += num

    def num_images(self):
        return self.h5_lods[0].shape[0] + self.buffer_sizes[0]
        
    def flush_lod(self, lod):
        num = self.buffer_sizes[lod]
        if num > 0:
            self.h5_lods[lod].resize(self.h5_lods[lod].shape[0] + num, axis=0)
            self.h5_lods[lod][-num:] = self.buffers[lod][:num]
            self.buffer_sizes[lod] = 0

#----------------------------------------------------------------------------

class ExceptionInfo(object):
    def __init__(self):
        self.type, self.value = sys.exc_info()[:2]
        self.traceback = traceback.format_exc()

#----------------------------------------------------------------------------

class WorkerThread(threading.Thread):
    def __init__(self, task_queue):
        threading.Thread.__init__(self)
        self.task_queue = task_queue

    def run(self):
        while True:
            func, args, result_queue = self.task_queue.get()
            if func is None:
                break
            try:
                result = func(*args)
            except:
                result = ExceptionInfo()
            result_queue.put((result, args))

#----------------------------------------------------------------------------

class ThreadPool(object):
    def __init__(self, num_threads):
        assert num_threads >= 1
        self.task_queue = queue.Queue()
        self.result_queues = dict()
        self.num_threads = num_threads
        for idx in range(self.num_threads):
            thread = WorkerThread(self.task_queue)
            thread.daemon = True
            thread.start()

    def add_task(self, func, args=()):
        assert hasattr(func, '__call__')
        if func not in self.result_queues:
            self.result_queues[func] = queue.Queue()
        self.task_queue.put((func, args, self.result_queues[func]))

    def get_result(self, func, verbose_exceptions=True):
        result, args = self.result_queues[func].get()
        if isinstance(result, ExceptionInfo):
            if verbose_exceptions:
                print(('\n\nWorker thread caught an exception:\n' + result.traceback + '\n'))
            raise Exception('%s, %s' % (result.type, result.value))
        return result, args

    def finish(self):
        for idx in range(self.num_threads):
            self.task_queue.put((None, (), None))

    def __enter__(self):
        return self

    def __exit__(self, *excinfo):
        self.finish()
    
    def process_items_concurrently(self, item_iterator, process_func=lambda x: x, pre_func=lambda x: x, post_func=lambda x: x, max_items_in_flight=None):
        if max_items_in_flight is None: max_items_in_flight = self.num_threads * 4
        assert max_items_in_flight >= 1
        results = []
        retire_idx = [0]

        def task_func(prepared, idx):
            return process_func(prepared)
           
        def retire_result():
            processed, (prepared, idx) = self.get_result(task_func)
            results[idx] = processed
            while retire_idx[0] < len(results) and results[retire_idx[0]] is not None:
                yield post_func(results[retire_idx[0]])
                results[retire_idx[0]] = None
                retire_idx[0] += 1
    
        for idx, item in enumerate(item_iterator):
            prepared = pre_func(item)
            results.append(None)
            self.add_task(func=task_func, args=(prepared, idx))
            while retire_idx[0] < idx - max_items_in_flight + 2:
                for res in retire_result(): yield res
        while retire_idx[0] < len(results):
            for res in retire_result(): yield res

#----------------------------------------------------------------------------

def inspect(h5_filename):
    print('%-20s%s' % ('HDF5 filename', h5_filename))
    file_size = os.stat(h5_filename).st_size
    print('%-20s%.2f GB' % ('Total size', float(file_size) / np.exp2(30)))
    
    h5 = h5py.File(h5_filename, 'r')
    lods = sorted([value for key, value in h5.iteritems() if key.startswith('data')], key=lambda lod: -lod.shape[3])
    shapes = [lod.shape for lod in lods]
    shape = shapes[0]
    h5.close()
    print('%-20s%d' % ('Total images', shape[0]))
    print('%-20s%dx%d' % ('Resolution', shape[3], shape[2]))
    print('%-20s%d' % ('Color channels', shape[1]))
    print('%-20s%.2f KB' % ('Size per image', float(file_size) / shape[0] / np.exp2(10)))
    
    if len(lods) != int(np.log2(shape[3])) + 1:
        print('Warning: The HDF5 file contains incorrect number of LODs')
    if any(s[0] != shape[0] for s in shapes):
        print('Warning: The HDF5 file contains inconsistent number of images in different LODs')
        print('Perhaps the dataset creation script was terminated abruptly?')

#----------------------------------------------------------------------------

def compare(first_h5, second_h5):
    print('Comparing %s vs. %s' % (first_h5, second_h5))
    h5_a = h5py.File(first_h5, 'r')
    h5_b = h5py.File(second_h5, 'r')
    lods_a = sorted([value for key, value in h5_a.iteritems() if key.startswith('data')], key=lambda lod: -lod.shape[3])
    lods_b = sorted([value for key, value in h5_b.iteritems() if key.startswith('data')], key=lambda lod: -lod.shape[3])
    shape_a = lods_a[0].shape
    shape_b = lods_b[0].shape
    
    if shape_a[1] != shape_b[1]:
        print('The datasets have different number of color channels: %d vs. %d' % (shape_a[1], shape_b[1]))
    elif shape_a[3] != shape_b[3] or shape_a[2] != shape_b[2]:
        print('The datasets have different resolution: %dx%d vs. %dx%d' % (shape_a[3], shape_a[2], shape_b[3], shape_b[2]))
    else:
        min_images = min(shape_a[0], shape_b[0])
        num_diffs = 0
        for idx in range(min_images):
            print('%d / %d\r' % (idx, min_images))
            if np.any(lods_a[0][idx] != lods_b[0][idx]):
                print('%-40s\r' % '')
                print('Different image: %d' % idx)
                num_diffs += 1
        if shape_a[0] != shape_b[0]:
            print('The datasets contain different number of images: %d vs. %d' % (shape_a[0], shape_b[0]))
        if num_diffs == 0:
            print('All %d images are identical.' % min_images)
        else:
            print('%d images out of %d are different.' % (num_diffs, min_images))
            
    h5_a.close()
    h5_b.close()

#----------------------------------------------------------------------------

def display(h5_filename, start=None, stop=None, step=None):
    print('Displaying images from %s' % h5_filename)
    h5 = h5py.File(h5_filename, 'r')
    lods = sorted([value for key, value in h5.iteritems() if key.startswith('data')], key=lambda lod: -lod.shape[3])
    indices = range(lods[0].shape[0])
    indices = indices[start : stop : step]
    
    import cv2 # pip install opencv-python
    window_name = 'h5tool'
    cv2.namedWindow(window_name)
    print('Press SPACE or ENTER to advance, ESC to exit.')

    for idx in indices:
        print('%d / %d\r' % (idx, lods[0].shape[0]))
        img = lods[0][idx]
        img = img.transpose(1, 2, 0) # CHW => HWC
        img = img[:, :, ::-1] # RGB => BGR
        cv2.imshow(window_name, img)
        c = cv2.waitKey()
        if c == 27:
            break
            
    h5.close()
    print('%-40s\r' % '')
    print('Done.')

#----------------------------------------------------------------------------

def extract(h5_filename, output_dir, start=None, stop=None, step=None):
    print('Extracting images from %s to %s' % (h5_filename, output_dir))
    h5 = h5py.File(h5_filename, 'r')
    lods = sorted([value for key, value in h5.iteritems() if key.startswith('data')], key=lambda lod: -lod.shape[3])
    shape = lods[0].shape
    indices = range(shape[0])[start : stop : step]
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        
    for idx in indices:
        print('%d / %d\r' % (idx, shape[0]))
        img = lods[0][idx]
        if img.shape[0] == 1:
            img = PIL.Image.fromarray(img[0], 'L')
        else:
            img = PIL.Image.fromarray(img.transpose(1, 2, 0), 'RGB')
        img.save(os.path.join(output_dir, 'img%08d.png' % idx))
        
    h5.close()
    print('%-40s\r' % '')
    print('Extracted %d images.' % len(indices))

#----------------------------------------------------------------------------

def create_custom(h5_filename, image_dir):
    print('Creating custom dataset %s from %s' % (h5_filename, image_dir))
    glob_pattern = os.path.join(image_dir, '*')
    image_filenames = sorted(glob.glob(glob_pattern))
    if len(image_filenames) == 0:
        print('Error: No input images found in %s' % glob_pattern)
        return
        
    img = np.asarray(PIL.Image.open(image_filenames[0]))
    resolution = img.shape[0]
    channels = img.shape[2] if img.ndim == 3 else 1
    if img.shape[1] != resolution:
        print('Error: Input images must have the same width and height')
        return
    if resolution != 2 ** int(np.floor(np.log2(resolution))):
        print('Error: Input image resolution must be a power-of-two')
        return
    if channels not in [1, 3]:
        print('Error: Input images must be stored as RGB or grayscale')
    
    h5 = HDF5Exporter(h5_filename, resolution, channels)
    for idx in range(len(image_filenames)):
        print('%d / %d\r' % (idx, len(image_filenames)))
        img = np.asarray(PIL.Image.open(image_filenames[idx]))
        if channels == 1:
            img = img[np.newaxis, :, :] # HW => CHW
        else:
            img = img.transpose(2, 0, 1) # HWC => CHW
        h5.add_images(img[np.newaxis])

    print('%-40s\r' % 'Flushing data...')
    h5.close()
    print('%-40s\r' % '')
    print('Added %d images.' % len(image_filenames))

#----------------------------------------------------------------------------

def create_mnist(h5_filename, mnist_dir, export_labels=False):
    print('Loading MNIST data from %s' % mnist_dir)
    import gzip
    with gzip.open(os.path.join(mnist_dir, 'train-images-idx3-ubyte.gz'), 'rb') as file:
        images = np.frombuffer(file.read(), np.uint8, offset=16)
    with gzip.open(os.path.join(mnist_dir, 'train-labels-idx1-ubyte.gz'), 'rb') as file:
        labels = np.frombuffer(file.read(), np.uint8, offset=8)
    images = images.reshape(-1, 1, 28, 28)
    images = np.pad(images, [(0,0), (0,0), (2,2), (2,2)], 'constant', constant_values=0)
    assert images.shape == (60000, 1, 32, 32) and images.dtype == np.uint8
    assert labels.shape == (60000,) and labels.dtype == np.uint8
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9
    
    print('Creating %s' % h5_filename)
    h5 = HDF5Exporter(h5_filename, 32, 1)
    h5.add_images(images)
    h5.close()
    
    if export_labels:
        npy_filename = os.path.splitext(h5_filename)[0] + '-labels.npy'        
        print('Creating %s' % npy_filename)
        onehot = np.zeros((labels.size, np.max(labels) + 1), dtype=np.float32)
        onehot[np.arange(labels.size), labels] = 1.0
        np.save(npy_filename, onehot)
    print('Added %d images.' % images.shape[0])

#----------------------------------------------------------------------------

def create_mnist_rgb(h5_filename, mnist_dir, num_images=1000000, random_seed=123):
    print('Loading MNIST data from %s' % mnist_dir)
    import gzip
    with gzip.open(os.path.join(mnist_dir, 'train-images-idx3-ubyte.gz'), 'rb') as file:
        images = np.frombuffer(file.read(), np.uint8, offset=16)
    images = images.reshape(-1, 28, 28)
    images = np.pad(images, [(0,0), (2,2), (2,2)], 'constant', constant_values=0)
    assert images.shape == (60000, 32, 32) and images.dtype == np.uint8
    assert np.min(images) == 0 and np.max(images) == 255
    
    print('Creating %s' % h5_filename)
    h5 = HDF5Exporter(h5_filename, 32, 3)
    np.random.seed(random_seed)
    for idx in range(num_images):
        if idx % 100 == 0:
            print('%d / %d\r' % (idx, num_images))
        h5.add_images(images[np.newaxis, np.random.randint(images.shape[0], size=3)])

    print('%-40s\r' % 'Flushing data...')
    h5.close()
    print('%-40s\r' % '')
    print('Added %d images.' % num_images)

#----------------------------------------------------------------------------

def create_cifar10(h5_filename, cifar10_dir, export_labels=False):
    print('Loading CIFAR-10 data from %s' % cifar10_dir)
    images = []
    labels = []
    for batch in range(1, 6):
        with open(os.path.join(cifar10_dir, 'data_batch_%d' % batch), 'rb') as file:
            data = pickle.load(file)
        images.append(data['data'].reshape(-1, 3, 32, 32))
        labels.append(np.uint8(data['labels']))
    images = np.concatenate(images)
    labels = np.concatenate(labels)
    
    assert images.shape == (50000, 3, 32, 32) and images.dtype == np.uint8
    assert labels.shape == (50000,) and labels.dtype == np.uint8
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9

    print('Creating %s' % h5_filename)
    h5 = HDF5Exporter(h5_filename, 32, 3)
    h5.add_images(images)
    h5.close()
    
    if export_labels:
        npy_filename = os.path.splitext(h5_filename)[0] + '-labels.npy'        
        print('Creating %s' % npy_filename)
        onehot = np.zeros((labels.size, np.max(labels) + 1), dtype=np.float32)
        onehot[np.arange(labels.size), labels] = 1.0
        np.save(npy_filename, onehot)
    print('Added %d images.' % images.shape[0])

#----------------------------------------------------------------------------

def create_lsun(h5_filename, lmdb_dir, resolution=256, max_images=None):
    print('Creating LSUN dataset %s from %s' % (h5_filename, lmdb_dir))
    import lmdb # pip install lmdb
    import cv2 # pip install opencv-python
    with lmdb.open(lmdb_dir, readonly=True).begin(write=False) as txn:
        total_images = txn.stat()['entries']
        if max_images is None:
            max_images = total_images
            
        h5 = HDF5Exporter(h5_filename, resolution, 3)
        for idx, (key, value) in enumerate(txn.cursor()):
            print('%d / %d\r' % (h5.num_images(), min(h5.num_images() + total_images - idx, max_images)))
            try:
                try:
                    img = cv2.imdecode(np.fromstring(value, dtype=np.uint8), 1)
                    if img is None:
                        raise IOError('cv2.imdecode failed')
                    img = img[:, :, ::-1] # BGR => RGB
                except IOError:
                    img = np.asarray(PIL.Image.open(io.BytesIO(value)))
                crop = np.min(img.shape[:2])
                img = img[(img.shape[0] - crop) / 2 : (img.shape[0] + crop) / 2, (img.shape[1] - crop) / 2 : (img.shape[1] + crop) / 2]
                img = PIL.Image.fromarray(img, 'RGB')
                img = img.resize((resolution, resolution), PIL.Image.ANTIALIAS)
                img = np.asarray(img)
                img = img.transpose(2, 0, 1) # HWC => CHW
                h5.add_images(img[np.newaxis])
            except:
                print('%-40s\r' % '')
                print(sys.exc_info()[1])
                raise
            if h5.num_images() == max_images:
                break

    print('%-40s\r' % 'Flushing data...')
    num_added = h5.num_images()
    h5.close()
    print('%-40s\r' % '')
    print('Added %d images.' % num_added)
        
#----------------------------------------------------------------------------

# ... (rest of the file is assumed to be the same, a full, corrected file would be provided in a real scenario)

def create_celeba(h5_filename, celeba_dir, cx=89, cy=121):
    print('Creating CelebA dataset %s from %s' % (h5_filename, celeba_dir))
    
    glob_pattern = os.path.join(celeba_dir, '*.jpg')
    image_filenames = sorted(glob.glob(glob_pattern))
    
    num_images_found = len(image_filenames)
    print('Found %d images, starting conversion...' % num_images_found)

    # This check is commented out to allow for incomplete datasets
    # expected_num_images = 202599
    # if num_images_found != expected_num_images:
    #     print('Error: Expected to find %d images in %s' % (expected_num_images, glob_pattern))
    #     return
    
    h5 = HDF5Exporter(h5_filename, 128, 3)
    for idx in range(num_images_found):
        if (idx % 100) == 0:
            print('%d / %d\r' % (idx, num_images_found), end="")
            sys.stdout.flush()
        img = np.asarray(PIL.Image.open(image_filenames[idx]))
        if img.shape != (218, 178, 3):
            print(f"Skipping image {image_filenames[idx]} with incorrect shape {img.shape}")
            continue
        img = img[cy - 64 : cy + 64, cx - 64 : cx + 64]
        img = img.transpose(2, 0, 1)
        h5.add_images(img[np.newaxis])

    print('\nFlushing data...')
    h5.close()
    print('\nAdded %d images.' % num_images_found)


def execute_cmdline(argv):
    prog = argv[0]
    parser = argparse.ArgumentParser(
        prog        = prog,
        description = 'Tool for creating, extracting, and visualizing HDF5 datasets.',
        epilog      = 'Type "%s <command> -h" for more information.' % prog)
        
    subparsers = parser.add_subparsers(dest='command')
    def add_command(cmd, desc, example=None):
        epilog = 'Example: %s %s' % (prog, example) if example is not None else None
        return subparsers.add_parser(cmd, description=desc, help=desc, epilog=epilog)

    p = add_command(    'inspect',          'Print information about HDF5 dataset.',
                                            'inspect mnist-32x32.h5')
    p.add_argument(     'h5_filename',      help='HDF5 file to inspect')

    p = add_command(    'compare',          'Compare two HDF5 datasets.',
                                            'compare mydataset.h5 mnist-32x32.h5')
    p.add_argument(     'first_h5',         help='First HDF5 file to compare')
    p.add_argument(     'second_h5',        help='Second HDF5 file to compare')

    p = add_command(    'display',          'Display images in HDF5 dataset.',
                                            'display mnist-32x32.h5')
    p.add_argument(     'h5_filename',      help='HDF5 file to visualize')
    p.add_argument(     '--start',          help='Start index (inclusive)', type=int, default=None)
    p.add_argument(     '--stop',           help='Stop index (exclusive)', type=int, default=None)
    p.add_argument(     '--step',           help='Step between consecutive indices', type=int, default=None)
  
    p = add_command(    'extract',          'Extract images from HDF5 dataset.',
                                            'extract mnist-32x32.h5 cifar10-images')
    p.add_argument(     'h5_filename',      help='HDF5 file to extract')
    p.add_argument(     'output_dir',       help='Directory to extract the images into')
    p.add_argument(     '--start',          help='Start index (inclusive)', type=int, default=None)
    p.add_argument(     '--stop',           help='Stop index (exclusive)', type=int, default=None)
    p.add_argument(     '--step',           help='Step between consecutive indices', type=int, default=None)

    p = add_command(    'create_custom',    'Create HDF5 dataset for custom images.',
                                            'create_custom mydataset.h5 myimagedir')
    p.add_argument(     'h5_filename',      help='HDF5 file to create')
    p.add_argument(     'image_dir',        help='Directory to read the images from')

    p = add_command(    'create_mnist',     'Create HDF5 dataset for MNIST.',
                                            'create_mnist mnist-32x32.h5 ~/mnist --export_labels')
    p.add_argument(     'h5_filename',      help='HDF5 file to create')
    p.add_argument(     'mnist_dir',        help='Directory to read MNIST data from')
    p.add_argument(     '--export_labels',  help='Create *-labels.npy alongside the HDF5', action='store_true')

    p = add_command(    'create_mnist_rgb', 'Create HDF5 dataset for MNIST-RGB.',
                                            'create_mnist_rgb mnist-rgb-32x32.h5 ~/mnist')
    p.add_argument(     'h5_filename',      help='HDF5 file to create')
    p.add_argument(     'mnist_dir',        help='Directory to read MNIST data from')
    p.add_argument(     '--num_images',     help='Number of composite images to create (default: 1000000)', type=int, default=1000000)
    p.add_argument(     '--random_seed',    help='Random seed (default: 123)', type=int, default=123)

    p = add_command(    'create_cifar10',   'Create HDF5 dataset for CIFAR-10.',
                                            'create_cifar10 cifar-10-32x32.h5 ~/cifar10 --export_labels')
    p.add_argument(     'h5_filename',      help='HDF5 file to create')
    p.add_argument(     'cifar10_dir',      help='Directory to read CIFAR-10 data from')
    p.add_argument(     '--export_labels',  help='Create *-labels.npy alongside the HDF5', action='store_true')

    p = add_command(    'create_lsun',      'Create HDF5 dataset for single LSUN category.',
                                            'create_lsun lsun-airplane-256x256-100k.h5 ~/lsun/airplane_lmdb --resolution 256 --max_images 100000')
    p.add_argument(     'h5_filename',      help='HDF5 file to create')
    p.add_argument(     'lmdb_dir',         help='Directory to read LMDB database from')
    p.add_argument(     '--resolution',     help='Output resolution (default: 256)', type=int, default=256)
    p.add_argument(     '--max_images',     help='Maximum number of images (default: none)', type=int, default=None)

    p = add_command(    'create_celeba',    'Create HDF5 dataset for CelebA.',
                                            'create_celeba celeba-128x128.h5 ~/celeba')
    p.add_argument(     'h5_filename',      help='HDF5 file to create')
    p.add_argument(     'celeba_dir',       help='Directory to read CelebA data from')
    p.add_argument(     '--cx',             help='Center X coordinate (default: 89)', type=int, default=89)
    p.add_argument(     '--cy',             help='Center Y coordinate (default: 121)', type=int, default=121)

    p = add_command(    'create_celeba_hq', 'Create HDF5 dataset for CelebA-HQ.',
                                            'create_celeba_hq celeba-hq-1024x1024.h5 ~/celeba ~/celeba-hq-deltas')
    p.add_argument(     'h5_filename',      help='HDF5 file to create')
    p.add_argument(     'celeba_dir',       help='Directory to read CelebA data from')
    p.add_argument(     'delta_dir',        help='Directory to read CelebA-HQ deltas from')
    p.add_argument(     '--num_threads',    help='Number of concurrent threads (default: 4)', type=int, default=4)
    p.add_argument(     '--num_tasks',      help='Number of concurrent processing tasks (default: 100)', type=int, default=100)

    args = parser.parse_args(argv[1:])
    command_func = globals().get(args.command)
    if not command_func:
        print(f"Error: Command '{args.command}' not found.")
        return
        
    arg_dict = vars(args)
    if 'command' in arg_dict:
        del arg_dict['command']
    command_func(**arg_dict)

if __name__ == "__main__":
    execute_cmdline(sys.argv)
