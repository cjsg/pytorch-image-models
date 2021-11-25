import os
import torchvision
import logging

from .dataset import IterableImageDataset, ImageDataset
from lmdbdataset import LMDBIterDataset, LMDBDataset

_logger = logging.getLogger(__name__)

def _search_split(root, split):
    # look for sub-folder with name of split in root and use that if it exists
    split_name = split.split('[')[0]
    try_root = os.path.join(root, split_name)
    if os.path.exists(try_root):
        return try_root
    if split_name == 'validation':
        try_root = os.path.join(root, 'val')
        if os.path.exists(try_root):
            return try_root
    return root


def create_dataset(name, root, split='validation', search_split=True, is_training=False,
                   batch_size=None, distributed=False, map_style_lmdb=False, log_info=True, **kwargs):
    name = name.lower()
    if (not (name.startswith('lmdb') or 'lmdb' in root)) and map_style_lmdb:
        # raise ValueError('`map_style_lmdb` can only be set to True when using an lmdb dataset')
        if log_info:
            _logger.warn('`map_style_lmdb` can only be set to True when using an lmdb dataset')

    if name.startswith('tfds'):
        if log_info:
            _logger.info('Using tfds-based IterableImageDataset')
        ds = IterableImageDataset(
            root, parser=name, split=split, is_training=is_training, batch_size=batch_size, **kwargs)
    elif name == 'cifar10':
        if log_info:
            _logger.info('Using CIFAR-10 dataset')
        ds = torchvision.datasets.CIFAR10(root=root, train=is_training, download=False)
    elif name.startswith('lmdb') or 'lmdb' in root:
        kwargs.pop('repeats', 0)  # FIXME currently only Iterable dataset support the repeat multiplier
        split_name = 'train' if (split == 'train') else 'val'
        if map_style_lmdb:
            if log_info:
                _logger.info(f'Using **map-style** LMDB dataset found at {root}')
            ds = LMDBDataset(
                root, split_name, img_type='raw', return_type='raw', **kwargs)
        else:
            if log_info:
                _logger.info(f'Using **iterable** LMDB dataset found at {root}')
            ds = LMDBIterDataset(
                root, split_name, img_type='raw', return_type='raw',
                distributed=distributed, **kwargs)
    # elif name == 'imgnet':  # CJ's hack to accelearate loading
    #     folder = '/train' if is_training else '/val'
    #     ds = torchvision.datasets.ImageFolder(root + folder)
    else:
        # FIXME support more advance split cfg for ImageFolder/Tar datasets in the future
        if log_info:
            _logger.info(f'Using ImageDataset (usually folder of JPEGs) found at {root}')
        kwargs.pop('repeats', 0)  # FIXME currently only Iterable dataset support the repeat multiplier
        if search_split and os.path.isdir(root):
            root = _search_split(root, split)
        ds = ImageDataset(root, parser=name, **kwargs)
    return ds
