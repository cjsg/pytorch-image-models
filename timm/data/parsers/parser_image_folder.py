""" A dataset parser that reads images from folders

Folders are scannerd recursively to find image files. Labels are based
on the folder hierarchy, just leaf folders by default.

Hacked together by / Copyright 2020 Ross Wightman
"""
import os
import logging
from torch import save, load

from timm.utils.misc import natural_key

from .parser import Parser
from .class_map import load_class_map
from .constants import IMG_EXTENSIONS


_logger = logging.getLogger(__name__)

def find_images_and_targets(folder, types=IMG_EXTENSIONS, class_to_idx=None, leaf_name_only=True, sort=True):
    labels = []
    filenames = []
    for root, subdirs, files in os.walk(folder, topdown=False, followlinks=True):
        rel_path = os.path.relpath(root, folder) if (root != folder) else ''
        label = os.path.basename(rel_path) if leaf_name_only else rel_path.replace(os.path.sep, '_')
        for f in files:
            base, ext = os.path.splitext(f)
            if ext.lower() in types:
                filenames.append(os.path.join(root, f))
                labels.append(label)
    if class_to_idx is None:
        # building class index
        unique_labels = set(labels)
        sorted_labels = list(sorted(unique_labels, key=natural_key))
        class_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}
    images_and_targets = [(f, class_to_idx[l]) for f, l in zip(filenames, labels) if l in class_to_idx]
    if sort:
        images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
    return images_and_targets, class_to_idx


class ParserImageFolder(Parser):

    def __init__(
            self,
            root,
            class_map=''):
        super().__init__()

        self.root = root
        class_to_idx = None
        path_to_samples = os.path.join(root, 'samples.pth')
        if os.path.exists(path_to_samples):
            _logger.info(
                f'Using pre-computed (sample, ix) files samples.pth found in {root}')
            self.samples = load(path_to_samples)
        else:
            _logger.info(
                f'No file samples.pth in folder {root}.\n'
                f'I will reccursively traverse the folder to construct and save it.')
            if class_map:
                class_to_idx = load_class_map(class_map, root)
            self.samples, self.class_to_idx = find_images_and_targets(root, class_to_idx=class_to_idx)
            if len(self.samples) == 0:
                raise RuntimeError(
                    f'Found 0 images in subfolders of {root}. Supported image extensions are {", ".join(IMG_EXTENSIONS)}')
            try:
                save(self.samples, path_to_samples)
            except PermissionError:
                _logger.info(
                    f'Could not save (sample, ix)-dictionnary, because I have no writing permissions for {root}'
                )

    def __getitem__(self, index):
        path, target = self.samples[index]
        return open(path, 'rb'), target

    def __len__(self):
        return len(self.samples)

    def _filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename
