import torch
import torchvision.transforms.functional as F
from PIL import Image
import warnings
import math
import random
import numpy as np


class ToNumpy:

    def __call__(self, img):
        if type(img) == Image.Image:
            np_img = np.array(img, dtype=np.uint8)
            if np_img.ndim < 3:
                np_img = np.expand_dims(np_img, axis=-1)
            np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        elif type(img) == np.ndarray:
            np_img = img
        else:
            raise ValueError(f"img must be of type PIL.Image.Image or "
                             f"np.ndarray. Provided {type(img)}.")
        return np_img


class ToTensor:

    def __init__(self, dtype=torch.float32):
        self.dtype = dtype

    def __call__(self, img):
        if type(img) == Image.Image:
            img = np.array(img, dtype=np.uint8)
        if type(img) == np.ndarray:
            if img.ndim < 3:
                img = np.expand_dims(img, axis=-1)
            img = np.rollaxis(img, 2)  # HWC to CHW
            img = torch.from_numpy(img)
        return img.to(dtype=self.dtype)


try:  # torchvision version >= 9.1
    InterpolationPkg = F.InterpolationMode
    interpolation_pkg_name = 'torchvision.InterpolationMode'

except AttributeError:  # torchvision version < 9.1
    InterpolationPkg = Image
    interpolation_pkg_name = 'PIL.Image'

_pil_interpolation_to_str = {
    InterpolationPkg.NEAREST: f'{interpolation_pkg_name}.NEAREST',
    InterpolationPkg.BILINEAR: f'{interpolation_pkg_name}.BILINEAR',
    InterpolationPkg.BICUBIC: f'{interpolation_pkg_name}.BICUBIC',
    InterpolationPkg.LANCZOS: f'{interpolation_pkg_name}.LANCZOS',
    InterpolationPkg.HAMMING: f'{interpolation_pkg_name}.HAMMING',
    InterpolationPkg.BOX: f'{interpolation_pkg_name}.BOX',
}


def _pil_interp(method):
    if method == 'bicubic':
        return InterpolationPkg.BICUBIC
    elif method == 'lanczos':
        return InterpolationPkg.LANCZOS
    elif method == 'hamming':
        return InterpolationPkg.HAMMING
    elif method == 'nearest':
        return InterpolationPkg.NEAREST
    else:
        # default bilinear
        return InterpolationPkg.BILINEAR


_RANDOM_INTERPOLATION = (InterpolationPkg.BILINEAR, InterpolationPkg.BICUBIC)


class RandomResizedCropAndInterpolation:
    """Crop the given PIL Image to random size and aspect ratio with random interpolation.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation='bilinear'):
        if isinstance(size, (list, tuple)):
            self.size = tuple(size)
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        if interpolation == 'random':
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = _pil_interp(interpolation)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        h_img = img.size(0) if type(img) == torch.Tensor else img.size[0]
        w_img = img.size(1) if type(img) == torch.Tensor else img.size[1]

        area = h_img * w_img

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= h_img and h <= w_img:
                i = random.randint(0, w_img - h)
                j = random.randint(0, h_img - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = h_img / w_img
        if in_ratio < min(ratio):
            w = h_img
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = w_img
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = h_img
            h = w_img
        i = (w_img - h) // 2
        j = (h_img - w) // 2
        return i, j, h, w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        return F.resized_crop(img, i, j, h, w, self.size, interpolation)

    def __repr__(self):
        if isinstance(self.interpolation, (tuple, list)):
            interpolate_str = ' '.join([_pil_interpolation_to_str[x] for x in self.interpolation])
        else:
            interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


