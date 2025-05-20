import math
import numbers
import random
import warnings
from typing import List, Sequence, Tuple, Union

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
try:
    from torchvision.transforms.functional import InterpolationMode
    has_interpolation_mode = True
except ImportError:
    has_interpolation_mode = False
from PIL import Image
import numpy as np

__all__ = [
    "ToNumpy", "ToTensor", "str_to_interp_mode", "str_to_pil_interp", "interp_mode_to_str",
    "RandomResizedCropAndInterpolation", "CenterCropOrPad", "center_crop_or_pad", "crop_or_pad",
    "ColorMode", "StatsPrinter",
    "RandomCropOrPad", "RandomPad", "ResizeKeepRatio", "TrimBorder", "MaybeToTensor", "MaybePILToTensor"
]


class ToNumpy:

    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        return np_img


class ToTensor:
    """ ToTensor with no rescaling of values"""
    def __init__(self, dtype=torch.float32):
        self.dtype = dtype

    def __call__(self, pil_img):
        return F.pil_to_tensor(pil_img).to(dtype=self.dtype)


class MaybeToTensor(transforms.ToTensor):
    """Convert a PIL Image or ndarray to tensor if it's not already one.
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, pic) -> torch.Tensor:
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return F.to_tensor(pic)   # convert from [0, 255] to [0, 1]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class MaybePILToTensor:
    """Convert a PIL Image to a tensor of the same type - this does not scale values.
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, pic):
        """
        Note: A deep copy of the underlying array is performed.

        Args:
            pic (PIL Image): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return F.pil_to_tensor(pic)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# Pillow is deprecating the top-level resampling attributes (e.g., Image.BILINEAR) in
# favor of the Image.Resampling enum. The top-level resampling attributes will be
# removed in Pillow 10.
if hasattr(Image, "Resampling"):
    _pil_interpolation_to_str = {
        Image.Resampling.NEAREST: 'nearest',
        Image.Resampling.BILINEAR: 'bilinear',
        Image.Resampling.BICUBIC: 'bicubic',
        Image.Resampling.BOX: 'box',
        Image.Resampling.HAMMING: 'hamming',
        Image.Resampling.LANCZOS: 'lanczos',
    }
else:
    _pil_interpolation_to_str = {
        Image.NEAREST: 'nearest',
        Image.BILINEAR: 'bilinear',
        Image.BICUBIC: 'bicubic',
        Image.BOX: 'box',
        Image.HAMMING: 'hamming',
        Image.LANCZOS: 'lanczos',
    }

_str_to_pil_interpolation = {b: a for a, b in _pil_interpolation_to_str.items()}


if has_interpolation_mode:
    _torch_interpolation_to_str = {
        InterpolationMode.NEAREST: 'nearest',
        InterpolationMode.BILINEAR: 'bilinear',
        InterpolationMode.BICUBIC: 'bicubic',
        InterpolationMode.BOX: 'box',
        InterpolationMode.HAMMING: 'hamming',
        InterpolationMode.LANCZOS: 'lanczos',
    }
    _str_to_torch_interpolation = {b: a for a, b in _torch_interpolation_to_str.items()}
else:
    _pil_interpolation_to_torch = {}
    _torch_interpolation_to_str = {}


def str_to_pil_interp(mode_str):
    return _str_to_pil_interpolation[mode_str]


def str_to_interp_mode(mode_str):
    if has_interpolation_mode:
        return _str_to_torch_interpolation[mode_str]
    else:
        return _str_to_pil_interpolation[mode_str]


def interp_mode_to_str(mode):
    if has_interpolation_mode:
        return _torch_interpolation_to_str[mode]
    else:
        return _pil_interpolation_to_str[mode]


_RANDOM_INTERPOLATION = (str_to_interp_mode('bilinear'), str_to_interp_mode('bicubic'))


def _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size."):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


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

    def __init__(
            self,
            size,
            scale=(0.08, 1.0),
            ratio=(3. / 4., 4. / 3.),
            interpolation='bilinear',
    ):
        if isinstance(size, (list, tuple)):
            self.size = tuple(size)
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        if interpolation == 'random':
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = str_to_interp_mode(interpolation)
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
        img_w, img_h = F.get_image_size(img)
        area = img_w * img_h

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            target_w = int(round(math.sqrt(target_area * aspect_ratio)))
            target_h = int(round(math.sqrt(target_area / aspect_ratio)))
            if target_w <= img_w and target_h <= img_h:
                i = random.randint(0, img_h - target_h)
                j = random.randint(0, img_w - target_w)
                return i, j, target_h, target_w

        # Fallback to central crop
        in_ratio = img_w / img_h
        if in_ratio < min(ratio):
            target_w = img_w
            target_h = int(round(target_w / min(ratio)))
        elif in_ratio > max(ratio):
            target_h = img_h
            target_w = int(round(target_h * max(ratio)))
        else:  # whole image
            target_w = img_w
            target_h = img_h
        i = (img_h - target_h) // 2
        j = (img_w - target_w) // 2
        return i, j, target_h, target_w

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
            interpolate_str = ' '.join([interp_mode_to_str(x) for x in self.interpolation])
        else:
            interpolate_str = interp_mode_to_str(self.interpolation)
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


def center_crop_or_pad(
        img: torch.Tensor,
        output_size: Union[int, List[int]],
        fill: Union[int, Tuple[int, int, int]] = 0,
        padding_mode: str = 'constant',
) -> torch.Tensor:
    """Center crops and/or pads the given image.

    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        img (PIL Image or Tensor): Image to be cropped.
        output_size (sequence or int): (height, width) of the crop box. If int or sequence with single int,
            it is used for both directions.
        fill (int, Tuple[int]): Padding color

    Returns:
        PIL Image or Tensor: Cropped image.
    """
    output_size = _setup_size(output_size)
    crop_height, crop_width = output_size
    _, image_height, image_width = F.get_dimensions(img)

    if crop_width > image_width or crop_height > image_height:
        padding_ltrb = [
            (crop_width - image_width) // 2 if crop_width > image_width else 0,
            (crop_height - image_height) // 2 if crop_height > image_height else 0,
            (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
            (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
        ]
        img = F.pad(img, padding_ltrb, fill=fill, padding_mode=padding_mode)
        _, image_height, image_width = F.get_dimensions(img)
        if crop_width == image_width and crop_height == image_height:
            return img

    crop_top = int(round((image_height - crop_height) / 2.0))
    crop_left = int(round((image_width - crop_width) / 2.0))
    return F.crop(img, crop_top, crop_left, crop_height, crop_width)


class CenterCropOrPad(torch.nn.Module):
    """Crops the given image at the center.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
    """

    def __init__(
            self,
            size: Union[int, List[int]],
            fill: Union[int, Tuple[int, int, int]] = 0,
            padding_mode: str = 'constant',
    ):
        super().__init__()
        self.size = _setup_size(size)
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        return center_crop_or_pad(img, self.size, fill=self.fill, padding_mode=self.padding_mode)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


def crop_or_pad(
        img: torch.Tensor,
        top: int,
        left: int,
        height: int,
        width: int,
        fill: Union[int, Tuple[int, int, int]] = 0,
        padding_mode: str = 'constant',
) -> torch.Tensor:
    """ Crops and/or pads image to meet target size, with control over fill and padding_mode.
    """
    _, image_height, image_width = F.get_dimensions(img)
    right = left + width
    bottom = top + height
    if left < 0 or top < 0 or right > image_width or bottom > image_height:
        padding_ltrb = [
            max(-left + min(0, right), 0),
            max(-top + min(0, bottom), 0),
            max(right - max(image_width, left), 0),
            max(bottom - max(image_height, top), 0),
        ]
        img = F.pad(img, padding_ltrb, fill=fill, padding_mode=padding_mode)

    top = max(top, 0)
    left = max(left, 0)
    return F.crop(img, top, left, height, width)


class RandomCropOrPad(torch.nn.Module):
    """ Crop and/or pad image with random placement within the crop or pad margin.
    """

    def __init__(
            self,
            size: Union[int, List[int]],
            fill: Union[int, Tuple[int, int, int]] = 0,
            padding_mode: str = 'constant',
    ):
        super().__init__()
        self.size = _setup_size(size)
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, size):
        _, image_height, image_width = F.get_dimensions(img)
        delta_height = image_height - size[0]
        delta_width = image_width - size[1]
        top = int(math.copysign(random.randint(0, abs(delta_height)), delta_height))
        left = int(math.copysign(random.randint(0, abs(delta_width)), delta_width))
        return top, left

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        top, left = self.get_params(img, self.size)
        return crop_or_pad(
            img,
            top=top,
            left=left,
            height=self.size[0],
            width=self.size[1],
            fill=self.fill,
            padding_mode=self.padding_mode,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class RandomPad:
    def __init__(self, input_size, fill=0):
        self.input_size = input_size
        self.fill = fill

    @staticmethod
    def get_params(img, input_size):
        width, height = F.get_image_size(img)
        delta_width = max(input_size[1] - width, 0)
        delta_height = max(input_size[0] - height, 0)
        pad_left = random.randint(0, delta_width)
        pad_top = random.randint(0, delta_height)
        pad_right = delta_width - pad_left
        pad_bottom = delta_height - pad_top
        return pad_left, pad_top, pad_right, pad_bottom

    def __call__(self, img):
        padding = self.get_params(img, self.input_size)
        img = F.pad(img, padding, self.fill)
        return img


class ResizeKeepRatio:
    """ Resize and Keep Aspect Ratio
    """

    def __init__(
            self,
            size,
            longest=0.,
            interpolation='bilinear',
            random_scale_prob=0.,
            random_scale_range=(0.85, 1.05),
            random_scale_area=False,
            random_aspect_prob=0.,
            random_aspect_range=(0.9, 1.11),
    ):
        """

        Args:
            size:
            longest:
            interpolation:
            random_scale_prob:
            random_scale_range:
            random_scale_area:
            random_aspect_prob:
            random_aspect_range:
        """
        if isinstance(size, (list, tuple)):
            self.size = tuple(size)
        else:
            self.size = (size, size)
        if interpolation == 'random':
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = str_to_interp_mode(interpolation)
        self.longest = float(longest)
        self.random_scale_prob = random_scale_prob
        self.random_scale_range = random_scale_range
        self.random_scale_area = random_scale_area
        self.random_aspect_prob = random_aspect_prob
        self.random_aspect_range = random_aspect_range

    @staticmethod
    def get_params(
            img,
            target_size,
            longest,
            random_scale_prob=0.,
            random_scale_range=(1.0, 1.33),
            random_scale_area=False,
            random_aspect_prob=0.,
            random_aspect_range=(0.9, 1.11)
    ):
        """Get parameters
        """
        img_h, img_w = img_size = F.get_dimensions(img)[1:]
        target_h, target_w = target_size
        ratio_h = img_h / target_h
        ratio_w = img_w / target_w
        ratio = max(ratio_h, ratio_w) * longest + min(ratio_h, ratio_w) * (1. - longest)

        if random_scale_prob > 0 and random.random() < random_scale_prob:
            ratio_factor = random.uniform(random_scale_range[0], random_scale_range[1])
            if random_scale_area:
                # make ratio factor equivalent to RRC area crop where < 1.0 = area zoom,
                # otherwise like affine scale where < 1.0 = linear zoom out
                ratio_factor = 1. / math.sqrt(ratio_factor)
            ratio_factor = (ratio_factor, ratio_factor)
        else:
            ratio_factor = (1., 1.)

        if random_aspect_prob > 0 and random.random() < random_aspect_prob:
            log_aspect = (math.log(random_aspect_range[0]), math.log(random_aspect_range[1]))
            aspect_factor = math.exp(random.uniform(*log_aspect))
            aspect_factor = math.sqrt(aspect_factor)
            # currently applying random aspect adjustment equally to both dims,
            # could change to keep output sizes above their target where possible
            ratio_factor = (ratio_factor[0] / aspect_factor, ratio_factor[1] * aspect_factor)

        size = [round(x * f / ratio) for x, f in zip(img_size, ratio_factor)]
        return size

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Resized, padded to at least target size, possibly cropped to exactly target size
        """
        size = self.get_params(
            img, self.size, self.longest,
            self.random_scale_prob, self.random_scale_range, self.random_scale_area,
            self.random_aspect_prob, self.random_aspect_range
        )
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        img = F.resize(img, size, interpolation)
        return img

    def __repr__(self):
        if isinstance(self.interpolation, (tuple, list)):
            interpolate_str = ' '.join([interp_mode_to_str(x) for x in self.interpolation])
        else:
            interpolate_str = interp_mode_to_str(self.interpolation)
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += f', interpolation={interpolate_str}'
        format_string += f', longest={self.longest:.3f}'
        format_string += f', random_scale_prob={self.random_scale_prob:.3f}'
        format_string += f', random_scale_range=(' \
                         f'{self.random_scale_range[0]:.3f}, {self.random_aspect_range[1]:.3f})'
        format_string += f', random_aspect_prob={self.random_aspect_prob:.3f}'
        format_string += f', random_aspect_range=(' \
                         f'{self.random_aspect_range[0]:.3f}, {self.random_aspect_range[1]:.3f}))'
        return format_string


class TrimBorder(torch.nn.Module):

    def __init__(
            self,
            border_size: int,
    ):
        super().__init__()
        self.border_size = border_size

    def forward(self, img):
        w, h = F.get_image_size(img)
        top = left = self.border_size
        top = min(top, h)
        left = min(left, h)
        height = max(0, h - 2 * self.border_size)
        width = max(0, w - 2 * self.border_size)
        return F.crop(img, top, left, height, width)


class ColorMode(torch.nn.Module):
    """Change color mode from RGB to something else
    If the image is torch Tensor, it is expected
    to have [..., C, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        color_mode (str): Either HSL, HSV, LAB, YUV, or YCbCr. Case insensitive
        use_numpy (bool): If True, expect numpy uint8 input, otherwise PyTorch tensor float32 in range [0, 1]
    """

    def __init__(
            self,
            mode: str,
            use_numpy: bool = False,
    ):
        super().__init__()
        self.mode = str(mode).lower()
        self.use_numpy = use_numpy
        assert mode in ["hsl", "hsv", "lab", "yuv", "ycbcr"]

    def stack(self, channels):
        if self.use_numpy:
            return np.stack(channels, axis=-3).clip(0, 255).astype(np.uint8)
        else:
            return torch.stack(channels, dim=-3)

    def forward(self, img):
        """
        Args:
            img (Tensor): Image in RGB, value range [0, 1] if use_numpy=False, otherwise [0, 255] if use_numpy=True

        Returns:
            Tensor: Image convert to other color mode, value range [0, 1] if use_numpy=False, otherwise [0, 255] if use_numpy=True
        """
        assert img.shape[-3] == 3, "Image must have 3 channels in NCHW format"
        R = img[..., 0, :, :]
        G = img[..., 1, :, :]
        B = img[..., 2, :, :]
        if self.mode in ["hsv", "hsl"]:
            V = img.max(axis=-3).values   # max = value
            m = img.min(axis=-3).values
            delta = V - m    # = chroma = max - min
            S = torch.where(V == 0, torch.tensor(0), delta / V)
            H = torch.where(delta == 0, torch.tensor(0), torch.where(V == R, (G-B)/delta % 6, torch.where(V == G, (B-R)/delta + 2, (R-G)/delta + 4))) * 60
            if self.use_numpy:
                H *= 255/360
                S *= 255
            else:
                H /= 360
            if self.mode == "hsv":
                img = self.stack([H, S, V])
            else:
                img = self.stack([H, S, (V+m)/2])
            return img
        elif self.mode == "lab":
            if self.use_numpy:
                R = R / 255.0
                G = G / 255.0
                B = B / 255.0
            # non-linear transform of RGB
            R_ = torch.where(R > 0.04045, ((R + 0.055) / 1.055) ** 2.4, R / 12.92)
            G_ = torch.where(G > 0.04045, ((G + 0.055) / 1.055) ** 2.4, G / 12.92)
            B_ = torch.where(B > 0.04045, ((B + 0.055) / 1.055) ** 2.4, B / 12.92)
            # XYZ in range [0, 1]
            X = (R_ * 0.412453 + G_ * 0.357580 + B_ * 0.180423) / 0.95047
            Y = R_ * 0.212671 + G_ * 0.715160 + B_ * 0.072169
            Z = (R_ * 0.019334 + G_ * 0.119193 + B_ * 0.950227) / 1.08883
            X_ = torch.where(X > 0.008856, X ** (1/3), 7.787 * X + 16/116)
            Y_ = torch.where(Y > 0.008856, Y ** (1/3), 7.787 * Y + 16/116)
            Z_ = torch.where(Z > 0.008856, Z ** (1/3), 7.787 * Z + 16/116)
            # L in range [0, 100], a in range [-86.183, 98.231], b in range [-107.8573, 94.4781]
            L = 116 * Y_ - 16
            a = 500 * (X_ - Y_)
            b = 200 * (Y_ - Z_)
            # scale to [0, 1]
            L = L / 100
            a = (a + 86.183) / 184.4161
            b = (b + 107.8573) / 202.3354
            # stack and scale to [0, 1]
            if self.use_numpy:
                img = self.stack([L*255, a*255, b*255])
            else:
                img = self.stack([L, a, b])
            return img
        elif self.mode == "yuv":
            # U and V unbiased, will be in range[-0.5, 0.5]
            Y = R * 0.299 + G * 0.587 + B * 0.114
            U = R * -0.147 - G * 0.289 + B * 0.436
            V = R * 0.615 - G * 0.515 - B * 0.100
            if self.use_numpy:
                img = self.stack([Y, (U+111.18)*255/222.36, (V+156.825)*255/313.65])
            else:
                img = self.stack([Y, (U+0.436)/0.872, (V+0.615)/1.23])
            return img
        elif self.mode == "ycbcr":
            # with RGB in [0,1], computed Cb and Cr will be in range[-0.5, 0.5]
            Y = R * 0.299 + G * 0.587 + B * 0.114
            Cb = R * -0.168736 - G * 0.331264 + B * 0.5
            Cr = R * 0.5 - G * 0.418688 - B * 0.081312
            if self.use_numpy:
                img = self.stack([Y, Cb+127.5, Cr+127.5])
            else:
                img = self.stack([Y, Cb+0.5, Cr+0.5])
            return img
        else:
            raise ValueError(f"Unsupported color mode: {self.mode}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.mode}, use_numpy={self.use_numpy})"


class StatsPrinter:
    def __init__(self, prefix=""):
        self.prefix = prefix
    def __call__(self, img):
        x = np.array(img)
        print(f"{self.prefix} type={x.dtype} min={x.min()} max={x.max()} mean={x.mean()} std={x.std()}")
        return img
    def __repr__(self):
        return f"StatsPrinter(prefix={self.prefix})"
