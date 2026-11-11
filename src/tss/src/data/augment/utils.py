#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import math
from typing import Optional

from torch import Tensor
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode

__all__ = [
    "apply_transform_op",
]


def apply_transform_op(
    image        : Tensor,
    op_name      : str,
    magnitude    : float,
    interpolation: InterpolationMode,
    fill         : Optional[list[float]]
) -> Tensor:
    if op_name == "auto_contrast":
        image = F.autocontrast(image)
    elif op_name == "brightness":
        image = F.adjust_brightness(image, 1.0 + magnitude)
    elif op_name == "color":
        image = F.adjust_saturation(image, 1.0 + magnitude)
    elif op_name == "contrast":
        image = F.adjust_contrast(image, 1.0 + magnitude)
    elif op_name == "equalize":
        image = F.equalize(image)
    elif op_name == "fliplr":
        image = F.hflip(image)
    elif op_name == "flipud":
        image = F.vflip(image)
    elif op_name == "identity":
        pass
    elif op_name == "invert":
        image = F.invert(image)
    elif op_name == "posterize":
        image = F.posterize(image, int(magnitude))
    elif op_name == "rotate":
        image = F.rotate(
            image, magnitude, interpolation=interpolation, fill=fill
        )
    elif op_name == "sharpness":
        image = F.adjust_sharpness(image, 1.0 + magnitude)
    elif op_name == "shear_x":
        image = F.affine(
            image,
            angle         = 0.0,
            translate     = [0, 0],
            scale         = 1.0,
            shear         = [math.degrees(magnitude), 0.0],
            interpolation = interpolation,
            fill          = fill,
        )
    elif op_name == "shear_y":
        image = F.affine(
            image,
            angle         = 0.0,
            translate     = [0, 0],
            scale         = 1.0,
            shear         = [0.0, math.degrees(magnitude)],
            interpolation = interpolation,
            fill          = fill,
        )
    elif op_name == "solarize":
        image = F.solarize(image, magnitude)
    elif op_name == "translate_x":
        image = F.affine(
            image,
            angle         = 0.0,
            translate     = [int(magnitude), 0],
            scale         = 1.0,
            interpolation = interpolation,
            shear         = [0.0, 0.0],
            fill          = fill,
        )
    elif op_name == "translate_y":
        image = F.affine(
            image,
            angle         = 0.0,
            translate     = [0, int(magnitude)],
            scale         = 1.0,
            interpolation = interpolation,
            shear         = [0.0, 0.0],
            fill          = fill,
        )
    else:
        raise ValueError(f"Provided operator {op_name} is not recognized.")
    return image
