#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from src.data.data_class import ObjectAnnotation
from src.factory import AUGMENTS
from src.vision import adjust_hsv
from src.vision import bbox_xyxy_to_cxcywh_norm
from src.vision import random_bbox_perspective
from .base import BaseAugment

__all__ = [
    "ImageBoxAugment",
]


# MARK: - ImageBoxAugment

@AUGMENTS.register(name="image_box_augment")
class ImageBoxAugment(BaseAugment):
    r"""
    
    Args:
        policy (str):
			Augmentation policy. One of: [`scratch`]. Default: `scratch`.
    """

    cfgs = {
        "scratch": [
            (
                ("random_bbox_perspective", 0.5, (0.0, 0.5, 0.5, 0.0, 0.0)),
                ("adjust_hsv", 0.5, (0.015, 0.7, 0.4)),
                ("fliplr", 0.5, None),
                ("flipud", 0.0, None),
            ),
        ],
        "finetune": [
            (
                ("random_bbox_perspective", 0.5, (0.0, 0.5, 0.8, 0.0, 0.0)),
                ("adjust_hsv", 0.5, (0.015, 0.7, 0.4)),
                ("fliplr", 0.5, None),
                ("flipud", 0.0, None),
            ),
        ],
    }
    
    # MARK: Magic Functions

    def __init__(self, policy: str = "scratch", *args, **kwargs):
        super().__init__(*args, **kwargs)
        if policy not in self.cfgs:
            raise ValueError(f"transforms must be one of: {self.cfgs.keys()}")
        self.transforms = self.cfgs[policy]

    def __repr__(self) -> str:
        return self.__class__.__name__ + \
               f"(policy={self.policy}, fill={self.fill})"
    
    # MARK: Configure

    def _augmentation_space(
        self, *args, **kwargs
    ) -> dict[str, tuple[Tensor, bool]]:
        pass

    # MARK: Forward Pass
    
    def forward(
        self, input: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        
        Args:
            input (np.ndarray):
                Image to be transformed.
            target (np.ndarray):
                Target to be transformed.
        """
        # NOTE: Transform
        transform_id = int(torch.randint(len(self.transforms), (1,)).item())
        num_ops      = len(self.transforms[transform_id])
        probs        = torch.rand((num_ops,))
        for i, (op_name, p, magnitude) in enumerate(
            self.transforms[transform_id]
        ):
            if probs[i] <= p:
                magnitude = (magnitude if magnitude is not None else 0.0)
                if op_name == "random_bbox_perspective":
                    input, target = random_bbox_perspective(
                        image       = input,
                        bbox        = target,
                        rotate      = magnitude[0],
                        translate   = magnitude[1],
                        scale       = magnitude[2],
                        shear       = magnitude[3],
                        perspective = magnitude[4],
                    )
                    nl = len(target)  # Number of labels
                    if nl:
                        target = target
                    else:
                        target = np.zeros((nl, ObjectAnnotation.bbox_label_len()))
                    target[:, 2:6] = bbox_xyxy_to_cxcywh_norm(
                        target[:, 2:6], input.shape[0], input.shape[1]
                    )
                elif op_name == "adjust_hsv":
                    input = adjust_hsv(
                        input,
                        h_factor = magnitude[0],
                        s_factor = magnitude[1],
                        v_factor = magnitude[2],
                    )
                elif op_name == "fliplr":
                    input        = np.fliplr(input)
                    target[:, 2] = 1 - target[:, 2]
                elif op_name == "flipud":
                    input        = np.flipud(input)
                    target[:, 3] = 1 - target[:, 3]
         
        return input, target
