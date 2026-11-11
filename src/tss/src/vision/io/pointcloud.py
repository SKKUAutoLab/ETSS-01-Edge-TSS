#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import os

import torch
from torch import Tensor

__all__ = [
    "save_pointcloud_ply",
    "load_pointcloud_ply"
]


def save_pointcloud_ply(filename: str, pointcloud: Tensor):
    """Utility function to save to disk a pointcloud in PLY format.

    Args:
        filename (str):
            Path to save the pointcloud.
        pointcloud (Tensor):
            Tensor containing the pointcloud to save. Image must be in the
            shape of [*, 3] where the last component is assumed to be a 3d
            point coordinate [X, Y, Z].
    """
    if not isinstance(filename, str) and filename[-3:] == ".ply":
        raise TypeError("Input filename must be a string in with the .ply "
                        "extension. Got: {}".format(filename))
    if not torch.is_tensor(pointcloud):
        raise TypeError(f"Input pointcloud type is not a Tensor. "
                        f"Got: {type(pointcloud)}")
    if not len(pointcloud.shape) == 3 and pointcloud.shape[-1] == 3:
        raise TypeError("Input pointcloud must be in the following shape "
                        "[H, W, 3]. Got: {}.".format(pointcloud.shape))

    # Flatten the input pointcloud in a vector to iterate points
    xyz_vec = pointcloud.reshape(-1, 3)

    with open(filename, "w") as f:
        data_str   = ""
        num_points = xyz_vec.shape[0]
        for idx in range(num_points):
            xyz = xyz_vec[idx]
            if not bool(torch.isfinite(xyz).any()):
                num_points -= 1
                continue
            x = xyz[0].item()
            y = xyz[1].item()
            z = xyz[2].item()
            data_str += f'{x} {y} {z}\n'

        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment array generated\n")
        f.write("element vertex %d\n" % num_points)
        f.write("property double x\n")
        f.write("property double y\n")
        f.write("property double z\n")
        f.write("end_header\n")
        f.write(data_str)


def load_pointcloud_ply(filename: str, header_size: int = 8) -> Tensor:
    """Utility function to load from disk a pointcloud in PLY format.

    Args:
        filename (str):
            Path to the pointcloud.
        header_size (int):
            Fsize of the ply file header that will be skipped during loading.

    Return:
        pointcloud (Tensor):
            Tensor containing the loaded point with shape [*, 3] where `*`
            represents the number of points.
    """
    if not isinstance(filename, str) and filename[-3:] == ".ply":
        raise TypeError("Input filename must be a string in with the .ply "
                        "extension. Got: {}".format(filename))
    if not os.path.isfile(filename):
        raise ValueError("Input filename is not an existing file.")
    if not (isinstance(header_size, int) and header_size > 0):
        raise TypeError(f"Input header_size must be a positive integer. "
                        f"Got: {header_size}.")
    
    # Open the file and populate image
    with open(filename) as f:
        points = []
        
        # Skip header
        lines = f.readlines()[header_size:]

        # Iterate over the points
        for line in lines:
            x_str, y_str, z_str = line.split()
            points.append((torch.tensor(float(x_str)),
                           torch.tensor(float(y_str)),
                           torch.tensor(float(z_str)) ))

        # Create image from list
        pointcloud = torch.tensor(points)
        return pointcloud
