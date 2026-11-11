#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Geometric image transformations is another key ingredient in computer vision
to manipulate images. Since geometry operations are typically performed in
2D or 3D, we provide several algorithms to work with both cases. This module,
the original core of the library, consists of the following submodules:
transforms, camera, conversions, linalg and depth. We next describe each of
them:

- transforms: Module provides low level interfaces to manipulate 2D images,
  with routines for Rotating, Scaling, Translating, Shearing; Cropping
  functions in several modalities such as central crops, crop and resize;
  Flipping transformations in the vertical and horizontal axis; Resizing
  operations; Functions to warp tensors given affine or perspective
  transformations, and utilities to compute the transformation matrices to
  perform the mentioned operations.

- camera: A set of routines specific to different types of camera
  representations such as Pinhole or Orthographic models containing
  functionalities such as projecting and unprojecting points from the camera to
  a world frame.

- conversions: Routines to perform conversions between angle representation
  such as radians to degrees, coordinates normalization, and homogeneous to
  euclidean. Moreover, we include advanced conversions for 3D geometry
  representations such as Quaternion, Axis-Angle, Rotation Matrix, or Rodrigues
  formula.

- linalg: Functions to perform general rigid-body homogeneous transformations.
  We include implementations to transform points between frames and for
  homogeneous transformations, manipulation such as composition, inverse and to
  compute relative poses.

- depth: A set of layers to manipulate depth maps such as how to compute 3D
  point clouds given depth maps and calibrated cameras; compute surface normals
  per pixel and warp image frames given calibrated cameras setup.
  
For a full list of geometry's taxonomy, refer to:
https://mathworld.wolfram.com/topics/Geometry.html
"""

from __future__ import annotations

from .anchor import *
from .conversion import *
from .distance import *
from .line_geometry import *
from .plane_geometry import *
from .point import *
from .transformation import *
