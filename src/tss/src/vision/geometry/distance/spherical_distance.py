#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from math import asin
from math import cos
from math import pow
from math import sin
from math import sqrt

import numpy as np
from multipledispatch import dispatch

from src.factory import DISTANCES

__all__ = [
	"haversine_distance",
	"HaversineDistance"
]


# MARK: - HaversineDistance

@dispatch(np.ndarray, np.ndarray)
def haversine_distance(x: np.ndarray, y: np.ndarray) -> float:
	"""FHaversine (or great circle) distance is the angular distance between
	two points on the surface of a sphere. First coordinate of each point
	is assumed to be the latitude, the second is the longitude, given in
	radians. Dimension of the data must be 2.
	"""
	R 		= 6378.0
	radians = np.pi / 180.0
	lat_x 	= radians * x[0]
	lon_x 	= radians * x[1]
	lat_y 	= radians * y[0]
	lon_y 	= radians * y[1]
	dlon  	= lon_y - lon_x
	dlat  	= lat_y - lat_x
	a 		= (pow(sin(dlat / 2.0), 2.0)
	            + cos(lat_x)
	            * cos(lat_y)
	            * pow(sin(dlon / 2.0), 2.0))
	return R * 2 * asin(sqrt(a))


@DISTANCES.register(name="haversine")
class HaversineDistance:
	"""Calculate Haversine distance."""

	# MARK: Magic Functions

	def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
		return haversine_distance(x=x, y=y)
