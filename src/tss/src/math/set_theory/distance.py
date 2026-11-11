#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import numpy as np

from src.factory import DISTANCES
from src.math.algebra import euclidean_distance

__all__ = [
	"hausdorff_distance",
	"HausdorffDistance",
]


# MARK: - HausdorffDistance

def hausdorff_distance(x: np.ndarray, y: np.ndarray) -> float:
	"""Calculation of Hausdorff distance btw 2 arrays.
	
	`euclidean_distance`, `manhattan_distance`, `chebyshev_distance`,
	`cosine_distance`, `haversine_distance` could be use for this function.
	"""
	cmax = 0.0
	for i in range(len(x)):
		cmin = np.inf
		for j in range(len(y)):
			d = euclidean_distance(x[i, :], y[j, :])
			if d < cmin:
				cmin = d
			if cmin < cmax:
				break
		if cmax < cmin < np.inf:
			cmax = cmin
	return cmax


@DISTANCES.register(name="hausdorff")
class HausdorffDistance:
	"""Calculate Hausdorff distance."""

	# MARK: Magic Functions

	def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
		return hausdorff_distance(x=x, y=y)
