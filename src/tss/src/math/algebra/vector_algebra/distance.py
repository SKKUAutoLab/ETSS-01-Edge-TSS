#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from math import sqrt

import numpy as np
from multipledispatch import dispatch

from src.factory import DISTANCES

__all__ = [
	"angle_between_vectors",
	"chebyshev_distance",
	"ChebyshevDistance",
	"cosine_distance",
	"CosineDistance",
	"euclidean_distance",
	"EuclideanDistance",
]


# MARK: - Angle

def angle_between_vectors(x: np.ndarray, y: np.ndarray) -> float:
	"""Calculate angle of 2 trajectories between two trajectories.
	"""
	vec1 = np.array([x[-1][0] - x[0][0],
	                 x[-1][1] - x[0][1]])
	vec2 = np.array([y[-1][0] - y[0][0],
	                 y[-1][1] - y[0][1]])
	
	L1 = np.sqrt(vec1.dot(vec1))
	L2 = np.sqrt(vec2.dot(vec2))
	
	if L1 == 0 or L2 == 0:
		return False
	
	cos   = vec1.dot(vec2) / (L1 * L2)
	angle = np.arccos(cos) * 360 / (2 * np.pi)
	
	return angle


# MARK: - ChebyshevDistance

@dispatch(np.ndarray, np.ndarray)
def chebyshev_distance(x: np.ndarray, y: np.ndarray) -> float:
	"""Chebyshev distance is a metric defined on a vector space where the
	distance between two vectors is the greatest of their differences along any
	coordinate dimension.
	"""
	n   = x.shape[0]
	ret = -1 * np.inf
	for i in range(n):
		d = abs(x[i] - y[i])
		if d > ret:
			ret = d
	return ret


@DISTANCES.register(name="chebyshev")
class ChebyshevDistance:
	"""Calculate Chebyshev distance."""

	# MARK: Magic Functions

	def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
		return chebyshev_distance(x=x, y=y)


# MARK: - CosineDistance

@dispatch(np.ndarray, np.ndarray)
def cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
	"""Cosine similarity is a measure of similarity between two non-zero
	vectors of an inner product space.
	"""
	n 	= x.shape[0]
	xy_dot = 0.0
	x_norm = 0.0
	y_norm = 0.0
	for i in range(n):
		xy_dot += x[i] * y[i]
		x_norm += x[i] * x[i]
		y_norm += y[i] * y[i]
	return 1.0 - xy_dot / (sqrt(x_norm) * sqrt(y_norm))


@DISTANCES.register(name="cosine")
class CosineDistance:
	"""Calculate Cosine distance."""

	# MARK: Magic Functions

	def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
		return cosine_distance(x=x, y=y)


# MARK: - EuclideanDistance

@dispatch(np.ndarray, np.ndarray)
def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
	"""Calculation of Euclidean distance btw 2 arrays."""
	n   = x.shape[0]
	ret = 0.0
	for i in range(n):
		ret += ((x[i] - y[i]) ** 2)
	return sqrt(ret)


@DISTANCES.register(name="euclidean")
class EuclideanDistance:
	"""Calculate Euclidean distance."""

	# MARK: Magic Functions

	def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
		return euclidean_distance(x=x, y=y)


# MARK: - ManhattanDistance

def manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
	"""Calculation of Manhattan distance btw 2 arrays."""
	n   = x.shape[0]
	ret = 0.0
	for i in range(n):
		ret += abs(x[i] - y[i])
	return ret


@DISTANCES.register(name="manhattan")
class ManhattanDistance:
	"""Calculate Manhattan distance."""

	# MARK: Magic Functions

	def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
		return manhattan_distance(x=x, y=y)
