#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Object motion modelled by Kalman Filter with bbox as the matching
feature.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from filterpy.kalman import KalmanFilter
from src.builder import MOTIONS
from src.objects.instance import Instance

from .base import Motion

__all__ = [
    "KFBBoxMotion",
    "bbox_x_to_xyxy",
    "bbox_xyxy_to_z"
]


def bbox_xyxy_to_z(xyxy: np.ndarray) -> np.ndarray:
    """Converting bounding box for Kalman Filter. Takes a bounding box in the
    form [x1, y1, x2, y2] and returns z in the form z=[cx, cy, s, r]
    Where:
        x1, y1 is the top left
        x2, y2 is the bottom right
        cx, cy is the centre of the box
        s is the scale/area
        r is the aspect ratio
    """
    w = xyxy[2] - xyxy[0]
    h = xyxy[3] - xyxy[1]
    x = xyxy[0] + w / 2.0
    y = xyxy[1] + h / 2.0
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def bbox_x_to_xyxy(x: np.ndarray, score: float = None) -> np.ndarray:
    """Return bounding box from Kalman Filter. Takes a bounding box in the
    centre form [cx, cy, s, r] and returns it in the form [x1, y1, x2, y2]
    Where:
        x1, y1 is the top left
        x2, y2 is the bottom right
        cx, cy is the centre of the box
        s is the scale/area
        r is the aspect ratio
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([
            x[0] - w / 2.0, x[1] - h / 2.0,
            x[0] + w / 2.0, x[1] + h / 2.0
        ]).reshape((1, 4))
    else:
        return np.array([
            x[0] - w / 2.0, x[1] - h / 2.0,
            x[0] + w / 2.0, x[1] + h / 2.0,
            score
        ]).reshape((1, 5))


# MARK: - KFBBoxMotion

@MOTIONS.register(name="kf_bbox_motion")
class KFBBoxMotion(Motion):
    """ This class represents the motion model as Kalman Filter of an
    individual tracked object observed as bbox.

    Attributes:
        kf (KalmanFilter):
            Kalman Filter model.
    
    Args:
        bbox (np.ndarray):
            Bbox to initialize Kalman Filter as
            [top_left_x, top_left_y, bottom_right_x, bottom_right_y].
        hits (int):
            Number of frame has that track appear.
        hit_streak (int):
            Number of `consecutive` frame has that track appear.
        age (int):
            Number of frame while the track is alive,
            from Candidate -> Deleted.
        time_since_update (int):
            Number of `consecutive` frame that track disappear.
    """

    # MARK: Magic Functions

    def __init__(
        self,
        bbox             : Optional[np.ndarray] = None,
        hits             : int = 0,
        hit_streak       : int = 0,
        age              : int = 0,
        time_since_update: int = 0,
        *args, **kwargs
    ):
        super().__init__(
            hits              = hits,
            hit_streak        = hit_streak,
            age               = age,
            time_since_update = time_since_update,
            *args, **kwargs
        )

        # NOte: Define Kalman Filter (constant velocity model)
        self.kf   = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0  # give high uncertainty to the unobservable
                                     # initial velocities
        self.kf.P         *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # Here we assume that the `MovingObject` has already been init()
        if bbox is not None:
            self.kf.x[0:4] = bbox_xyxy_to_z(bbox)
        
    # MARK: Update

    def update_motion_state(self, instance: Instance, **kwargs):
        """Updates the state of the motion model with observed bbox.

        Args:
            instance (Instance):
                Get the specific features used to update the motion model from
                new `Detection`.
        """
        self.time_since_update = 0
        self.history           = []
        self.hits              += 1
        self.hit_streak        += 1
        self.kf.update(bbox_xyxy_to_z(instance.bbox))

    def predict_motion_state(self) -> np.ndarray:
        """Advances the state of the motion model and returns the predicted
        estimate.

        Returns:
            prediction (np.ndarray):
                Predicted bbox in
                 [top_left_x, top_left_y, bottom_right_x, bottom_right_y].
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0

        self.time_since_update += 1
        self.history.append(bbox_x_to_xyxy(self.kf.x))
        return self.history[-1]

    def current_motion_state(self) -> np.ndarray:
        """Returns the current motion model estimate.

        Returns:
            current (np.ndarray):
                Current bbox in
                [top_left_x, top_left_y, bottom_right_x, bottom_right_y].
        """
        return bbox_x_to_xyxy(self.kf.x)
