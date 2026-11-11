#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class for all tracker.
"""

from __future__ import annotations

import abc
from typing import Union

import numpy as np
from src.builder import MOTIONS
from src.trackers.motion.base import Motion
from src.trackers.motion.kf_bbox_motion import KFBBoxMotion
from src.type import Callable, ListOrTuple3T

__all__ = [
    "BaseTracker"
]


# MARK: - BaseTracker

class BaseTracker(metaclass=abc.ABCMeta):
    """Base Tracker.

    Attributes:
        name (str):
            Name of the tracker.
        motion_model (FuncCls):
            Motion model class. Default: `KFBBoxMotion`
        max_age (int):
            Time to store the track before deleting, that mean track could
            live in `max_age` frame with no match bounding box, consecutive
            frame that track disappear. Default: `1`.
        min_hits (int):
            Number of frame which has matching bounding box of the detected
            object before the object is considered becoming the track.
            Default: `3`.
        iou_threshold (float):
            Intersection over Union threshold between two track with their
            bounding box. Default: `0.3`.
        frame_count (int):
            Current index of reading frame.
        tracks (list):
            List of `Track`.
    """

    # MARK: Magic Functions

    def __init__(
        self,
        name         : str,
        max_age      : int   = 1,
        min_hits     : int   = 3,
        iou_threshold: float = 0.3,
        motion_model : Union[str, dict, Motion, Callable] = "kf_bbox_motion",
        *args, **kwargs
    ):
        super().__init__()
        self.name          = name
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self.frame_count   = 0
        self.motion_model  = KFBBoxMotion

        from src.objects import MovingObject
        self.tracks: list[MovingObject] = []
        
        self.init_motion_model(motion_model=motion_model)
    
    # MARK: Configure
    
    def init_motion_model(
        self, motion_model: Union[str, dict, Motion, Callable]
    ):
        """Initialize the motion model for tracked objects."""
        if isinstance(motion_model, str):
            motion_model = MOTIONS.build(name=motion_model).__class__
        elif isinstance(motion_model, dict):
            motion_model = MOTIONS.build_from_dict(cfg=motion_model).__class__
        elif isinstance(motion_model, Motion):
            motion_model = motion_model.__class__
        self.motion_model = motion_model
    
    # MARK: Update

    @abc.abstractmethod
    def update(self, instances: list):
        """Update `self.tracks` with new instances.

        Args:
            instances (list):
                List of newly detected instances.

        Requires:
            This method must be called once for each frame even with empty
            detections, just call update with empty container.
        """
        pass

    @abc.abstractmethod
    def associate_instances_to_tracks(
        self, instances: np.ndarray, tracks: np.ndarray
    ) -> ListOrTuple3T[np.ndarray]:
        """Assigns `instances` to `self.tracks`.

        Args:
            instances (np.ndarray):
                Newly detected instances.
            tracks (np.ndarray):
                Current tracks.

        Returns:
            matched_indexes (np.ndarray):
            unmatched_inst_indexes (np.ndarray):
            unmatched_trks_indexes (np.ndarray):
        """
        pass

    @abc.abstractmethod
    def update_matched_tracks(
        self, matched_indexes: np.ndarray, instances: list
    ):
        """Update tracks that have been matched with new detected instances.

        Args:
            matched_indexes (np.ndarray):
                Indexes of `self.tracks` that have not been matched with new
                instances.
            instances (list):
                Newly detected instances.
        """
        pass

    @abc.abstractmethod
    def create_new_tracks(
        self, unmatched_inst_indexes: np.ndarray, instances: list
    ):
        """Create new tracks.

        Args:
            unmatched_inst_indexes (np.ndarray):
                Indexes of `instances` that have not been matched with any
                tracks.
            instances (list):
                Newly detected instances.
        """
        pass

    @abc.abstractmethod
    def delete_dead_tracks(self):
        """Delete dead tracks."""
        pass
