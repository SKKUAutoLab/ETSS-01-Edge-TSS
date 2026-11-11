#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""All factory classes used in the `torchkit` package. We define them here to
avoid circular dependency.
"""

from __future__ import annotations

from .factory import Factory

# MARK: - Augment

AUGMENTS   = Factory(name="augments")
TRANSFORMS = Factory(name="transforms")


# MARK: - Data

LABEL_HANDLERS = Factory(name="label_handlers")
DATASETS       = Factory(name="datasets")
DATAMODULES    = Factory(name="datamodules")


# MARK: - File

FILE_HANDLERS = Factory(name="file_handler")


# MARK: - Layers

ACT_LAYERS           = Factory(name="act_layers")
ATTN_LAYERS          = Factory(name="attn_layers")
ATTN_POOL_LAYERS     = Factory(name="attn_pool_layers")
BOTTLENECK_LAYERS    = Factory(name="bottleneck_layers")
CONV_LAYERS          = Factory(name="conv_layers")
CONV_ACT_LAYERS      = Factory(name="conv_act_layers")
CONV_NORM_ACT_LAYERS = Factory(name="conv_norm_act_layers")
DROP_LAYERS          = Factory(name="drop_layers")
EMBED_LAYERS         = Factory(name="embed_layers")
HEADS 	             = Factory(name="heads")
LINEAR_LAYERS        = Factory(name="linear_layers")
MLP_LAYERS           = Factory(name="mlp_layers")
NORM_LAYERS          = Factory(name="norm_layers")
NORM_ACT_LAYERS      = Factory(name="norm_act_layers")
PADDING_LAYERS       = Factory(name="padding_layers")
PLUGIN_LAYERS        = Factory(name="plugin_layers")
POOL_LAYERS          = Factory(name="pool_layers")
SAMPLING_LAYERS      = Factory(name="sampling_layers")


# MARK: - Losses & Metrics

LOSSES  = Factory(name="losses")
METRICS = Factory(name="metrics")


# MARK: - Math

DISTANCES = Factory(name="distance_functions")


# MARK: - Models

ACTION_DETECTORS  = Factory(name="action_detectors")
BACKBONES         = Factory(name="backbones")
CALLBACKS         = Factory(name="callbacks")
IMAGE_CLASSIFIERS = Factory(name="image_classifiers")
IMAGE_ENHANCERS   = Factory(name="image_enhancers")
INFERENCES        = Factory(name="inferences")
DETECTORS         = Factory(name="object_detectors")
LOGGERS           = Factory(name="loggers")
MODELS            = Factory(name="models")
MODULE_WRAPPERS   = Factory(name="module_wrappers")
NECKS             = Factory(name="necks")
IDENTIFICATIONS   = Factory(name="identifications")
HEURISTICS        = Factory(name="heuristics")
TRACKERS          = Factory(name="trackers")
MATCHERS          = Factory(name="matchers")
ANALYZERS         = Factory(name="analyzers")
