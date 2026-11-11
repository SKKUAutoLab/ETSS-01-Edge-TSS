#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import os
from shutil import copyfile
from typing import Union

from munch import Munch

# MARK: - Directories

"""Folder Hierarchy
|
|__ backup
|__ datasets    # Locally store many datasets.
|__ MLKit
|__ tools
"""

# NOTE: Inside MLKit/tss
tss_dir        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # "workspaces/MLKit/projects/tss"; this file is in tss/src
src_dir        = os.path.join(tss_dir, "src")                # "workspaces/MLKit/projects/tss/src"
test_dir       = os.path.join(tss_dir, "test")               # "workspaces/MLKit/projects/tss/test"
data_dir       = os.path.join(tss_dir, "data")               # "workspaces/MLKit/projects/tss/data"
# cai data_dir nay co the thay doi de ra cai dataset ngoai
# khong nhat hiet phai trong MLKit

# NOTE: Inside MLKit
root_dir       = os.path.dirname(os.path.dirname(tss_dir))   # "workspaces/MLKit"
models_zoo_dir = os.path.join(tss_dir, "models_zoo")         # "workspaces/MLKit/projects/tss/models_zoo"

# NOTE: Inside MLKit/projects/tss/models_zoo
checkpoints_dir = os.path.join(models_zoo_dir, "checkpoints")  # "workspaces/MLKit/projects/tss/models_zoo/checkpoints"
pretrained_dir  = os.path.join(models_zoo_dir, "pretrained")   # "workspaces/MLKit/projects/tss/models_zoo/pretrained"
results_dir     = os.path.join(models_zoo_dir, "results")      # "workspaces/MLKit/projects/tss/models_zoo/results"

# NOTE: Inside workspaces
workspaces_dir = os.path.dirname(root_dir)                   # "workspaces/"
datasets_dir   = os.path.join(workspaces_dir, "datasets")    # "workspaces/datasets"


# MARK: - Process Config

def load_config(config: Union[str, dict]) -> Munch:
	"""Load and process config from file.

	Args:
		config (str, dict):
			Config filepath that contains configuration values or the
			config dict.

	Returns:
		config (Munch):
			Config dictionary as namespace.
	"""
	# NOTE: Local import: src imports this module back (camera/*), so a top-level import is circular
	from src.file import load

	# NOTE: Load dictionary from file and convert to namespace using Munch
	if isinstance(config, str):
		config_dict = load(path=config)
	elif isinstance(config, dict):
		config_dict = config
	else:
		raise ValueError

	assert config_dict is not None, f"No configuration is found at {config}!"
	config = Munch.fromDict(config_dict)
	return config


def copy_config_file(config_file: str, dst: str):
	"""Copy config file to destination dir."""
	# NOTE: Local import, for the same reason as in load_config()
	from src.file import create_dirs

	create_dirs(paths=[dst])
	copyfile(config_file, os.path.join(dst, os.path.basename(config_file)))
