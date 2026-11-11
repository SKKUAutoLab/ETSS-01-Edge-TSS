#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Handler for .yaml file.
"""

from __future__ import annotations

from typing import Optional
from typing import TextIO
from typing import Union

import yaml

from .base import BaseFileHandler
from src.factory import FILE_HANDLERS

try:
	from yaml import CLoader as FullLoader, CDumper as Dumper
except ImportError:
	from yaml import FullLoader, Dumper

__all__ = [
	"YamlHandler"
]


# MARK: - YamlHandler

@FILE_HANDLERS.register(name="yaml")
@FILE_HANDLERS.register(name="yml")
class YamlHandler(BaseFileHandler):
	"""YAML file handler."""
	
	def load_from_fileobj(
		self, path: Union[str, TextIO], **kwargs
	) -> Optional[Union[str, dict]]:
		"""Load the content from the given filepath or file-like object
		(input stream).
		"""
		kwargs.setdefault("Loader", FullLoader)
		return yaml.load(path, **kwargs)

	def dump_to_fileobj(self, obj, path: Union[str, TextIO], **kwargs):
		"""Dump data from the given obj to the filepath or file-like object.
		"""
		kwargs.setdefault("Dumper", Dumper)
		yaml.dump(obj, path, **kwargs)

	def dump_to_str(self, obj, **kwargs) -> str:
		"""Dump data from the given obj to string."""
		kwargs.setdefault("Dumper", Dumper)
		return yaml.dump(obj, **kwargs)
