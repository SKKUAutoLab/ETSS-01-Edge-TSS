#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Handler for pickle file.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional
from typing import TextIO
from typing import Union

from .base import BaseFileHandler
from src.factory import FILE_HANDLERS

__all__ = [
	"PickleHandler"
]


# MARK: - PickleHandler

@FILE_HANDLERS.register(name="pickle")
@FILE_HANDLERS.register(name="pkl")
class PickleHandler(BaseFileHandler):
	"""Pickle file handler."""
	
	def load_from_fileobj(
		self, path: Union[str, TextIO], **kwargs
	) -> Optional[Union[str, dict]]:
		"""Load the content from the given filepath or file-like object
		(input stream).
		"""
		return pickle.load(path, **kwargs)

	def dump_to_fileobj(self, obj, path: Union[str, TextIO], **kwargs):
		"""Dump data from the given obj to the filepath or file-like object.
		"""
		kwargs.setdefault("protocol", 2)
		pickle.dump(obj, path, **kwargs)
		
	def dump_to_str(self, obj, **kwargs) -> bytes:
		""""Dump data from the given obj to string."""
		kwargs.setdefault("protocol", 2)
		return pickle.dumps(obj, **kwargs)
		
	def load_from_file(
		self, file: Union[str, Path], **kwargs
	) -> Optional[Union[str, dict]]:
		"""Load content from the given file."""
		return super().load_from_file(file, mode="rb", **kwargs)
	
	def dump_to_file(self, obj, path: Union[str, Path], **kwargs):
		"""Dump data from object to file.
		
		Args:
			obj:
				Object.
			path (str, Path):
				Filepath.
		"""
		super().dump_to_file(obj, path, mode="wb", **kwargs)
