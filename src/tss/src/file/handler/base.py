#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class for different file handler, such as: yaml, json, pickle, ...
"""

from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod
from typing import Optional
from typing import TextIO
from typing import Union

__all__ = [
	"BaseFileHandler"
]


# MARK: - BaseFileHandler

class BaseFileHandler(metaclass=ABCMeta):
	"""Base file handler implements the template methods (i.e., skeleton) for
	read and write data from/to different file formats.
	"""
	
	@abstractmethod
	def load_from_fileobj(
		self, path: Union[str, TextIO], **kwargs
	) -> Optional[Union[str, dict]]:
		"""Load the content from the given filepath or file-like object
		(input stream).
		"""
		pass
		
	@abstractmethod
	def dump_to_fileobj(self, obj, path: Union[str, TextIO], **kwargs):
		"""Dump data from the given obj to the filepath or file-like object.
		"""
		pass

	@abstractmethod
	def dump_to_str(self, obj, **kwargs) -> str:
		"""Dump data from the given obj to string."""
		pass

	def load_from_file(
		self, path: str, mode: str = "r", **kwargs
	) -> Optional[Union[str, dict]]:
		"""Load content from the given file."""
		with open(path, mode) as f:
			return self.load_from_fileobj(f, **kwargs)

	def dump_to_file(self, obj, path: str, mode: str = "w", **kwargs):
		"""Dump data from object to file.
		
		Args:
			obj:
				Object.
			path (str):
				Filepath.
			mode (str):
				File opening mode.
		"""
		with open(path, mode) as f:
			self.dump_to_fileobj(obj, f, **kwargs)
