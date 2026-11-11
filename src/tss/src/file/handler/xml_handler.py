#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Handler for .xml file.
"""

from __future__ import annotations

from typing import TextIO
from typing import Union

import xmltodict

from .base import BaseFileHandler
from src.factory import FILE_HANDLERS

__all__ = [
	"XmlHandler"
]


# MARK: - XmlHandler

@FILE_HANDLERS.register(name="xml")
class XmlHandler(BaseFileHandler):
	"""XML file handler.
	"""
	
	def load_from_fileobj(
		self, path: Union[str, TextIO], **kwargs
	) -> Union[str, dict, None]:
		"""Load data from file object (input stream).

		Args:
			path (str, TextIO):
				Filepath or a file-like object.

		Returns:
			(str, dict, optional):
				Content from the file.
		"""
		doc = xmltodict.parse(path.read())
		return doc

	def dump_to_fileobj(self, obj, path: Union[str, TextIO], **kwargs):
		"""Dump data from obj to file.

		Args:
			obj:
				Object.
			path (str, TextIO):
				Filepath or a file-like object.
		"""
		if not isinstance(obj, dict):
			raise ValueError(f"obj {obj} must be a dictionary.")
		
		with open(path, "w") as path:
			path.write(xmltodict.unparse(obj, pretty=True))
		
	def dump_to_str(self, obj, **kwargs) -> str:
		"""Dump data from obj to string.

		Args:
			obj:
				Object.

		Returns:
			(str):
				Content from the file.
		"""
		if not isinstance(obj, dict):
			raise ValueError(f"obj {obj} must be a dictionary.")
		
		return xmltodict.unparse(obj, pretty=True)
