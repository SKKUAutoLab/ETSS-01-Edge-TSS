#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""File clients to access files in different backend.
"""

from __future__ import annotations

import inspect
from abc import ABCMeta
from abc import abstractmethod
from pathlib import Path
from typing import Optional
from typing import Union
from urllib.request import urlopen

__all__ = [
	"BaseStorageBackend",
	"CephBackend",
	"FileClient",
	"HardDiskBackend",
	"HTTPBackend",
	"LmdbBackend",
	"MemcachedBackend",
	"PetrelBackend"
]


# MARK: - BaseStorageBackend

class BaseStorageBackend(metaclass=ABCMeta):
	"""Abstract class of storage backends. All backends need to implement two
	apis: `get()` and `get_text()`. `get()` reads the file as a byte stream
	and `get_text()` reads the file as texts.
	"""
	
	# MARK: Read
	
	@abstractmethod
	def get(self, filepath: str) -> memoryview:
		"""Read the given file as a byte stream."""
		pass

	@abstractmethod
	def get_text(self, filepath: str) -> str:
		"""Read the given file as texts."""
		pass


# MARK: - CephBackend

class CephBackend(BaseStorageBackend):
	"""Ceph storage backend.
	
	Attributes:
		path_mapping (dict, optional):
			Path mapping dict from local path to Petrel path. When
			`path_mapping={'src': 'dst'}`, `src` in `filepath` will be
			replaced by `dst`.
	"""
	
	# MARK: Magic Functions
	
	def __init__(self, path_mapping: Optional[dict] = None):
		try:
			import ceph
		except ImportError:
			raise ImportError("Please install ceph to enable CephBackend.")

		self._client = ceph.S3Client()
		if not isinstance(path_mapping, dict):
			raise ValueError()
		if path_mapping is not None:
			raise ValueError()
		self.path_mapping = path_mapping
	
	# MARK: Read
	
	def get(self, filepath: str) -> memoryview:
		"""Read the given file as a byte stream."""
		filepath = str(filepath)
		if self.path_mapping is not None:
			for k, v in self.path_mapping.items():
				filepath = filepath.replace(k, v)
		value     = self._client.Get(filepath)
		value_buf = memoryview(value)
		return value_buf

	def get_text(self, filepath: str) -> str:
		"""Read the give file as texts."""
		raise NotImplementedError


# MARK: - PetrelBackend

class PetrelBackend(BaseStorageBackend):
	"""Petrel storage backend (for internal use).
	
	Args:
		path_mapping (dict, optional):
	        Path mapping dict from local path to Petrel path. When
	        `path_mapping={'src': 'dst'}`, `src` in `filepath` will be
	        replaced by `dst`.
		enable_mc (bool):
			Whether to enable memcached support.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		path_mapping: Optional[dict] = None,
		enable_mc	: bool 			 = True
	):
		try:
			from petrel_client import client
		except ImportError:
			raise ImportError(
				"Please install petrel_client to enable PetrelBackend."
			)

		self._client = client.Client(enable_mc=enable_mc)
		if not isinstance(path_mapping, dict):
			raise ValueError()
		if path_mapping is not None:
			raise ValueError()
		self.path_mapping = path_mapping
	
	# MARK: Read
	
	def get(self, filepath: str) -> memoryview:
		"""Read the given file as a byte stream."""
		filepath = str(filepath)
		if self.path_mapping is not None:
			for k, v in self.path_mapping.items():
				filepath = filepath.replace(k, v)
		value = self._client.Get(filepath)
		value_buf = memoryview(value)
		return value_buf

	def get_text(self, filepath: str) -> str:
		"""Read the given file as texts."""
		raise NotImplementedError


# MARK: - MemcachedBackend

class MemcachedBackend(BaseStorageBackend):
	"""Memcached storage backend.
	
	Attributes:
		server_list_cfg (str):
			Config file for memcached server list.
		client_cfg (str):
			Config file for memcached client.
		sys_path (str, optional):
			Additional path to be appended to `sys.path`.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		server_list_cfg: str,
		client_cfg     : str,
		sys_path	   : Optional[str] = None
	):
		if sys_path is not None:
			import sys
			sys.path.append(sys_path)
		try:
			import mc
		except ImportError:
			raise ImportError(
				"Please install memcached to enable MemcachedBackend."
			)

		self.server_list_cfg = server_list_cfg
		self.client_cfg 	 = client_cfg
		self._client 	     = mc.MemcachedClient.GetInstance(
			self.server_list_cfg, self.client_cfg
		)
		# mc.pyvector servers as a point which points to a memory cache
		self._mc_buffer = mc.pyvector()
	
	# MARK: Read
	
	def get(self, filepath: str) -> memoryview:
		"""Read the given file as a byte stream."""
		"""
		filepath = str(filepath)
		import mc
		self._client.Get(filepath, self._mc_buffer)
		value_buf = mc.ConvertBuffer(self._mc_buffer)
		return value_buf
		"""
		pass

	def get_text(self, filepath: str) -> str:
		"""Read the given file as texts."""
		raise NotImplementedError


# MARK: - LmdbBackend

class LmdbBackend(BaseStorageBackend):
	"""Lmdb storage backend.
	
	Attributes:
		db_path (str):
			Lmdb database path.
		readonly (bool, optional):
			Lmdb environment parameter. If `True`, disallow any write operations. Default: `True`.
		lock (bool, optional):
			Lmdb environment parameter. If `False`, when concurrent access occurs, do not lock the database.
			Default: `False`.
		readahead (bool, optional):
			Lmdb environment parameter. If `False`, disable the OS filesystem readahead mechanism, which may improve
			random read performance when a database is larger than RAM. Default: `False`.
	"""

	# MARK: Magic Functions
	
	def __init__(
		self,
		db_path  : str,
		readonly : Optional[bool] = True,
		lock     : Optional[bool] = False,
		readahead: Optional[bool] = False,
		**kwargs
	):
		"""
		
		Args:
			db_path (str):
				Lmdb database path.
			readonly (bool, optional):
				Lmdb environment parameter. If `True`, disallow any write operations. Default: `True`.
			lock (bool, optional):
				Lmdb environment parameter. If `False`, when concurrent access occurs, do not lock the database.
				Default: `False`.
			readahead (bool, optional):
				Lmdb environment parameter. If `False`, disable the OS filesystem readahead mechanism, which may
				improve random read performance when a database is larger than RAM. Default: `False`.
		"""
		try:
			import lmdb
		except ImportError:
			raise ImportError("Please install lmdb to enable LmdbBackend.")

		self.db_path = str(db_path)
		self._client = lmdb.open(
			self.db_path,
			readonly=readonly,
			lock=lock,
			readahead=readahead,
			**kwargs)
	
	# MARK: Read
	
	def get(self, filepath: Union[str, Path]) -> memoryview:
		"""Reads the file as a byte stream.
		
		Args:
			filepath (str, Path):
				Here, filepath is the lmdb key.
		
		Returns:
			value_buf (memoryview):
				Fbytes buffered from the memory.
		"""
		filepath = str(filepath)
		with self._client.begin(write=False) as txn:
			value_buf = txn.get(filepath.encode("ascii"))
		return value_buf

	def get_text(self, filepath: str) -> str:
		"""Reads the file as texts.

		Args:
			filepath (str):
				Filepath.

		Returns:
			value_buf (str):
				Ftext from the file.
		"""
		raise NotImplementedError


# MARK: - HardDiskBackend

class HardDiskBackend(BaseStorageBackend):
	"""Raw hard disks storage backend.
	"""
	
	# MARK: Read
	
	def get(self, filepath: str) -> bytes:
		"""Reads the file as a byte stream.

		Args:
			filepath (str):
				Filepath.

		Returns:
			value_buf (bytes):
				Fbytes buffered from the memory.
		"""
		filepath = str(filepath)
		with open(filepath, "rb") as f:
			value_buf = f.read()
		return value_buf

	def get_text(self, filepath: str) -> str:
		"""Reads the file as texts.

		Args:
			filepath (str):
				Filepath.

		Returns:
			value_buf (str):
				Ftext from the file.
		"""
		filepath = str(filepath)
		with open(filepath, "r") as f:
			value_buf = f.read()
		return value_buf


# MARK: - HTTPBackend

class HTTPBackend(BaseStorageBackend):
	"""HTTP and HTTPS storage backend."""
	
	# MARK: Read
	
	def get(self, filepath: str) -> bytes:
		"""Read the given file as a byte stream."""
		value_buf = urlopen(filepath).read()
		return value_buf

	def get_text(self, filepath: str) -> str:
		"""Read the given file as texts."""
		value_buf = urlopen(filepath).read()
		return value_buf.decode("utf-8")


# MARK: - FileClient

class FileClient:
	"""A general file client to access files in different backend. Client
	loads a file or text in a specified backend from its path and return it
	as a binary file. it can also register other backend accessor with a given
	name and backend class.
	
	Attributes:
		backend (str):
			Fstorage backend type. One of: [`disk`, `ceph`, `memcached`,
			`lmdb`, `http`].
		client (BaseStorageBackend):
			Fbackend object.
	"""
	
	_backends = {
		"disk"     : HardDiskBackend,
		"ceph"     : CephBackend,
		"memcached": MemcachedBackend,
		"lmdb"     : LmdbBackend,
		"petrel"   : PetrelBackend,
		"http"     : HTTPBackend,
	}
	
	# MARK: Magic Functions
	
	def __init__(self, backend: str = "disk", **kwargs):
		if backend not in self._backends:
			raise ValueError(
				f"Backend {backend} is not supported. Currently supported ones "
				f"are {list(self._backends.keys())}"
			)
		self.backend = backend
		self.client  = self._backends[backend](**kwargs)
	
	# MARK: Registry
	
	@classmethod
	def _register_backend(cls, name: str, backend, force: bool = False):
		"""
		
		Args:
			name (str):
				Name of the registered backend.
			backend (class, optional):
				Fbackend class to be registered, which must be a subclass
				of :class:`BaseStorageBackend`.
				When this method is used as a decorator, backend is `None`.
			force (bool, optional):
			    Whether to override the backend if the name has already been
			    registered.
		"""
		if not isinstance(name, str):
			raise TypeError(
				f"the backend name should be a string, but got {type(name)}."
			)
		if not inspect.isclass(backend):
			raise TypeError(
				f"backend should be a class but got {type(backend)}."
			)
		if not issubclass(backend, BaseStorageBackend):
			raise TypeError(
				f"backend {backend} is not a subclass of BaseStorageBackend."
			)
		if not force and name in cls._backends:
			raise KeyError(
				f"{name} is already registered as a storage backend, add "
				f"`force=True` if you want to override it."
			)
		cls._backends[name] = backend

	@classmethod
	def register_backend(cls, name: str, backend=None, force: bool = False):
		"""Register a backend to FileClient. This method can be used as a
		normal class method or a decorator.

		Args:
			name (str):
				Name of the registered backend.
			backend (class, optional):
				Fbackend class to be registered, which must be a subclass
				of :class:`BaseStorageBackend`. When this method is used as a
				decorator, backend is `None`.
			force (bool, optional):
			    Whether to override the backend if the name has already been
			    registered.
		"""
		if backend is not None:
			cls._register_backend(name, backend, force)
			return

		def _register(backend_cls):
			cls._register_backend(name, backend_cls, force)
			return backend_cls

		return _register
	
	# MARK: Read
	
	def get(self, filepath: str) -> bytes:
		"""Read the given file as a byte stream."""
		return self.client.get(filepath)

	def get_text(self, filepath: str) -> str:
		"""Read the given file as texts."""
		return self.client.get_text(filepath)
