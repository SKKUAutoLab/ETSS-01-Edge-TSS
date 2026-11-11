#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Function for parsing and dumping data to several file format, such as:
yaml, txt, json, ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Optional
from typing import TextIO
from typing import Union

from src.factory import FILE_HANDLERS

__all__ = [
    "dump",
    "load"
]


# MARK: - Load

def load(
    path: Union[str, Path, TextIO], file_format: Optional[str] = None, **kwargs
) -> Union[str, dict, None]:
    """Load data from json/yaml/pickle files. This method provides a unified
    api for loading data from serialized files.
   
    Args:
        path (str, Path, TextIO):
            Filename, path, or a file-like object.
        file_format (str, optional):
            If not specified, the file format will be inferred from the file
            extension, otherwise use the specified one. Currently supported
            formats include "json", "yaml/yml" and "pickle/pkl".
   
    Returns:
        data (str, dict, optional):
            Content from the file.
    """
    if isinstance(path, Path):
        path = str(path)
    if file_format is None and isinstance(path, str):
        file_format = path.split(".")[-1]
    if file_format not in FILE_HANDLERS:
        raise TypeError(f"Unsupported format: {file_format}.")

    handler = FILE_HANDLERS.build(name=file_format)
    if isinstance(path, str):
        data = handler.load_from_file(path, **kwargs)
    elif hasattr(path, "read"):
        data = handler.load_from_fileobj(path, **kwargs)
    else:
        raise TypeError("`file` must be a filepath str or a file-object.")
    return data


# MARK: - Dump

def dump(
    obj        : Any,
    path       : Union[str, Path, TextIO],
    file_format: Optional[str] = None,
    **kwargs
) -> Union[bool, str]:
    """Dump data to json/yaml/pickle strings or files. This method provides a
    unified api for dumping data as strings or to files, and also supports
    custom arguments for each file format.
    
    Args:
        obj (any):
            Python object to be dumped.
        path (str, Path, TextIO):
            If not specified, then the object is dump to a str, otherwise to a
            file specified by the filename or file-like object.
        file_format (str, optional):
            If not specified, the file format will be inferred from the file
            extension, otherwise use the specified one. Currently supported
            formats include "json", "yaml/yml" and "pickle/pkl".
    
    Returns:
        (bool, str):
            `True` for success, `False` otherwise.
    """
    if isinstance(path, Path):
        path = str(path)
    if file_format is None:
        if isinstance(path, str):
            file_format = path.split(".")[-1]
        elif path is None:
            raise ValueError(
                "`file_format` must be specified since file is None."
            )
    if file_format not in FILE_HANDLERS:
        raise TypeError(f"Unsupported format: {file_format}.")

    handler = FILE_HANDLERS.build(name=file_format)
    if path is None:
        return handler.dump_to_str(obj, **kwargs)
    elif isinstance(path, str):
        handler.dump_to_file(obj, path, **kwargs)
    elif hasattr(path, "write"):
        handler.dump_to_fileobj(obj, path, **kwargs)
    else:
        raise TypeError("`file` must be a filename str or a file-object.")
