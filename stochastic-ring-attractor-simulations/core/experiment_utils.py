import hashlib
import json
import os
import sys
from datetime import datetime
from typing import Iterable, Mapping, Optional, Sequence

import h5py
import numpy as np


def code_fingerprint(path: Optional[str] = None) -> str:
    """
    A tiny 'version' string to stash in metadata (based on a file’s content).
    Defaults to hashing the current entrypoint script.
    """
    script_path = path or os.path.abspath(sys.argv[0])
    try:
        with open(script_path, "rb") as f:
            return hashlib.sha1(f.read()).hexdigest()[:12]
    except Exception:
        return "unknown"


def _to_np(value, dtype=None):
    try:
        return np.asarray(value, dtype=dtype) if dtype is not None else np.asarray(value)
    except Exception:
        return np.array(value, dtype=object)


class H5ExperimentWriter:
    """
    Single-writer utility for experiment HDF5 files.

    - Stores base parameters + runtime metadata on the file root.
    - Each call to `write_group` creates/overwrites a group under /runs.
    - Handles JSON fallback for object arrays and strings.
    """

    def __init__(self, h5_path: str, base_params: Mapping, runtime_meta: Optional[Mapping] = None,
                 compression: str = "gzip", compression_lvl: int = 4):
        self.h5_path = h5_path
        self.compression = compression
        self.compression_lvl = compression_lvl
        self._init_file(base_params, runtime_meta or {})

    def _init_file(self, base_params: Mapping, runtime_meta: Mapping):
        os.makedirs(os.path.dirname(self.h5_path), exist_ok=True)
        with h5py.File(self.h5_path, "w") as f:
            f.attrs["created_at"] = datetime.now().isoformat()
            f.attrs["base_params_json"] = json.dumps(base_params)
            for k, v in runtime_meta.items():
                f.attrs[k] = json.dumps(v) if not isinstance(v, (str, bytes)) else v
            f.create_group("runs")

    def _create_ds(self, group, name: str, data):
        """Robust dataset creation with JSON fallback for object arrays and scalar handling."""
        # Lists/tuples -> ndarray
        if isinstance(data, (list, tuple)):
            try:
                data = np.asarray(data)
            except Exception:
                data = np.array(data, dtype=object)

        # Strings
        if isinstance(data, str):
            dt = h5py.string_dtype(encoding="utf-8")
            group.create_dataset(name, data=data, dtype=dt)
            return

        # Object arrays → JSON bytes
        if isinstance(data, np.ndarray) and data.dtype == object:
            try:
                payload = json.dumps(data.tolist()).encode("utf-8")
                ds = group.create_dataset(name, data=np.frombuffer(payload, dtype="uint8"))
                ds.attrs["encoding"] = "json-utf8"
                return
            except Exception as e:
                group.attrs[f"{name}__json_error"] = str(e)
                return

        # Scalars (no compression allowed)
        is_scalar = np.isscalar(data) or (isinstance(data, np.ndarray) and data.shape == ())
        try:
            if is_scalar:
                group.create_dataset(name, data=data)
            else:
                group.create_dataset(
                    name,
                    data=data,
                    compression=self.compression,
                    compression_opts=self.compression_lvl,
                    shuffle=True,
                )
        except Exception as e:
            # Fallback: try without compression; else store error as attribute
            try:
                group.create_dataset(name, data=data)
            except Exception:
                group.attrs[f"{name}__store_error"] = str(e)

    def write_group(self, name: str, attrs: Mapping, datasets: Mapping,
                    skip_keys: Optional[Iterable[str]] = None):
        """
        Create (or overwrite) a group under /runs with the provided attrs and datasets.
        """
        skip_keys = set(skip_keys or [])
        with h5py.File(self.h5_path, "a") as f:
            runs = f["runs"]
            if name in runs:
                del runs[name]
            g = runs.create_group(name)
            for k, v in attrs.items():
                g.attrs[k] = json.dumps(v) if isinstance(v, (list, dict, tuple)) else v
            for key, val in datasets.items():
                if key in skip_keys:
                    continue
                self._create_ds(g, key, val)


def write_parameters_file(path: str, header: str, base_params: Mapping,
                          extra_lines: Optional[Sequence[str]] = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(header.rstrip() + "\n\n")
        if extra_lines:
            for line in extra_lines:
                f.write(f"{line}\n")
            f.write("\n")
        f.write("Base params:\n")
        f.write(json.dumps(base_params, indent=2) + "\n")
