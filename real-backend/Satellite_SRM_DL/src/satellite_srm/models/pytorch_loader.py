"""
pytorch_loader.py — Zero-dependency loader for real PyTorch .pt checkpoint files.

Reads the ZIP-based PyTorch serialization format and reconstructs state dicts
as plain numpy arrays, without requiring the torch package.

PyTorch ZIP layout:
  <archive_name>/data.pkl       - pickle referencing persistent storage IDs
  <archive_name>/data/<id>      - raw tensor data blobs (binary float32)
  <archive_name>/.format_version
  <archive_name>/byteorder
"""

import zipfile
import pickle
import io
import numpy as np
import struct
from typing import Dict, Any


class _TorchStorage:
    """Stub for torch.FloatStorage / torch.HalfStorage etc."""
    def __init__(self, data: np.ndarray, dtype=np.float32):
        self.data = data.astype(dtype)


class _NumpyUnpickler(pickle.Unpickler):
    """Custom unpickler that intercepts torch persistent-id references
    and replaces them with numpy arrays read from the ZIP blob store."""

    def __init__(self, f, zip_file: zipfile.ZipFile, archive_name: str, byteorder: str = "little"):
        super().__init__(f)
        self.zip_file = zip_file
        self.archive_name = archive_name
        self.byteorder = byteorder
        self._storage_cache: Dict[str, np.ndarray] = {}

    def persistent_load(self, pid):
        """
        PyTorch persistent IDs have the form:
          ('storage', storage_type, storage_key, location, numel)
        """
        type_tag = pid[0]
        if type_tag == 'storage':
            _, storage_type, key, location, numel = pid
            if key not in self._storage_cache:
                blob_path = f"{self.archive_name}/data/{key}"
                with self.zip_file.open(blob_path) as bf:
                    raw = bf.read()
                # Determine element size from storage type name
                type_name = storage_type.__name__ if hasattr(storage_type, '__name__') else str(storage_type)
                if 'Half' in type_name:
                    dtype = np.float16
                elif 'Double' in type_name:
                    dtype = np.float64
                elif 'Int' in type_name:
                    dtype = np.int32
                elif 'Long' in type_name:
                    dtype = np.int64
                elif 'Bool' in type_name:
                    dtype = np.bool_
                else:
                    dtype = np.float32
                arr = np.frombuffer(raw, dtype=np.dtype(dtype).newbyteorder('<'))
                self._storage_cache[key] = arr
            return _TorchStorage(self._storage_cache[key])
        raise pickle.UnpicklingError(f"Unknown persistent id type: {type_tag!r}")

    def find_class(self, module, name):
        # Return stub classes for torch storage types
        if module == 'torch' and 'Storage' in name:
            dtype_map = {
                'FloatStorage': np.float32,
                'HalfStorage': np.float16,
                'DoubleStorage': np.float64,
                'LongStorage': np.int64,
                'IntStorage': np.int32,
                'ShortStorage': np.int16,
                'ByteStorage': np.uint8,
                'CharStorage': np.int8,
                'BoolStorage': np.bool_,
            }
            dtype = dtype_map.get(name, np.float32)
            class StorageStub:
                __name__ = name
                _dtype = dtype
            return StorageStub
        # For _rebuild_tensor_v2 — return our numpy reconstruction function
        if module == 'torch._utils' and name == '_rebuild_tensor_v2':
            return _rebuild_tensor_v2
        # OrderedDict and other safe builtins
        return super().find_class(module, name)


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None):
    """Reconstruct a tensor from storage as a numpy array."""
    data = storage.data
    if len(size) == 0:
        return data[storage_offset:storage_offset + 1].reshape(())
    
    # Compute total number of elements needed
    numel = 1
    for s in size:
        numel *= s
    
    # Use stride to extract data correctly
    arr = data[storage_offset:storage_offset + max(numel, 1)]
    
    try:
        result = arr.reshape(size)
    except ValueError:
        # Fallback for non-contiguous strides
        result = np.zeros(size, dtype=arr.dtype)
        flat = result.ravel()
        for i in range(len(flat)):
            idx = storage_offset
            rem = i
            for dim_idx, (s, st) in enumerate(zip(reversed(size), reversed(stride))):
                idx += (rem % s) * st
                rem //= s
            if idx < len(data):
                flat[i] = data[idx]
    
    return result


def load_pytorch_checkpoint(filepath: str) -> Dict[str, Any]:
    """
    Load a real PyTorch .pt checkpoint file without requiring the torch package.
    
    Returns a dict with keys matching the checkpoint structure, with tensor values
    replaced by numpy arrays.
    """
    with zipfile.ZipFile(filepath, 'r') as zf:
        # Determine archive name (usually filename without extension)
        entries = zf.namelist()
        archive_name = entries[0].split('/')[0]
        
        # Read byteorder if present
        byteorder = 'little'
        bo_path = f"{archive_name}/byteorder"
        if bo_path in entries:
            with zf.open(bo_path) as f:
                byteorder = f.read().decode().strip()
        
        # Read and unpickle data.pkl
        pkl_path = f"{archive_name}/data.pkl"
        with zf.open(pkl_path) as f:
            pkl_bytes = f.read()
        
        unpickler = _NumpyUnpickler(
            io.BytesIO(pkl_bytes),
            zip_file=zf,
            archive_name=archive_name,
            byteorder=byteorder
        )
        checkpoint = unpickler.load()
    
    return checkpoint


if __name__ == "__main__":
    import sys
    ckpt = load_pytorch_checkpoint(sys.argv[1])
    print("Checkpoint keys:", list(ckpt.keys()))
    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
        print(f"State dict layers: {len(sd)}")
        for k, v in list(sd.items())[:5]:
            shape = v.shape if hasattr(v, 'shape') else '?'
            print(f"  {k}: {shape}")
    print("epoch:", ckpt.get("epoch"))
    print("metrics:", ckpt.get("metrics"))
