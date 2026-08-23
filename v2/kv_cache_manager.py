"""
====================================================================================================
VERSION 2: KV Cache Quantization & Incremental Delta SSD Storage
====================================================================================================
"""

import os
import sys
import time
import json
from typing import List, Any, Optional, Tuple
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.models.cache import make_prompt_cache, QuantizedKVCache

class LocalKVCacheStore:
    """
    Manages Quantized KV Cache storage with Incremental Delta Chunking.
    Persists only the new token slices per turn to maximize speed and minimize SSD wear.
    """
    def __init__(self, storage_dir: str = "kv_cache_store", trace_log: str = "kv_cache_events.log"):
        self.storage_dir = storage_dir
        self.trace_log = trace_log
        os.makedirs(self.storage_dir, exist_ok=True)

    def _get_session_dir(self, session_id: str) -> str:
        safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in session_id)
        session_dir = os.path.join(self.storage_dir, safe_id)
        os.makedirs(session_dir, exist_ok=True)
        return session_dir

    def _get_manifest_path(self, session_id: str) -> str:
        return os.path.join(self._get_session_dir(session_id), "manifest.json")

    def _read_manifest(self, session_id: str) -> dict:
        path = self._get_manifest_path(session_id)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"session_id": session_id, "chunks": [], "total_tokens": 0, "bits": 8, "group_size": 64}

    def _write_manifest(self, session_id: str, manifest: dict):
        path = self._get_manifest_path(session_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def exists(self, session_id: str) -> bool:
        # Check chunked manifest first
        manifest_path = self._get_manifest_path(session_id)
        if os.path.exists(manifest_path) and len(self._read_manifest(session_id).get("chunks", [])) > 0:
            return True
        # Check monolithic legacy file for backwards compatibility
        safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in session_id)
        return os.path.exists(os.path.join(self.storage_dir, f"{safe_id}.safetensors"))

    def create_cache(self, model: Any, bits: int = 8, group_size: int = 64) -> List[Any]:
        if bits in (4, 8):
            return [QuantizedKVCache(group_size=group_size, bits=bits) for _ in model.layers]
        return make_prompt_cache(model)

    def offload_delta(
        self,
        cache: List[Any],
        session_id: str,
        offset_before: int,
        offset_after: int
    ) -> Tuple[str, float, float]:
        """
        Extracts and persists only the newly generated token slice [offset_before:offset_after].
        Returns: (chunk_filepath, chunk_size_kb, duration_ms)
        """
        t0 = time.perf_counter()
        session_dir = self._get_session_dir(session_id)
        manifest = self._read_manifest(session_id)

        chunk_idx = len(manifest.get("chunks", []))
        chunk_filename = f"chunk_{chunk_idx:04d}.safetensors"
        chunk_path = os.path.join(session_dir, chunk_filename)

        tensor_dict = {}
        is_quantized = len(cache) > 0 and isinstance(cache[0], QuantizedKVCache)
        bits = cache[0].bits if is_quantized else 16
        group_size = cache[0].group_size if is_quantized else 64

        tokens_in_delta = offset_after - offset_before

        tensor_dict["_meta_is_quantized"] = mx.array([1 if is_quantized else 0])
        tensor_dict["_meta_bits"] = mx.array([bits])
        tensor_dict["_meta_group_size"] = mx.array([group_size])
        tensor_dict["_meta_offset_before"] = mx.array([offset_before])
        tensor_dict["_meta_offset_after"] = mx.array([offset_after])

        for i, layer in enumerate(cache):
            if is_quantized:
                tensor_dict[f"l{i}_k_arr"] = layer.keys[0][:, :, offset_before:offset_after, :]
                tensor_dict[f"l{i}_k_sc"] = layer.keys[1][:, :, offset_before:offset_after, :]
                tensor_dict[f"l{i}_k_bi"] = layer.keys[2][:, :, offset_before:offset_after, :]
                tensor_dict[f"l{i}_v_arr"] = layer.values[0][:, :, offset_before:offset_after, :]
                tensor_dict[f"l{i}_v_sc"] = layer.values[1][:, :, offset_before:offset_after, :]
                tensor_dict[f"l{i}_v_bi"] = layer.values[2][:, :, offset_before:offset_after, :]
            else:
                if hasattr(layer, "keys") and layer.keys is not None:
                    tensor_dict[f"l{i}_k"] = layer.keys[:, :, offset_before:offset_after, :]
                    tensor_dict[f"l{i}_v"] = layer.values[:, :, offset_before:offset_after, :]

        mx.save_safetensors(chunk_path, tensor_dict)
        t1 = time.perf_counter()

        duration_ms = (t1 - t0) * 1000
        size_kb = os.path.getsize(chunk_path) / 1024

        manifest["bits"] = bits
        manifest["group_size"] = group_size
        manifest["total_tokens"] = offset_after
        manifest["chunks"].append({
            "chunk_idx": chunk_idx,
            "filename": chunk_filename,
            "offset_before": offset_before,
            "offset_after": offset_after,
            "token_count": tokens_in_delta,
            "size_kb": round(size_kb, 2),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")
        })
        self._write_manifest(session_id, manifest)

        return chunk_path, size_kb, duration_ms

    def onload_from_disk(self, model: Any, session_id: str) -> Optional[List[Any]]:
        """
        Loads all delta chunks and concatenates them seamlessly into in-memory QuantizedKVCache.
        """
        session_dir = self._get_session_dir(session_id)
        manifest = self._read_manifest(session_id)
        chunks = manifest.get("chunks", [])

        # Check for legacy monolithic file if no chunks
        if not chunks:
            safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in session_id)
            legacy_file = os.path.join(self.storage_dir, f"{safe_id}.safetensors")
            if os.path.exists(legacy_file):
                tensor_dict = mx.load(legacy_file)
                bits = int(tensor_dict["_meta_bits"][0].item())
                group_size = int(tensor_dict["_meta_group_size"][0].item())
                cache = [QuantizedKVCache(group_size=group_size, bits=bits) for _ in model.layers]
                for i, layer_cache in enumerate(cache):
                    layer_cache.keys = (
                        tensor_dict[f"layer_{i}_k_arr"],
                        tensor_dict[f"layer_{i}_k_sc"],
                        tensor_dict[f"layer_{i}_k_bi"],
                    )
                    layer_cache.values = (
                        tensor_dict[f"layer_{i}_v_arr"],
                        tensor_dict[f"layer_{i}_v_sc"],
                        tensor_dict[f"layer_{i}_v_bi"],
                    )
                    layer_cache.offset = int(tensor_dict[f"layer_{i}_offset"][0].item())
                return cache
            return None

        bits = manifest.get("bits", 8)
        group_size = manifest.get("group_size", 64)
        is_quantized = bits in (4, 8)

        chunk_tensors = []
        for chk in chunks:
            chk_file = os.path.join(session_dir, chk["filename"])
            if os.path.exists(chk_file):
                chunk_tensors.append(mx.load(chk_file))

        if not chunk_tensors:
            return None

        if is_quantized:
            cache = [QuantizedKVCache(group_size=group_size, bits=bits) for _ in model.layers]
            for i, layer in enumerate(cache):
                if len(chunk_tensors) == 1:
                    ct = chunk_tensors[0]
                    layer.keys = (ct[f"l{i}_k_arr"], ct[f"l{i}_k_sc"], ct[f"l{i}_k_bi"])
                    layer.values = (ct[f"l{i}_v_arr"], ct[f"l{i}_v_sc"], ct[f"l{i}_v_bi"])
                else:
                    k_arr = mx.concatenate([ct[f"l{i}_k_arr"] for ct in chunk_tensors], axis=2)
                    k_sc = mx.concatenate([ct[f"l{i}_k_sc"] for ct in chunk_tensors], axis=2)
                    k_bi = mx.concatenate([ct[f"l{i}_k_bi"] for ct in chunk_tensors], axis=2)
                    v_arr = mx.concatenate([ct[f"l{i}_v_arr"] for ct in chunk_tensors], axis=2)
                    v_sc = mx.concatenate([ct[f"l{i}_v_sc"] for ct in chunk_tensors], axis=2)
                    v_bi = mx.concatenate([ct[f"l{i}_v_bi"] for ct in chunk_tensors], axis=2)
                    layer.keys = (k_arr, k_sc, k_bi)
                    layer.values = (v_arr, v_sc, v_bi)

                layer.offset = manifest["total_tokens"]
        else:
            cache = make_prompt_cache(model)
            for i, layer in enumerate(cache):
                if len(chunk_tensors) == 1:
                    ct = chunk_tensors[0]
                    layer.keys = ct[f"l{i}_k"]
                    layer.values = ct[f"l{i}_v"]
                else:
                    k = mx.concatenate([ct[f"l{i}_k"] for ct in chunk_tensors], axis=2)
                    v = mx.concatenate([ct[f"l{i}_v"] for ct in chunk_tensors], axis=2)
                    layer.keys = k
                    layer.values = v
                layer.offset = manifest["total_tokens"]

        return cache

    def free_gpu_memory(self):
        mx.clear_cache()

    def list_sessions(self) -> List[dict]:
        if not os.path.exists(self.storage_dir):
            return []
        sessions = []
        for s in os.listdir(self.storage_dir):
            man_path = os.path.join(self.storage_dir, s, "manifest.json")
            if os.path.exists(man_path):
                man = self._read_manifest(s)
                sessions.append({
                    "session_id": s,
                    "total_tokens": man.get("total_tokens", 0),
                    "chunk_count": len(man.get("chunks", [])),
                    "total_size_kb": sum(c.get("size_kb", 0) for c in man.get("chunks", []))
                })
        return sessions
