# Apple Silicon Quantized KV Cache Management

A high-performance, persistent **8-bit / 4-bit Quantized KV Cache Manager** for Apple Silicon (M-series) Macs using **Apple MLX** and `.safetensors`.

---

## 🌟 Highlights

- **8-bit / 4-bit Quantization**: Reduces KV Cache memory footprint by 2× to 4× with zero perceptible loss in generation quality.
- **Ultra-fast NVMe Persistence**: Serializes in-memory KV caches to `.safetensors` on SSD and restores in **~8 ms**, bypassing the costly prompt prefill phase.
- **Cross-Session Reuse & Prefix Caching**: Save and restore base codebase caches across multiple conversation threads or sessions.
- **Zero Memory Leaks**: Clean unified memory reclamation using MLX Metal buffer cache controls.

---

## 📁 Repository Structure

```text
├── kv_cache_manager.py       # Core Quantized KV Cache Manager & Safetensors persistence
├── verify_kv_reuse.py        # Benchmark verifying 100% cache retrieval & speedup
├── test_inference.py         # End-to-end inference verification script
├── inspect_safetensors.py    # Tensor inspector for .safetensors KV cache files
├── requirements.txt          # Python dependencies
└── .gitignore
```

---

## 🚀 Usage

### 1. Basic In-Memory & Disk Offloading

```python
from mlx_lm import load, generate
from kv_cache_manager import LocalKVCacheStore

# Load model
model, tokenizer = load("models/qwen2.5-coder-3b-mlx-4bit")
store = LocalKVCacheStore(storage_dir="kv_cache_store")

# 1. Create an 8-bit quantized KV cache
kv_cache = store.create_cache(model, bits=8)

# 2. Run inference (populates the KV cache)
response = generate(model, tokenizer, prompt="Hello!", prompt_cache=kv_cache)

# 3. Offload to SSD & free RAM
store.offload_to_disk(kv_cache, session_id="my_session")
store.free_gpu_memory()

# 4. Restore instantly (8ms) in a new run
kv_restored = store.onload_from_disk(model, session_id="my_session")
```

### 2. Verify Cache Speedup

```bash
python verify_kv_reuse.py
```

---

## 🛡️ License

MIT License. Designed for Apple Silicon unified memory architecture.
