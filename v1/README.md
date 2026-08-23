# 📦 Version 1: Session-Level Persistent KV Cache Offloading

### 📖 Architecture Overview:
* **Offload Trigger:** When the user closes the app, exits the CLI (`/exit`), or branches (`/branch`).
* **Active State:** Keeps the quantized KV cache in RAM throughout the conversation for fast turn-by-turn generation.
* **Storage Format:** Serializes 8-bit & 4-bit KV caches into `.safetensors` on local NVMe SSD.
* **Onload Speed:** Restores pre-computed sessions in **~8 ms**.

---

### 📂 Files in `v1/`:
* `codex_cli.py`: Interactive Terminal Agent with session branching and multi-file ingestion.
* `kv_cache_manager.py`: Core `LocalKVCacheStore` handling 4-bit/8-bit quantization and `.safetensors` I/O.
* `verify_kv_reuse.py`: Script verifying 100% token preservation and cold-start bypass.
* `test_inference.py`: End-to-end inference and persistence test.
* `inspect_safetensors.py`: Utility inspecting tensor shapes, scales, biases, and metadata on disk.

---

### 🚀 How to Run V1:
```bash
# Run the Interactive Terminal Agent
python v1/codex_cli.py

# Run the Verification Test
python v1/verify_kv_reuse.py
```
