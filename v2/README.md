# Version 2: Turn-by-Turn Disaggregated KV Cache Engine with Incremental Delta Appending

A high-performance **Turn-by-Turn Disaggregated KV Cache Engine** for Apple Silicon Macs using **Apple MLX** and `.safetensors`.

---

## 🌟 Key Features

1. **Zero Idle RAM Held:** The instant the model finishes answering, active KV attention tensors are flushed to SSD, and unified memory is reclaimed with `mx.clear_cache()`. Only base weights remain in memory.
2. **Incremental Delta Appending:** Instead of rewriting a full monolithic file after every question, V2 extracts and persists *only* the new slice `[offset_before:offset_after]` as a lightweight delta chunk (`chunk_0000.safetensors`, `chunk_0001.safetensors`, etc.).
3. **Sub-4ms Offload Latency:** Writing small delta chunks takes **~3.3 ms**, reducing SSD write volume by **80.2%**.
4. **Sub-1ms JIT Onload:** Seamlessly concatenates chunked tensors on-demand before prompt decoding.

---

## 🚀 Usage

### 1. Interactive Agent with Real-Time Delta HUD:
```bash
.venv/bin/python3 v2/codex_cli_v2.py
```

### 2. Run Verification Benchmark:
```bash
.venv/bin/python3 v2/verify_v2_per_turn.py
```
