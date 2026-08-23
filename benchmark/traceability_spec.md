# Production Traceability & Telemetry Blueprint (Saved for Later Phase)

This document preserves the architecture and implementation specifications for adding OpenTelemetry / Rich production tracing to the Mac Quantization and KV Cache offloading pipeline.

---

## 1. Core Observability Spans & Metrics

In production, each request will emit structured spans with millisecond timings and Apple Silicon Unified Memory telemetry:

```
[Request Start]
  ├── 1. span: "prefill"
  │     ├── metrics: `ttft_ms` (Time To First Token)
  │     ├── metrics: `prefill_tok_per_sec`
  │     └── metrics: `prompt_token_count`
  │
  ├── 2. span: "decode"
  │     ├── metrics: `tpot_ms` (Time Per Output Token)
  │     ├── metrics: `decode_tok_per_sec`
  │     ├── metrics: `inter_token_jitter_ms`
  │     └── metrics: `generated_token_count`
  │
  ├── 3. span: "kv_cache_offload"
  │     ├── metrics: `kv_serialize_ms`
  │     ├── metrics: `kv_disk_write_bandwidth_mb_s`
  │     └── metrics: `kv_file_size_bytes`
  │
  ├── 4. span: "kv_cache_onload"
  │     ├── metrics: `kv_deserialize_ms`
  │     ├── metrics: `kv_disk_read_bandwidth_mb_s`
  │     └── metrics: `first_token_latency_after_restore_ms`
  │
  └── 5. span: "memory_footprint"
        ├── metrics: `process_rss_mb`
        ├── metrics: `metal_active_memory_mb`
        ├── metrics: `metal_peak_memory_mb`
        └── metrics: `persistent_leak_detected` (Boolean)
```

---

## 2. Telemetry Collector Schema (JSON)

```json
{
  "trace_id": "req-892f3a",
  "timestamp": "2026-08-16T09:20:00Z",
  "model": "qwen2.5-coder-3b-mlx-4bit",
  "kv_precision": "fp16",
  "context_length": 2048,
  "timings": {
    "ttft_ms": 112.4,
    "tpot_ms": 19.2,
    "decode_tps": 52.1,
    "kv_offload_ms": 6.8,
    "kv_onload_ms": 4.1
  },
  "memory": {
    "rss_mb": 2210.5,
    "metal_active_mb": 1845.0,
    "metal_peak_mb": 2100.2
  },
  "status": "SUCCESS"
}
```

---

## 3. Activation Roadmap
* When ready to activate production tracing, integrate `TraceLogger` and `@trace_span` wrappers into `kv_cache_manager.py` and `mlx_lm.server`.
