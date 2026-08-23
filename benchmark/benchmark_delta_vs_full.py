#!/usr/bin/env python3
"""
====================================================================================================
HEAD-TO-HEAD BENCHMARK: V2 (Full-File Rewrite) vs V3 (Incremental Delta Appending)
====================================================================================================
Compares:
1. Offload Latency (ms) on each turn as conversation context scales
2. Total disk bytes written to SSD (MB)
3. Onload Latency & Accuracy
====================================================================================================
"""

import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from v2.kv_cache_engine_v2 import TurnBasedKVCacheEngine as V2Engine
from v3.kv_cache_engine_v3 import IncrementalTurnEngine as V3Engine

def run_benchmark():
    print("=" * 90)
    print("⚡ HEAD-TO-HEAD BENCHMARK: V2 (Full Monolithic Rewrite) vs V3 (Incremental Delta)")
    print("=" * 90)

    model_path = "models/qwen2.5-coder-3b-mlx-4bit"
    if not os.path.exists(model_path):
        model_path = "../models/qwen2.5-coder-3b-mlx-4bit"

    prompts = [
        ("Turn 1 (Large Context Ingest)", "Here is a Python distributed cache implementation with Raft consensus, heartbeats, and leader election: " * 12 + "Question: What algorithm is used?"),
        ("Turn 2 (Follow-up 1)", "Explain how leader election handles network partitions."),
        ("Turn 3 (Follow-up 2)", "Write a Python function for heartbeat timeout detection."),
        ("Turn 4 (Follow-up 3)", "How does the follower log replication handle uncommitted entries?"),
        ("Turn 5 (Follow-up 4)", "Provide a quick summary of all components discussed.")
    ]

    # ---------------------------------------------------------------------------------------------
    # 1. RUN V2 (Monolithic Rewrite on Every Turn)
    # ---------------------------------------------------------------------------------------------
    print("\n[Phase 1] Benchmarking V2 (Rewriting full accumulated cache on every turn)...")
    v2_session = "bench_v2_monolithic"
    engine_v2 = V2Engine(model_path=model_path, storage_dir="kv_cache_store", kv_bits=8)
    v2_results = []
    total_v2_bytes_written = 0

    for turn_idx, (turn_name, p) in enumerate(prompts, 1):
        _, meta = engine_v2.ask(p, session_id=v2_session, max_tokens=40)
        offload_ms = meta["latency"]["offload_ms"]
        file_kb = meta["storage"]["ssd_file_size_kb"]
        total_v2_bytes_written += (file_kb * 1024)
        tokens = meta["tokens"]["total_context"]
        v2_results.append({
            "turn": turn_name,
            "tokens": tokens,
            "offload_ms": offload_ms,
            "file_kb": file_kb
        })
        print(f"  • {turn_name:<28} | {tokens:<4} tok | 💾 Full Rewrite: {offload_ms:>5.2f} ms ({file_kb:>8.1f} KB written)")

    # ---------------------------------------------------------------------------------------------
    # 2. RUN V3 (Incremental Delta-Only Appending)
    # ---------------------------------------------------------------------------------------------
    print("\n[Phase 2] Benchmarking V3 (Writing *only* new delta slices on every turn)...")
    v3_session = "bench_v3_delta"
    engine_v3 = V3Engine(model_path=model_path, storage_dir="kv_cache_store_v3", kv_bits=8)
    v3_results = []
    total_v3_bytes_written = 0

    for turn_idx, (turn_name, p) in enumerate(prompts, 1):
        _, meta = engine_v3.ask(p, session_id=v3_session, max_tokens=40)
        delta_ms = meta["latency"]["delta_write_ms"]
        chunk_kb = meta["delta_storage"]["delta_chunk_kb"]
        total_v3_bytes_written += (chunk_kb * 1024)
        tokens = meta["tokens"]["total_context"]
        v3_results.append({
            "turn": turn_name,
            "tokens": tokens,
            "delta_ms": delta_ms,
            "chunk_kb": chunk_kb
        })
        print(f"  • {turn_name:<28} | {tokens:<4} tok | ⚡ Delta Append: {delta_ms:>5.2f} ms ({chunk_kb:>8.1f} KB written)")

    # ---------------------------------------------------------------------------------------------
    # 3. SIDE-BY-SIDE REPORT
    # ---------------------------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("📊 FINAL HEAD-TO-HEAD COMPARISON: FULL REWRITE (V2) vs DELTA APPEND (V3)")
    print("=" * 90)
    print(f"{'Turn':<26} | {'Tokens':<8} | {'V2 Full Write Time':<20} | {'V3 Delta Write Time':<20} | {'Speedup':<8}")
    print("-" * 90)
    for r2, r3 in zip(v2_results, v3_results):
        speedup = f"{(r2['offload_ms'] / max(r3['delta_ms'], 0.1)):.1f}x"
        print(f"{r2['turn']:<26} | {r2['tokens']:<8} | 💾 {r2['offload_ms']:>6.2f} ms ({r2['file_kb']:.0f} KB)   | ⚡ {r3['delta_ms']:>6.2f} ms ({r3['chunk_kb']:.0f} KB)    | 🚀 {speedup}")
    print("=" * 90)

    v2_mb = total_v2_bytes_written / (1024 * 1024)
    v3_mb = total_v3_bytes_written / (1024 * 1024)
    saved_mb = v2_mb - v3_mb
    pct = (saved_mb / v2_mb) * 100

    print(f"💾 Total SSD Volume Written (V2 Monolithic) : {v2_mb:.2f} MB")
    print(f"⚡ Total SSD Volume Written (V3 Delta Only) : {v3_mb:.2f} MB")
    print(f"💰 SSD Write Reduction                       : {saved_mb:.2f} MB saved ({pct:.1f}% less disk wear!)")
    print("=" * 90)

if __name__ == "__main__":
    run_benchmark()
