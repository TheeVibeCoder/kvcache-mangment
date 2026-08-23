#!/usr/bin/env python3
"""
====================================================================================================
SIDE-BY-SIDE BENCHMARK: V1 (In-RAM Retention) vs V2 (Turn-by-Turn Disaggregated Offload)
====================================================================================================
Measures and compares exact RAM held during idle think-time across multi-turn conversation.
====================================================================================================
"""

import os
import sys
import gc
import time
import mlx.core as mx
from mlx_lm import load, generate

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from v1.kv_cache_manager import LocalKVCacheStore as V1Store
from v2.kv_cache_engine_v2 import TurnBasedKVCacheEngine as V2Engine

def get_metal_active_mb():
    try:
        fn = getattr(mx, "get_active_memory", getattr(mx.metal, "get_active_memory", None))
        return round(fn() / (1024 * 1024), 2) if fn else 0.0
    except Exception:
        return 0.0

def run_comparison():
    print("=" * 85)
    print("🔬 MEMORY BENCHMARK: V1 (In-RAM Caching) vs V2 (Turn-by-Turn SSD Disaggregation)")
    print("=" * 85)

    model_path = "models/qwen2.5-coder-3b-mlx-4bit"
    if not os.path.exists(model_path):
        model_path = "../models/qwen2.5-coder-3b-mlx-4bit"

    prompts = [
        ("Turn 1 (Codebase Ingest)", "Here is a Python class for a PaymentGateway with Stripe, PayPal, and Crypto support. " * 15 + "Question: Summarize what PaymentGateway does in 1 sentence."),
        ("Turn 2 (Follow-up 1)", "Now explain how the Stripe transaction validation logic works."),
        ("Turn 3 (Follow-up 2)", "Write a test case using pytest for the PayPal refund error handler."),
        ("Turn 4 (Follow-up 3)", "Add a retry decorator with exponential backoff for network timeouts.")
    ]

    # ---------------------------------------------------------------------------------------------
    # 1. RUN V1 (In-Memory KV Cache Retention)
    # ---------------------------------------------------------------------------------------------
    print("\n[Phase 1] Running V1 (Holding active KV cache continuously in Mac RAM)...")
    model_v1, tokenizer_v1 = load(model_path)
    store_v1 = V1Store(storage_dir="kv_cache_store")
    cache_v1 = store_v1.create_cache(model_v1, bits=8)
    
    v1_results = []
    base_ram_v1 = get_metal_active_mb()

    conversation_history = []
    for turn_idx, (turn_name, prompt_text) in enumerate(prompts, 1):
        conversation_history.append({"role": "user", "content": prompt_text})
        formatted = tokenizer_v1.apply_chat_template(conversation_history, tokenize=False, add_generation_prompt=True)
        
        t0 = time.perf_counter()
        resp = generate(model_v1, tokenizer_v1, prompt=formatted, prompt_cache=cache_v1, max_tokens=60)
        t1 = time.perf_counter()
        conversation_history.append({"role": "assistant", "content": resp})

        # Measure RAM while IDLE (waiting for user input)
        idle_ram_v1 = get_metal_active_mb()
        kv_held_mb_v1 = round(max(0.0, idle_ram_v1 - base_ram_v1), 2)
        total_tokens_v1 = cache_v1[0].offset

        v1_results.append({
            "turn": turn_name,
            "tokens": total_tokens_v1,
            "idle_ram_mb": idle_ram_v1,
            "kv_held_in_ram_mb": kv_held_mb_v1,
            "time_sec": round(t1 - t0, 2)
        })
        print(f"  • {turn_name:<26} | Context: {total_tokens_v1:<4} tok | 🔴 Idle KV RAM Held: {kv_held_mb_v1:>6.2f} MB (Total RAM: {idle_ram_v1:.1f} MB)")

    # Clean up V1 before running V2
    del model_v1, tokenizer_v1, cache_v1
    gc.collect()
    mx.clear_cache()
    time.sleep(1)

    # ---------------------------------------------------------------------------------------------
    # 2. RUN V2 (Turn-by-Turn SSD Eviction & JIT Onload)
    # ---------------------------------------------------------------------------------------------
    print("\n[Phase 2] Running V2 (Turn-by-Turn SSD Offload & Instant RAM Purging)...")
    v2_session = "v1_v2_comparison_test"
    v2_file = f"kv_cache_store/{v2_session}.safetensors"
    if os.path.exists(v2_file):
        os.remove(v2_file)

    engine_v2 = V2Engine(model_path=model_path, storage_dir="kv_cache_store", kv_bits=8)
    v2_results = []

    for turn_idx, (turn_name, prompt_text) in enumerate(prompts, 1):
        resp, meta = engine_v2.ask(prompt_text, session_id=v2_session, max_tokens=60)
        
        idle_ram_v2 = meta["memory_mb"]["idle_after_purge"]
        kv_held_mb_v2 = meta["memory_mb"]["idle_kv_held"]
        total_tokens_v2 = meta["tokens"]["total_context"]
        onload_ms = meta["latency"]["onload_ms"]

        v2_results.append({
            "turn": turn_name,
            "tokens": total_tokens_v2,
            "idle_ram_mb": idle_ram_v2,
            "kv_held_in_ram_mb": kv_held_mb_v2,
            "onload_ms": onload_ms,
            "time_sec": meta["latency"]["generation_sec"]
        })
        print(f"  • {turn_name:<26} | Context: {total_tokens_v2:<4} tok | 🟢 Idle KV RAM Held: {kv_held_mb_v2:>6.2f} MB (JIT Onload: {onload_ms:.1f} ms)")

    # ---------------------------------------------------------------------------------------------
    # 3. SIDE-BY-SIDE SUMMARY TABLE
    # ---------------------------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("📊 FINAL SIDE-BY-SIDE COMPARISON TABLE")
    print("=" * 85)
    print(f"{'Turn':<25} | {'Context':<10} | {'V1 Idle KV RAM Held':<20} | {'V2 Idle KV RAM Held':<20} | {'RAM Saved':<10}")
    print("-" * 85)
    for r1, r2 in zip(v1_results, v2_results):
        saved = r1["kv_held_in_ram_mb"] - r2["kv_held_in_ram_mb"]
        print(f"{r1['turn']:<25} | {r1['tokens']:<4} tokens | 🔴 {r1['kv_held_in_ram_mb']:>6.2f} MB held in RAM | 🟢 {r2['kv_held_in_ram_mb']:>6.2f} MB in RAM    | 💰 {saved:>6.2f} MB")
    print("=" * 85)

if __name__ == "__main__":
    run_comparison()
