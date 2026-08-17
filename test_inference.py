"""
Inference test script demonstrating 8-bit KV Cache offload & onload with LocalKVCacheStore.
"""

import sys
import os
import time
from mlx_lm import load, generate
from kv_cache_manager import LocalKVCacheStore

def run_kv_inference_demo():
    model_path = "models/qwen2.5-coder-3b-mlx-4bit"
    if not os.path.exists(model_path):
        print(f"[!] Error: Model path '{model_path}' does not exist.")
        return

    print("=" * 70)
    print(f"[*] Step 1: Loading 4-bit Model from: {model_path}")
    print("=" * 70)
    model, tokenizer = load(model_path)

    # Initialize KV Cache Store (uses bits=8 by default)
    store = LocalKVCacheStore(storage_dir="kv_cache_store")
    cache_id = "demo_code_session"

    print("\n" + "=" * 70)
    print("[*] Step 2: Creating 8-bit Quantized KV Cache in RAM")
    print("=" * 70)
    kv_cache = store.create_cache(model, bits=8)
    print(f"[✓] Created 8-bit KV Cache across {len(kv_cache)} model layers.")

    # Context Prompt to ingest into KV cache
    repo_context = """
# System Context: Codebase Architecture
class DatabaseConnection:
    def __init__(self, host: str = "localhost", port: int = 5432):
        self.host = host
        self.port = port
        self.connected = False

    def connect(self):
        self.connected = True
        return f"Connected to {self.host}:{self.port}"

    def query(self, sql: str):
        if not self.connected:
            raise ConnectionError("Database not connected")
        return f"Executed: {sql}"
"""
    initial_prompt = f"Here is the database code:\n{repo_context}\nExplain what DatabaseConnection does in 2 bullet points."
    
    formatted_initial = tokenizer.apply_chat_template(
        [{"role": "user", "content": initial_prompt}],
        tokenize=False,
        add_generation_prompt=True
    )

    print("\n" + "=" * 70)
    print("[*] Step 3: Generating Response & Populating 8-bit KV Cache")
    print("=" * 70)
    print(f"[Prompt]: {initial_prompt.strip()}\n")
    print("[Model Output (Turn 1)]:")
    print("-" * 70)
    t0 = time.perf_counter()
    response1 = generate(
        model, 
        tokenizer, 
        prompt=formatted_initial, 
        prompt_cache=kv_cache,
        max_tokens=150, 
        verbose=True
    )
    t1 = time.perf_counter()
    print("-" * 70)
    print(f"[⏱ Generation Time]: {(t1 - t0):.2f}s")
    print(f"[KV Cache Token Offset]: {kv_cache[0].offset} tokens cached")

    # Step 4: Offload 8-bit KV Cache to disk
    print("\n" + "=" * 70)
    print("[*] Step 4: Offloading 8-bit KV Cache to Local SSD (.safetensors)")
    print("=" * 70)
    t_save_0 = time.perf_counter()
    saved_path = store.offload_to_disk(kv_cache, cache_id)
    t_save_1 = time.perf_counter()
    file_size_kb = os.path.getsize(saved_path) / 1024
    print(f"[✓] Saved KV Cache to: {saved_path}")
    print(f"[✓] File Size on SSD: {file_size_kb:.2f} KB")
    print(f"[⏱ Offload Time]: {(t_save_1 - t_save_0) * 1000:.2f} ms")

    # Step 5: Free memory
    print("\n" + "=" * 70)
    print("[*] Step 5: Freeing Unified RAM / Metal Cache")
    print("=" * 70)
    del kv_cache
    store.free_gpu_memory()
    print("[✓] GPU cache cleared. Memory freed.")

    # Step 6: Onload 8-bit KV Cache from disk
    print("\n" + "=" * 70)
    print("[*] Step 6: Onloading 8-bit KV Cache from SSD into Memory")
    print("=" * 70)
    t_load_0 = time.perf_counter()
    restored_cache = store.onload_from_disk(model, cache_id)
    t_load_1 = time.perf_counter()
    load_time_ms = (t_load_1 - t_load_0) * 1000
    print(f"[✓] Successfully restored KV Cache from {saved_path}")
    print(f"[⏱ Onload Restore Time]: {load_time_ms:.2f} ms")
    print(f"[✓] Restored token offset: {restored_cache[0].offset} tokens")

    # Step 7: Multi-turn inference using restored 8-bit KV Cache (Zero prefill recomputation!)
    followup_query = "Now write a method `disconnect()` for this class in Python."
    formatted_followup = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": initial_prompt},
            {"role": "assistant", "content": response1},
            {"role": "user", "content": followup_query}
        ],
        tokenize=False,
        add_generation_prompt=True
    )

    print("\n" + "=" * 70)
    print("[*] Step 7: Follow-up Generation Using Restored 8-bit KV Cache")
    print("=" * 70)
    print(f"[Follow-up Prompt]: {followup_query}\n")
    print("[Model Output (Turn 2)]:")
    print("-" * 70)
    t2 = time.perf_counter()
    response2 = generate(
        model, 
        tokenizer, 
        prompt=formatted_followup, 
        prompt_cache=restored_cache,
        max_tokens=150, 
        verbose=True
    )
    t3 = time.perf_counter()
    print("-" * 70)
    print(f"[⏱ Generation Time]: {(t3 - t2):.2f}s")
    print("=" * 70)
    print("[✓] Full 8-bit KV Cache Pipeline Test Completed Successfully!")
    print("=" * 70)

if __name__ == "__main__":
    run_kv_inference_demo()

