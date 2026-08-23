#!/usr/bin/env python3
"""
====================================================================================================
VERSION 1: Interactive Terminal Codex CLI
====================================================================================================
Usage:
  python v1/codex_cli.py
  python v1/codex_cli.py --session my_project
"""

import os
import sys
import argparse
import time

# Ensure local v1 imports resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mlx_lm import load, stream_generate
from kv_cache_manager import LocalKVCacheStore

def run_interactive_codex(session_id: str = "default_session", model_path: str = "models/qwen2.5-coder-3b-mlx-4bit", base_session: str = None):
    print("=" * 75)
    print("  🚀 CODEX TERMINAL AGENT V1 (Session-Level Persistent KV Cache)")
    print("=" * 75)

    if not os.path.exists(model_path):
        model_path = "../models/qwen2.5-coder-3b-mlx-4bit"

    if not os.path.exists(model_path):
        print(f"[!] Error: Model path '{model_path}' not found.")
        return

    print(f"[*] Loading model from '{model_path}'...")
    model, tokenizer = load(model_path)
    store = LocalKVCacheStore(storage_dir="kv_cache_store")

    # Load existing KV cache if present, else check base_session, else create fresh
    if store.exists(session_id):
        print(f"[*] Restoring existing 8-bit session cache '{session_id}' from SSD...")
        t0 = time.perf_counter()
        kv_cache = store.onload_from_disk(model, session_id)
        t1 = time.perf_counter()
        offset = kv_cache[0].offset if kv_cache else 0
        print(f"[✓] Restored {offset} cached tokens in {(t1 - t0)*1000:.2f} ms.")
    elif base_session and store.exists(base_session):
        print(f"[*] Initializing session '{session_id}' branched from base '{base_session}'...")
        t0 = time.perf_counter()
        kv_cache = store.onload_from_disk(model, base_session)
        t1 = time.perf_counter()
        offset = kv_cache[0].offset if kv_cache else 0
        print(f"[✓] Seeded {offset} cached tokens from base '{base_session}' in {(t1 - t0)*1000:.2f} ms.")
    else:
        print(f"[*] Initializing fresh 8-bit KV cache for session '{session_id}'...")
        kv_cache = store.create_cache(model, bits=8)
        print(f"[✓] 8-bit KV Cache ready across {len(kv_cache)} layers.")

    history = []

    print("\n" + "-" * 75)
    print("Commands:")
    print("  /sessions                : List all saved session caches on SSD")
    print("  /branch <new_session>    : Branch current KV cache into a new independent session")
    print("  /load <session_name>     : Switch and load another session cache from SSD")
    print("  /read <filepath>         : Ingest a codebase file directly into 8-bit KV cache")
    print("  /save                    : Explicitly persist active KV cache to disk right now")
    print("  /clear                   : Clear current session cache and start fresh")
    print("  /exit, /quit             : Save and exit")
    print("-" * 75)

    last_response_text = ""

    while True:
        try:
            user_input = input(f"\n[{session_id}] 👤 You > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("/exit", "/quit"):
                print("[*] Persisting 8-bit KV Cache to disk before exit...")
                saved_path = store.offload_to_disk(kv_cache, session_id)
                print(f"[✓] Cache saved to: {saved_path}")
                store.free_gpu_memory()
                print("[*] GPU memory freed. Goodbye!")
                break

            elif user_input.startswith("/branch "):
                new_session = user_input.split(maxsplit=1)[1].strip()
                if not new_session:
                    print("[!] Usage: /branch <new_session_name>")
                    continue
                print(f"[*] Branching current context ({kv_cache[0].offset} tokens) to new session '{new_session}'...")
                saved_path = store.offload_to_disk(kv_cache, new_session)
                session_id = new_session
                print(f"[✓] Successfully branched! Now active on session: '{session_id}'")
                continue

            elif user_input.startswith("/load "):
                load_session = user_input.split(maxsplit=1)[1].strip()
                if not load_session:
                    print("[!] Usage: /load <session_name>")
                    continue
                if not store.exists(load_session):
                    print(f"[!] Session '{load_session}' not found in `kv_cache_store/`.")
                    continue
                # Save current
                store.offload_to_disk(kv_cache, session_id)
                print(f"[*] Loading session '{load_session}' from SSD...")
                t0 = time.perf_counter()
                kv_cache = store.onload_from_disk(model, load_session)
                t1 = time.perf_counter()
                session_id = load_session
                print(f"[✓] Loaded session '{session_id}' ({kv_cache[0].offset} tokens) in {(t1 - t0)*1000:.2f} ms.")
                continue

            elif user_input == "/sessions":
                caches = store.list_caches()
                print(f"[*] Available session caches on SSD ({len(caches)}):")
                for c in caches:
                    marker = " (Active)" if c == session_id else ""
                    print(f"   • {c}{marker}")
                continue

            elif user_input == "/save":
                saved_path = store.offload_to_disk(kv_cache, session_id)
                print(f"[✓] KV Cache explicitly saved to: {saved_path} ({kv_cache[0].offset} tokens)")
                continue

            elif user_input == "/clear":
                print("[*] Clearing active KV Cache...")
                del kv_cache
                store.free_gpu_memory()
                kv_cache = store.create_cache(model, bits=8)
                history = []
                print("[✓] Session cache reset to 0 tokens.")
                continue

            elif user_input.startswith("/read "):
                filepath = user_input.split(maxsplit=1)[1].strip()
                if not os.path.exists(filepath):
                    print(f"[!] File not found: '{filepath}'")
                    continue
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                print(f"[*] Ingesting '{filepath}' ({len(content)} characters) into 8-bit KV Cache...")
                ingest_prompt = f"Please read and store this code for context:\nFile: {filepath}\n```python\n{content}\n```"
                formatted_ingest = tokenizer.apply_chat_template([{"role": "user", "content": ingest_prompt}], tokenize=False, add_generation_prompt=True)
                
                t0 = time.perf_counter()
                for resp in stream_generate(model, tokenizer, prompt=formatted_ingest, prompt_cache=kv_cache, max_tokens=10):
                    pass
                t1 = time.perf_counter()
                print(f"[✓] File ingested and cached! ({kv_cache[0].offset} tokens in KV state in {(t1 - t0):.2f}s).")
                continue

            # Standard chat turn
            formatted_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_input}],
                tokenize=False,
                add_generation_prompt=True
            )

            print(f"🤖 Codex > ", end="", flush=True)
            t0 = time.perf_counter()
            response_tokens = []
            
            for resp in stream_generate(model, tokenizer, prompt=formatted_prompt, prompt_cache=kv_cache, max_tokens=512):
                token_str = resp.text if hasattr(resp, "text") else str(resp)
                print(token_str, end="", flush=True)
                response_tokens.append(token_str)
            t1 = time.perf_counter()

            last_response_text = "".join(response_tokens)
            print(f"\n  [⏱ {(t1 - t0):.2f}s | 🎯 {kv_cache[0].offset} cached tokens]")

        except KeyboardInterrupt:
            print("\n[*] Interrupted. Exiting...")
            break
        except Exception as e:
            print(f"\n[!] Error during execution: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Codex Terminal CLI")
    parser.add_argument("--session", default="default_session", help="Session ID for persistent KV cache")
    parser.add_argument("--model", default="models/qwen2.5-coder-3b-mlx-4bit", help="Path to MLX model")
    parser.add_argument("--from", dest="base_session", default=None, help="Base session ID to branch from")
    args = parser.parse_args()

    run_interactive_codex(session_id=args.session, model_path=args.model, base_session=args.base_session)
