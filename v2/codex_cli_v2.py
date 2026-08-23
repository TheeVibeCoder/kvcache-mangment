#!/usr/bin/env python3
"""
====================================================================================================
CODEX CLI V2 - Turn-by-Turn Disaggregated KV Agent with Incremental Delta Appending
====================================================================================================
• Protocol per Turn:
  1. JIT Onload: Concatenates lightweight chunk files into QuantizedKVCache in ~7-15ms.
  2. Stream Decode: Generates response with zero prefill overhead.
  3. Delta Offload: Slices & writes *only* the new turn delta [offset_before:offset_after] in ~2ms.
  4. Instant RAM Purge: Releases Metal memory back to baseline (0.0 MB KV cache held in RAM).
====================================================================================================
"""

import os
import sys
import argparse

# Ensure local v2 imports resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kv_cache_engine_v2 import TurnBasedKVCacheEngine

def main():
    parser = argparse.ArgumentParser(description="Codex CLI V2 - Incremental Delta KV Agent")
    parser.add_argument("--session", default="default_v2_session", help="Session identifier")
    parser.add_argument("--model", default="models/qwen2.5-coder-3b-mlx-4bit", help="Model path")
    parser.add_argument("--bits", type=int, default=8, choices=[4, 8, 16], help="KV Cache precision")
    args = parser.parse_args()

    print("=" * 85)
    print("  ⚡ CODEX CLI V2 (Turn-by-Turn Zero-Idle-RAM Incremental Delta Engine)")
    print("=" * 85)
    print(f"[*] Initializing model: '{args.model}' | KV Precision: {args.bits}-bit")
    
    engine = TurnBasedKVCacheEngine(model_path=args.model, kv_bits=args.bits)
    session_id = args.session

    print(f"[✓] Engine ready. Active Session: '{session_id}'")
    print(f"[*] Telemetry Log: `v2_turn_metrics.jsonl`")
    print("-" * 85)
    print("Commands:")
    print("  /sessions              : List all saved session caches on SSD")
    print("  /load <session_id>     : Switch active session")
    print("  /read <filepath>       : Ingest a code file into the session KV cache")
    print("  /exit                  : Quit")
    print("-" * 85)

    while True:
        try:
            user_input = input(f"\n[{session_id}] 👤 You > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("/exit", "/quit"):
                print("[*] Goodbye!")
                break

            elif user_input == "/sessions":
                sessions = engine.store.list_sessions()
                print(f"[*] Available session caches on SSD ({len(sessions)}):")
                for s in sessions:
                    marker = " (Active)" if s["session_id"] == session_id else ""
                    print(f"   • {s['session_id']}{marker} | {s['total_tokens']} tokens | {s['chunk_count']} chunks ({s['total_size_kb']:.1f} KB)")
                continue

            elif user_input.startswith("/load "):
                target_session = user_input.split(maxsplit=1)[1].strip()
                if engine.store.exists(target_session):
                    session_id = target_session
                    print(f"[✓] Switched active session to: '{session_id}'.")
                else:
                    print(f"[!] Session '{target_session}' not found on SSD.")
                continue

            elif user_input.startswith("/read "):
                filepath = user_input.split(maxsplit=1)[1].strip()
                if not os.path.exists(filepath):
                    print(f"[!] File not found: '{filepath}'")
                    continue
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                print(f"[*] Ingesting '{filepath}' ({len(content)} chars) into session '{session_id}'...")
                ingest_prompt = f"Please read and store this code for context:\nFile: {filepath}\n```python\n{content}\n```"
                _, meta = engine.ask(ingest_prompt, session_id=session_id, max_tokens=10)
                lat = meta.get("latency", {})
                tok = meta.get("tokens", {})
                stg = meta.get("storage", {})
                print(f"[✓] Ingested! Context: {tok.get('total_context')} tokens | Chunk: {stg.get('delta_chunk_file')} ({stg.get('delta_chunk_kb')} KB in {lat.get('delta_write_ms')}ms) | Idle RAM: 0MB")
                continue

            # Standard Turn Query with Streaming & Delta HUD
            print(f"🤖 Codex > ", end="", flush=True)
            meta = {}
            for token, step_meta in engine.ask_stream(user_input, session_id=session_id, max_tokens=512):
                if token:
                    print(token, end="", flush=True)
                if step_meta.get("status") == "complete":
                    meta = step_meta

            lat = meta.get("latency", {})
            tok = meta.get("tokens", {})
            mem = meta.get("memory_mb", {})
            stg = meta.get("storage", {})

            is_reused = tok.get("is_reused", False)
            reused_cnt = tok.get("reused_from_cache", 0)
            if is_reused:
                reuse_badge = f"♻️  CACHE REUSED: YES (Restored {reused_cnt} past tokens in {lat.get('onload_ms')} ms — 0 prefill recomputation!)"
            else:
                reuse_badge = f"🆕 CACHE: FRESH START (0 past tokens)"

            print("\n" + "─" * 85)
            print(f"  {reuse_badge}")
            print(f"  ⚡ Onload: {lat.get('onload_ms')} ms  │  ⏱ TTFT: {lat.get('ttft_ms')} ms  │  🚀 Speed: {lat.get('tokens_per_sec')} tok/s  │  💾 Delta Write: {lat.get('delta_write_ms')} ms")
            print(f"  🎯 Context: {tok.get('total_context')} tokens (+{tok.get('delta_tokens_added')} new)  │  📦 Delta Chunk: {stg.get('delta_chunk_kb')} KB (Total: {stg.get('total_session_kb')} KB)  │  🟢 Idle RAM: 0 MB")
            print("─" * 85)

        except KeyboardInterrupt:
            print("\n[*] Interrupted. Exiting...")
            break
        except Exception as e:
            print(f"\n[!] Error during turn: {e}")

if __name__ == "__main__":
    main()
