"""
Latency & Throughput Benchmarker for Apple Silicon.
Tests TTFT, Prefill Tok/s, Decode Tok/s, TPOT, and Peak RAM across context lengths:
128, 512, 2048, 4096, 8192 tokens.
"""

import time
import mlx.core as mx
from typing import Dict, Any, List
from benchmark.memory_tracker import UnifiedMemoryTracker

def generate_context_text(target_tokens: int) -> str:
    """Generates a realistic repetitive code block to achieve exact token counts."""
    base_snippet = """
def calculate_batch_statistics(records: list, threshold: float = 0.5) -> dict:
    valid_data = [x for x in records if x is not None and x > threshold]
    count = len(valid_data)
    total = sum(valid_data) if count > 0 else 0.0
    mean = total / count if count > 0 else 0.0
    variance = sum((x - mean) ** 2 for x in valid_data) / count if count > 0 else 0.0
    return {"count": count, "total": total, "mean": mean, "variance": variance}
"""
    # approx 70 tokens per repetition
    repeat_count = max(1, target_tokens // 70)
    return base_snippet * repeat_count

def benchmark_context_lengths(
    model: Any,
    tokenizer: Any,
    context_lengths: List[int] = [128, 512, 2048, 4096, 8192],
    max_decode_tokens: int = 64,
    prompt_cache: Any = None
) -> List[Dict[str, Any]]:
    """Runs latency and throughput measurements across multiple context lengths."""
    mem_tracker = UnifiedMemoryTracker()
    results = []

    for target_ctx in context_lengths:
        mem_tracker.clear_cache()
        mem_tracker.reset_peak()

        raw_context = generate_context_text(target_ctx)
        prompt = f"System: Analyze this codebase context.\n{raw_context}\n\nUser: Summarize the main function."
        
        prompt_tokens = tokenizer.encode(prompt)
        actual_ctx_len = len(prompt_tokens)
        
        # If actual tokens is larger than model max window, clamp or skip
        if actual_ctx_len > 8192 and target_ctx > 8192:
            continue

        # Measure Prefill / TTFT
        prompt_arr = mx.array(prompt_tokens)[None]
        
        t0 = time.perf_counter()
        # Prefill forward pass
        logits = model(prompt_arr)
        mx.eval(logits)
        t_prefill = time.perf_counter() - t0
        
        ttft_ms = round(t_prefill * 1000, 2)
        prefill_tps = round(actual_ctx_len / max(t_prefill, 1e-6), 2)

        # Measure Generation / Decode
        last_token = mx.argmax(logits[:, -1, :], axis=-1)[:, None]
        decode_tokens = [last_token.item()]
        
        t_gen_start = time.perf_counter()
        curr_tok = last_token
        
        for _ in range(max_decode_tokens - 1):
            curr_logits = model(curr_tok)
            curr_tok = mx.argmax(curr_logits[:, -1, :], axis=-1)[:, None]
            mx.eval(curr_tok)
            decode_tokens.append(curr_tok.item())

        t_decode_total = time.perf_counter() - t_gen_start
        decode_tps = round(len(decode_tokens) / max(t_decode_total, 1e-6), 2)
        tpot_ms = round((t_decode_total / max(len(decode_tokens), 1)) * 1000, 2)
        total_latency_ms = round((t_prefill + t_decode_total) * 1000, 2)

        mem_snap = mem_tracker.get_snapshot()

        results.append({
            "target_context": target_ctx,
            "actual_prompt_tokens": actual_ctx_len,
            "generated_tokens": len(decode_tokens),
            "ttft_ms": ttft_ms,
            "prefill_tok_per_sec": prefill_tps,
            "decode_tok_per_sec": decode_tps,
            "tpot_ms": tpot_ms,
            "total_latency_ms": total_latency_ms,
            "peak_metal_mb": mem_snap["metal_peak_mb"],
            "active_metal_mb": mem_snap["metal_active_mb"],
            "process_rss_mb": mem_snap["process_rss_mb"]
        })

    return results
