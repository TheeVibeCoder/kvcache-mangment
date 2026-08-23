"""
Apple Silicon Unified Memory Profiler.
Accurately tracks Process RSS, MLX Active Metal Memory, MLX Peak Memory, and Cache Memory.
"""

import os
import psutil
import mlx.core as mx

class UnifiedMemoryTracker:
    def __init__(self):
        self.process = psutil.Process(os.getpid())

    def get_snapshot(self) -> dict:
        """Returns instantaneous snapshot of Apple Silicon unified memory."""
        rss_bytes = self.process.memory_info().rss
        active_bytes = mx.metal.get_active_memory()
        peak_bytes = mx.metal.get_peak_memory()
        cache_bytes = getattr(mx.metal, "get_cache_memory", lambda: 0)()

        return {
            "process_rss_mb": round(rss_bytes / (1024 * 1024), 2),
            "metal_active_mb": round(active_bytes / (1024 * 1024), 2),
            "metal_peak_mb": round(peak_bytes / (1024 * 1024), 2),
            "metal_cache_mb": round(cache_bytes / (1024 * 1024), 2),
        }

    def reset_peak(self):
        """Resets the peak memory tracker in MLX."""
        mx.metal.reset_peak_memory()

    def clear_cache(self):
        """Flushes unused Metal allocations."""
        mx.clear_cache()
