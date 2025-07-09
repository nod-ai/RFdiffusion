#!/usr/bin/env python
"""
Script to benchmark inference steps.

Accepts the same options as run_inference.py with the addition of benchmark-specific options in benchmark.yaml
"""
import statistics

import hydra
from hydra.core.hydra_config import HydraConfig

from rfdiffusion.inference.benchmark_step import benchmark_inference_step


@hydra.main(version_base=None, config_path="../config/inference", config_name="benchmark")
def main(conf: HydraConfig):
    times = benchmark_inference_step(conf)
    median_time_per_step = statistics.median(times)
    print(
        f"Median time per step: {median_time_per_step*1000:.0f}ms ({times[0]*1000:.0f}ms-{times[-1]*1000:.0f}ms): ",
        ",".join([f"{t*1000:.0f}" for t in times]),
    )


if __name__ == "__main__":
    main()
