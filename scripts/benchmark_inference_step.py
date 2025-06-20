#!/usr/bin/env python
"""
Script to benchmark inference steps.

Accepts the same options as run_inference.py with the addition of benchmark-specific options in benchmark.yaml
"""
import hydra
from hydra.core.hydra_config import HydraConfig

from rfdiffusion.inference.benchmark_step import benchmark_inference_step


@hydra.main(version_base=None, config_path="../config/inference", config_name="benchmark")
def main(conf: HydraConfig):
    return benchmark_inference_step(conf)

if __name__ == "__main__":
    main()
