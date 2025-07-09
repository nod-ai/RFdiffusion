#!/usr/bin/env python3

import itertools
import os
import pathlib
import logging
import statistics
import sys

import hydra
from omegaconf import OmegaConf
import numpy as np
import yaml

from rfdiffusion.inference import utils as iu
from rfdiffusion.inference.benchmark_step import benchmark_inference_step

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

script_dir = pathlib.Path(__file__).parent
example_dir = script_dir.parent / "examples"

def benchmark_config(test_conf):
    os.chdir(example_dir)

    with hydra.initialize(config_path="../config/inference", version_base="1.2"):
        benchmark_conf = hydra.compose(config_name="benchmark")
        overrides = {
            "inference": {
                "num_designs": 5,
            }
        }

        conf = OmegaConf.merge(benchmark_conf, test_conf, overrides)

        return benchmark_inference_step(conf)


def log_times(name, times):
    median_time_per_step = statistics.median(times)
    print(
        f"{name} median time per step: {median_time_per_step*1000:.0f}ms ({times[0]*1000:.0f}ms-{times[-1]*1000:.0f}ms): ",
        ",".join([f"{t*1000:.0f}" for t in times]),
    )

def main():
    test_configs = yaml.safe_load((script_dir / "configs.gen.yaml").read_text())

    failures = []
    times_by_conf = {}
    for name, config in itertools.chain(test_configs["small"].items(), test_configs["large"].items()):
        try:
            logger.info(f"Benchmarking {name}")
            times = benchmark_config(config)
            log_times(name, times)
            times_by_conf[name] = times

        except Exception as e:
            failures.append(name)
            logger.error(f"Failed running: {name}")
            logger.exception(e)

    for name, times, in times_by_conf.items():
        log_times(name, times)

    if failures:
        print(f"{len(failures)} benchmarks failed: {failures}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
