#!/usr/bin/env python3
"""Run microbenchmarks from test configs across multiple GPUs at once.

This distributes across GPUs to speed up the time it takes to collect
benchmarks, but it isn't intended to benchmark distributed inference. The
results will be noisier due to conflicts between the processes, but faster to
collect.
"""


import argparse
from collections import defaultdict
import csv
import logging
import os
import pathlib
import statistics
import sys
import torch

import hydra
from omegaconf import OmegaConf
import numpy as np
import torch.multiprocessing as mp
import yaml

from rfdiffusion.inference import utils as iu
from rfdiffusion.inference.benchmark_step import benchmark_inference_step

NUM_DESIGNS = 5

def get_logger(name, log_dir):
    pid = os.getpid()
    worker_name = f"worker.{name}.{pid}"
    log_path = log_dir / f"{worker_name}.log"
    logger = logging.getLogger()
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("{asctime} {levelname}: {message}", style="{")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


script_dir = pathlib.Path(__file__).parent
example_dir = script_dir.parent / "examples"


def benchmark_config(test_conf, compile=False, compilation_mode=None):
    os.chdir(example_dir)

    with hydra.initialize(config_path="../config/inference", version_base="1.2"):
        benchmark_conf = hydra.compose(config_name="benchmark")
        overrides = {
            "inference": {
                "num_designs": NUM_DESIGNS,
                "compile": compile,
                "compilation_mode": compilation_mode,
            }
        }

        conf = OmegaConf.merge(benchmark_conf, test_conf, overrides)

        return benchmark_inference_step(conf)


def log_times(name, times, logger):
    median_time_per_step = statistics.median(times)
    logger.info(
        f"{name} median time per step: {median_time_per_step*1000:.0f}ms ({times[0]*1000:.0f}ms-{times[-1]*1000:.0f}ms): "
        + f",".join([f"{t*1000:.0f}" for t in times]),
    )


def benchmark_configs(configs, csv_path, logger, compile=False, compilation_mode=None):
    failures = []
    times_by_conf = {}
    with open(csv_path, mode="a", newline="") as csv_file:
        csv_writer = csv.writer(csv_file, dialect="unix")
        for name, config in configs.items():
            try:
                logger.info(f"Benchmarking {name}")
                times = benchmark_config(config, compile=compile, compilation_mode=compilation_mode)
                log_times(name, times, logger)
                csv_writer.writerow([name] + times)
                csv_writer.flush()
                times_by_conf[name] = times

            except Exception as e:
                failures.append(name)
                logger.error(f"Failed running: {name}")
                logger.exception(e)

    return times_by_conf, failures


def benchmark_configs_by_rank(rank, ranks_to_configs, job_dir, q, compile=False, compilation_mode=None):
    log_dir = job_dir / "logs"
    logger = get_logger(rank, log_dir)
    configs = ranks_to_configs[rank]
    logger.info(f"Processing {len(configs)} configs")
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        logger.info(f"Using GPU {current_device} with device_name {device_name}.")
    else:
        raise RuntimeError("NO GPU DETECTED!")

    result_dir = job_dir / "results"
    result_dir.mkdir(exist_ok=True)
    csv_path = job_dir / "results" / f"rank{rank}.csv"
    times_by_conf, failures = benchmark_configs(configs, csv_path, logger, compile=compile, compilation_mode=compilation_mode)
    q.put((times_by_conf, failures, csv_path))


def main(world_size, job_name, output_dir, compile=False, compilation_mode=None):
    job_dir = output_dir / job_name
    job_dir.mkdir(parents=True)
    csv_path = job_dir / "results.csv"
    log_dir = job_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("main", log_dir)

    times_by_conf = {}
    if csv_path.exists():
        with open(csv_path, newline="") as csv_file:
            reader = csv.DictReader(
                csv_file, dialect="unix", fieldnames=["name"], restkey="times"
            )
            for row in reader:
                times_by_conf[row["name"]] = [float(t) for t in row["times"]]

    test_configs = yaml.safe_load((script_dir / "configs.gen.yaml").read_text())

    configs = dict(**test_configs["small"], **test_configs["large"])

    for name in times_by_conf.keys():
        configs.pop(name, None)

    configs_by_type = defaultdict(dict)
    for name, config in configs.items():
        design_type, _, length = name.rpartition("_")
        if len(length) != 4:
            raise ValueError(f"Expected length to have 4 digits but got {len(length)}: '{length}'")

        # check it parses as an int also
        length = int(length.lstrip("0"))

        configs_by_type[design_type][name] = config

    ranks_to_configs = defaultdict(dict)
    for i, (design_type, configs) in enumerate(configs_by_type.items()):
        rank = i % world_size
        logger.info(f"Assigning {len(configs)} {design_type} configs to rank {rank}")
        ranks_to_configs[i % world_size].update(configs)

    q = mp.Queue()
    mp.spawn(
        benchmark_configs_by_rank,
        args=(ranks_to_configs, job_dir, q, compile, compilation_mode),
        nprocs=world_size,
        join=True,
    )

    failures = []
    csv_paths = []
    while not q.empty():
        times_by_conf_rank, failures_rank, csv_path_rank = q.get()
        times_by_conf.update(times_by_conf_rank)
        failures.extend(failures_rank)
        csv_paths.append(csv_path_rank)

    for name, times in times_by_conf.items():
        log_times(name, times, logger)

    with open(csv_path, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file, dialect="unix")
        for csv_path_rank in csv_paths:
            with open(csv_path_rank, newline="") as csv_file_rank:
                reader = csv.reader(csv_file_rank, dialect="unix")
                for row in reader:
                    csv_writer.writerow(row)

    print(f"Results saved to {csv_path}")

    if failures:
        logger.error(f"{len(failures)} benchmarks failed: {failures}", file=sys.stderr)
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("job_name")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compilation_mode")
    args = parser.parse_args()

    if args.compilation_mode and not args.compile:
        raise ValueError(f"Cannot set compilation mode '{args.compilation_mode}' without compilation")

    args.output_dir = pathlib.Path(args.output_dir)

    return args


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main(**vars(parse_args()))
