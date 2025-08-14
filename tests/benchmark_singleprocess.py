#!/usr/bin/env python3

import argparse
import csv
import logging
import os
import pathlib
import statistics
import sys

import hydra
from omegaconf import OmegaConf
import yaml

from rfdiffusion.inference import utils as iu
from rfdiffusion.inference.benchmark_step import benchmark_inference_step

NUM_DESIGNS = 5

script_dir = pathlib.Path(__file__).parent
example_dir = script_dir.parent / "examples"

def get_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("{asctime} {levelname}: {message}", style="{")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

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
                csv_file.flush()
                times_by_conf[name] = times

            except Exception as e:
                failures.append(name)
                logger.error(f"Failed running: {name}")
                logger.exception(e)

    return times_by_conf, failures


def main(job_name, output_dir, compile=False, compilation_mode=None, resume=False):
    job_dir = output_dir / job_name
    job_dir.mkdir(parents=True, exist_ok=resume)
    csv_path = job_dir / "results.csv"
    log_path = job_dir / "output.log"
    logger = get_logger(log_path)

    test_configs = yaml.safe_load((script_dir / "configs.gen.yaml").read_text())
    configs = dict(**test_configs["small"], **test_configs["large"])

    times_by_conf = {}
    if resume:
        logger.info("Resuming previous job")
        if csv_path.exists():
            logger.info(f"Reading existing results from {csv_path}")
            with open(csv_path, newline="") as csv_file:
                reader = csv.DictReader(
                    csv_file, dialect="unix", fieldnames=["name"], restkey="times"
                )
                for row in reader:
                    times_by_conf[row["name"]] = [float(t) for t in row["times"]]
            logger.info(f"Found {len(times_by_conf)} existing results")

            for name in times_by_conf.keys():
                if configs.pop(name, None):
                    logger.info(f"Skipping config {name} with existing results")
                else:
                    logger.warning(f"Found existing result for {name} which is not in configs")
        else:
            logger.warning(f"Found no existing results csv at {csv_path}")

    times_by_conf, failures = benchmark_configs(configs, csv_path, logger, compile=compile, compilation_mode=compilation_mode)

    if failures:
        logger.error(f"{len(failures)} benchmarks failed: {failures}", file=sys.stderr)
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("job_name")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compilation_mode")
    args = parser.parse_args()

    if args.compilation_mode and not args.compile:
        raise ValueError(f"Cannot set compilation mode '{args.compilation_mode}' without compilation")

    args.output_dir = pathlib.Path(args.output_dir)

    return args


if __name__ == "__main__":
    main(**vars(parse_args()))
