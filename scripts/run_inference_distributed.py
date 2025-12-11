#!/usr/bin/env python
"""
Inference script.

To run with base.yaml as the config,

> python run_inference.py

To specify a different config,

> python run_inference.py --config-name symmetry

where symmetry can be the filename of any other config (without .yaml extension)
See https://hydra.cc/docs/advanced/hydra-command-line-flags/ for more options.

"""
import logging
import os
import pathlib
import time

logging.captureWarnings(True)

import hydra
from hydra.core.hydra_config import HydraConfig
import torch
import torch.multiprocessing as mp

from rfdiffusion.inference.run import run_inference

LOG_FORMATTER = logging.Formatter("{asctime} {levelname}: {message}", style="{")

def get_logger(name, log_dir):
    pid = os.getpid()
    worker_name = f"worker.{name}.{pid}"
    log_path = log_dir / f"{worker_name}.log"
    logger = logging.getLogger()
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(LOG_FORMATTER)
    logger.addHandler(file_handler)
    return logger

def run_worker(rank: int, world_size: int, conf: HydraConfig, output_dir):
    logger = get_logger(rank, output_dir)
    torch.set_num_threads(os.cpu_count() // world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        logger.info(f"Using GPU {current_device} with device_name {device_name}.")
    else:
        raise RuntimeError("NO GPU DETECTED!")

    num_designs = conf.inference.num_designs
    designs_per_worker = num_designs // world_size
    extra_designs = num_designs % world_size

    if extra_designs != 0:
        logger.warning(
            f"Design count {num_designs} is not evenly divisible by world size {world_size}."
            " Some workers will do less work."
        )

    start_design = rank * designs_per_worker + min(rank, extra_designs)
    end_design = start_design + designs_per_worker + (1 if rank < extra_designs else 0)

    start_design = min(start_design, num_designs)
    end_design = min(end_design, num_designs)

    worker_num_designs = end_design - start_design

    if worker_num_designs == 0:
        logger.warning("More workers than designs. Nothing for this worker to do")
        return

    logger.info(f"Will create {worker_num_designs} designs")

    conf.inference.num_designs = worker_num_designs
    conf.inference.design_startnum = start_design

    start = time.perf_counter()
    run_inference(conf)
    elapsed = time.perf_counter() - start
    logger.info(f"Worker took {elapsed:.0f}s to create {worker_num_designs} designs")


@hydra.main(version_base=None, config_path="../config/inference", config_name="base")
def main(conf: HydraConfig):
    logger = logging.getLogger()
    if logger.handlers:
        for handler in logger.handlers:
            handler.setFormatter(LOG_FORMATTER)
    logger.info("Beginning distributed inference run")


    start = time.perf_counter()
    # TODO: figure out how to pass world size and just generally make hydra play nicely with multiprocessing
    world_size = 8
    output_dir = pathlib.Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger.info(f"Will save to {output_dir}")
    mp.spawn(run_worker, args=(world_size, conf, output_dir), nprocs=world_size, join=True)
    elapsed = time.perf_counter() - start
    logger.info(f"Whole job took {elapsed:.0f} seconds to create {conf.inference.num_designs} designs. {elapsed/conf.inference.num_designs:.0f}s per design")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
