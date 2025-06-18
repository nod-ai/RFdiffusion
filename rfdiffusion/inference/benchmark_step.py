import time
import torch
import logging
from rfdiffusion.inference import utils as iu
from hydra.core.hydra_config import HydraConfig
import statistics


def benchmark_inference_step(conf: HydraConfig) -> None:
    """Benchmarks individual steps of inference.

    This seeks to evaluate the steady-state performance of the key inference
    workload while avoiding running all the steps. This will not capture the
    contributions of overhead from initialization but should still give a good
    idea of overall perforamce on large workloads without having to run the
    whole thing. It expects a config as defined in
    config/inference/benchmark.yaml
    """
    log = logging.getLogger(__name__)

    bench_conf = conf.benchmark

    if conf.inference.deterministic:
        conf.inference.random_seed = conf.inference.random_seed or 0
        iu.make_deterministic(conf.inference.random_seed)

    iu.find_gpu(required=conf.inference.require_gpu)

    # Initialize sampler and target/contig.
    sampler = iu.sampler_selector(conf)
    total_steps = bench_conf.warmup_steps + bench_conf.benchmark_steps
    assert (
        sampler.t_step_input >= total_steps
    ), f"Sampler total steps {sampler.t_step_input} is less than requested total steps {total_steps}"

    # Loop over number of designs to sample.
    design_startnum = sampler.inf_conf.design_startnum
    total_designs = sampler.inf_conf.num_designs + bench_conf.warmup_designs

    times = []
    for i_des in range(design_startnum, design_startnum + total_designs):
        if conf.inference.random_seed is not None:
            iu.seed_rngs(conf.inference.random_seed + i_des)

        log.info(f"Making design {i_des}")

        x_init, seq_init = sampler.sample_init()

        x_t = torch.clone(x_init)
        seq_t = torch.clone(seq_init)
        t = int(sampler.t_step_input)
        for _ in range(bench_conf.warmup_steps):
            _, x_t, seq_t, _ = sampler.sample_step(
                t=t, x_t=x_t, seq_init=seq_t, final_step=sampler.inf_conf.final_step
            )
            t -= 1

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(bench_conf.benchmark_steps):
            _, x_t, seq_t, _ = sampler.sample_step(
                t=t, x_t=x_t, seq_init=seq_t, final_step=sampler.inf_conf.final_step
            )
            t -= 1
        torch.cuda.synchronize()
        elapsed_time = time.perf_counter() - start_time
        time_per_step = elapsed_time / bench_conf.benchmark_steps

        times.append(time_per_step)

    times = times[bench_conf.warmup_designs :]
    times = sorted(times)
    median_time_per_step = statistics.median(times)
    print(
        f"Median time per step: {median_time_per_step*1000:.0f}ms ({times[0]*1000:.0f}ms-{times[-1]*1000:.0f}ms): ",
        ",".join([f"{t*1000:.0f}" for t in times]),
    )
