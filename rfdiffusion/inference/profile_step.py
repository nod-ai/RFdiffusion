import contextlib
import torch
import logging
from rfdiffusion.inference import utils as iu
from hydra.core.hydra_config import HydraConfig
import os


def profile_inference_step(conf: HydraConfig) -> None:
    """Profiles individual steps of inference.

    This seeks to evaluate the steady-state performance of the key inference
    workload while avoiding running all the steps. This will not capture the
    contributions of overhead from initialization but should still give a good
    idea of overall perforamce on large workloads without having to run the
    whole thing. It expects a config as defined in
    config/inference/profile.yaml
    """
    log = logging.getLogger(__name__)

    bench_conf = conf.benchmark

    if conf.inference.deterministic:
        conf.inference.random_seed = conf.inference.random_seed or 0
        iu.make_deterministic(conf.inference.random_seed)

    iu.find_gpu(required=conf.inference.require_gpu)

    # Initialize sampler and target/contig.
    sampler = iu.sampler_selector(conf)
    total_steps = (
        conf.profile.wait_steps + bench_conf.warmup_steps + bench_conf.benchmark_steps
    )
    sampler_steps = sampler.t_step_input - sampler.inf_conf.final_step + 1
    assert (
        sampler_steps >= total_steps
    ), f"Sampler total steps {sampler_steps} is less than requested total steps {total_steps}"

    # Loop over number of designs to sample.
    design_startnum = sampler.inf_conf.design_startnum
    total_designs = sampler.inf_conf.num_designs + bench_conf.warmup_designs

    for i_des in range(design_startnum, design_startnum + total_designs):
        if conf.inference.random_seed is not None:
            iu.seed_rngs(conf.inference.random_seed + i_des)

        trace_path = f"{sampler.inf_conf.output_prefix}_{i_des}.json"
        log.info(f"Making design {i_des}")
        if sampler.inf_conf.cautious and os.path.exists(trace_path):
            log.warning(
                f"(cautious mode) Skipping this design because {trace_path} already exists."
            )
            continue

        x_init, seq_init = sampler.sample_init()

        x_t = torch.clone(x_init)
        seq_t = torch.clone(seq_init)
        start_step = int(sampler.t_step_input)

        cm = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            schedule=torch.profiler.schedule(
                wait=conf.profile.wait_steps,
                warmup=bench_conf.warmup_steps,
                active=bench_conf.benchmark_steps,
                repeat=1,
            ),
        )
        if i_des < design_startnum + bench_conf.warmup_designs:
            print("Skipping profiling on this warmup design")
            cm = contextlib.nullcontext()

        with cm as prof:
            for t in range(start_step, start_step - total_steps - 1, -1):
                _, x_t, seq_t, _ = sampler.sample_step(
                    t=t, x_t=x_t, seq_init=seq_t, final_step=sampler.inf_conf.final_step
                )
                if prof is not None:
                    prof.step()

        if prof is not None:
            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
            print(f"Exporting trace to {trace_path}...", end="")
            prof.export_chrome_trace(trace_path)
            print(f"done")
