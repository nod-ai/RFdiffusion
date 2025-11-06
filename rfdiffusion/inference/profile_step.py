import contextlib
import logging
import os

import hydra
from hydra.core.hydra_config import HydraConfig
import torch

from rfdiffusion.inference import utils as iu


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

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir


    for i_des in range(design_startnum, design_startnum + total_designs):
        if conf.inference.random_seed is not None:
            iu.seed_rngs(conf.inference.random_seed + i_des)


        log.info(f"Making design {i_des}")

        x_init, seq_init = sampler.sample_init()

        x_t = torch.clone(x_init)
        seq_t = torch.clone(seq_init)
        start_step = int(sampler.t_step_input)

        cm = contextlib.nullcontext()
        if i_des < design_startnum + bench_conf.warmup_designs:
            print("Skipping profiling on this warmup design")
        else:
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
            # torch.cuda.memory._record_memory_history()

        with cm as prof:
            for t in range(start_step, start_step - total_steps - 1, -1):
                _, x_t, seq_t, _ = sampler.sample_step(
                    t=t, x_t=x_t, seq_init=seq_t, final_step=sampler.inf_conf.final_step
                )
                if prof is not None:
                    prof.step()

        if prof is not None:
            trace_path = os.path.join(output_dir, f"trace_{i_des}.json")
            # mem_dump_path = os.path.join(output_dir, f"memdump_{i_des}.pkl")
            print(f"Exporting trace to {trace_path} ...", end="", flush=True)
            prof.export_chrome_trace(trace_path)
            print(f"done")
            # print(f"Exporting memory dump to {mem_dump_path} ...", end="", flush=True)
            # torch.cuda.memory._dump_snapshot(mem_dump_path)
            # print("done")

