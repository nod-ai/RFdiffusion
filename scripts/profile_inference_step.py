#!/usr/bin/env python
"""
Script to profile inference steps.

Accepts the same options as run_inference.py with the addition of
benchmark-specific options in benchmark.yaml and the profile-specific options in
profile.yaml.
"""
import hydra
from hydra.core.hydra_config import HydraConfig

from rfdiffusion.inference.profile_step import profile_inference_step


@hydra.main(version_base=None, config_path="../config/inference", config_name="profile")
def main(conf: HydraConfig):
    return profile_inference_step(conf)

if __name__ == "__main__":
    main()
