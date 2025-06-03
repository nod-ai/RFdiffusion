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
import hydra
from hydra.core.hydra_config import HydraConfig

from rfdiffusion.inference.run import run_inference


@hydra.main(version_base=None, config_path="../config/inference", config_name="base")
def main(conf: HydraConfig):
    return run_inference(conf)

if __name__ == "__main__":
    main()
