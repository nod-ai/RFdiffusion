import os
import pathlib
import shutil
import subprocess

from Bio.PDB import PDBParser, Superimposer
import hydra
from omegaconf import OmegaConf
import numpy as np
import pytest
import torch
import yaml

from rfdiffusion.inference import utils as iu
from rfdiffusion.inference.run import run_inference

script_dir = pathlib.Path(__file__).parent
example_dir = script_dir.parent / "examples"

# We have two different ways of splitting up GPUs depending on whether the test
# is launching a subprocess or running in-process. I tried just setting the
# torch device number and environment variables together, but this had two problems:
#   - setting the current device to anything other than 0 initalizes cuda, which
#     takes a long time. This is pretty wasteful when the process is just going
#     to launch another script.
#   - Torch does not behave well if you change CUDA_VISIBLE_DEVICES after it's
#     imported. Theoretically it's supposed to handle this, but it actually
#     won't change the current device. It just uses the environment variable to
#     calculate a new device_count, which means any subsequent attempt to change
#     the device fails.


def partition_gpus(idx, env, env_var):
    devs = env.get(env_var)
    if devs:
        devs = [int(d) for d in devs.split(",")]
        dev = devs[idx]
        print(f"Running with {env_var}={dev}")
        env[env_var] = str(dev)


@pytest.fixture(scope="session")
def child_env(worker_idx):
    env = os.environ.copy()

    partition_gpus(worker_idx, env, "CUDA_VISIBLE_DEVICES")
    partition_gpus(worker_idx, env, "HIP_VISIBLE_DEVICES")

    return env


@pytest.fixture(scope="session")
def set_torch_device(worker_idx):
    if worker_idx == 0:
        # these checks are slow. We can skip them in the simple case.
        return
    device_count = torch.cuda.device_count()
    torch.cuda.set_device(worker_idx % device_count)


@pytest.fixture(scope="module")
def reference_dir():
    d = script_dir / "reference_outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="module")
def symlink_inputs():
    # Make sure we have access to all the relevant files
    exclude_dirs = ["outputs", "example_outputs"]
    for p in example_dir.iterdir():
        link = script_dir / p.name
        if p.name not in exclude_dirs and p.is_dir() and not link.is_symlink():
            print(f"Symlinking {link} -> {p}")
            link.symlink_to(p)


def calc_atom_rmsd(ref_atoms, test_atoms):
    # First do this the simple way and just take the RMSD between the
    # coordinates directly. BioPython tries to find an RMSD-minimizing rotation
    # and translation, but that introduces some slight numerical errors. That
    # matters especially (only?) when the coordinates are identical to start
    # because it results in an RMSD > 0. It's nice to know when things literally
    # haven't changed.
    ref_coords = np.array([a.get_coord() for a in ref_atoms])
    test_coords = np.array([a.get_coord() for a in test_atoms])

    direct_rmsd = np.sqrt(((ref_coords - test_coords) ** 2).sum(-1).mean()).item()

    # The superimposed version can't possibly be smaller, so don't bother
    # computing it.
    if direct_rmsd == 0.0:
        return direct_rmsd

    sup = Superimposer()
    sup.set_atoms(ref_atoms, test_atoms)

    return min(direct_rmsd, sup.rms.item())


def get_backbone(pdb):
    # QUIET because RFdiffusion writes PDBs that are missing the element and Biopython complains.
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb", pdb)
    backbone = [a for a in structure.get_atoms() if a.get_id() in ["CA", "C", "N"]]
    return backbone


def calc_backbone_rmsd(ref_pdb, test_pdb):
    return calc_atom_rmsd(get_backbone(ref_pdb), get_backbone(test_pdb))


def handle_test_output(test_name, reference_dir, output_dir, request):
    test_file = output_dir / f"{test_name}_0.pdb"
    reference_file = reference_dir / test_file.relative_to(output_dir)

    # We store the reference path file or the RMSD value on the test node for
    # later reporting using user_properties. See `pytest_terminal_summary` in
    # conftest.py for how these are used. I'm not sure whether this it the best
    # way to do this. I didn't find a clear blessed way to have tests report
    # additional non-status outputs. Anything in these properties needs to be
    # serializable by execnet in order to use pytest-xdist (which basically
    # means only Python primitives).

    if request.config.getoption("--update-goldens"):
        reference_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(test_file, reference_file)
        # Anything in these properties needs to be serializable by execnet in
        # order to use pytest-xdist, hence converting to a string.
        request.node.user_properties.append(("reference_file", str(reference_file)))
        print(f"Updated reference file {reference_file}")
    else:
        rmsd = calc_backbone_rmsd(reference_file, test_file)
        request.node.user_properties.append(("rmsd", rmsd))
        print(f"RMSD={rmsd:.3}")

        assert rmsd == pytest.approx(0, rel=0, abs=0.01)


@pytest.mark.usefixtures("symlink_inputs")
@pytest.mark.parametrize(
    "script", sorted(example_dir.glob("*.sh")), ids=lambda x: x.stem
)
def test_command(script, tmp_path, reference_dir, child_env, request):
    # The pytest docs say you need to create this directory, but empirically it
    # is already created.
    output_dir = tmp_path
    modified_script = _write_command(script, output_dir)
    print(f"Running {modified_script}")
    # cwd is required because the scripts use relative paths like `../scripts/run_inference.py`
    subprocess.run(["bash", modified_script], check=True, cwd=script_dir, env=child_env)
    handle_test_output(modified_script.stem, reference_dir, output_dir, request)


def _write_command(bash_file, output_dir):
    """
    Takes a bash file from the examples folder, and writes
    a version of it to the output_dir folder.
    It appends to the python command the following arguments:
        inference.deterministic=True
        if partial_T is in the command, it grabs partial T and sets:
            inference.final_step=partial_T-2
        else:
            inference.final_step=48
    """
    out_lines = []
    with open(bash_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("python") or line.startswith("../"):
                command = line.rstrip()
                if "partial_T" in command:
                    final_step = int(command.split("partial_T=")[1].split(" ")[0]) - 2
                else:
                    final_step = 48

                test_name = bash_file.stem

                # Override these. It's ok if they're already specified, as last one wins
                command = (
                    f"{command}"
                    f" inference.num_designs=1"
                    f" inference.output_prefix={output_dir}/{test_name}"
                    f" inference.deterministic=True"
                    f" inference.final_step={final_step}"
                )

                out_lines.append(command)
            else:
                out_lines.append(line)

    output_script = output_dir / bash_file.name
    output_script.write_text("".join(out_lines))

    return output_script


def flatten_nested_dict(d, parent_key="", sep="."):
    """Flatten a nested dictionary using dot notation for keys."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def config_id(config):
    if name := config.pop("name", None):
        return name
    flattened = flatten_nested_dict(config)

    if "inference.input_pdb" in flattened:
        flattened["inference.input_pdb"] = pathlib.Path(
            flattened["inference.input_pdb"]
        ).stem

    fields = []
    for v in flattened.values():
        if isinstance(v, list):
            fields.append("_".join(v))
        else:
            fields.append(str(v))

    return "_".join(fields)


@pytest.mark.usefixtures("set_torch_device", "symlink_inputs")
@pytest.mark.parametrize(
    "spec",
    yaml.safe_load((script_dir / "configs.yaml").read_text())["tests"],
    ids=config_id,
)
def test_config(spec, tmp_path, reference_dir, request):
    os.chdir(script_dir)

    with hydra.initialize(config_path="../config/inference"):
        conf = hydra.compose(config_name="base")
        # hydra.compose has an overrides argument but it only accepts
        # dot-notation string syntax like
        # "configmap.contigs=[A151-180/70-70/A251-300]" for some reason. It
        # seems annoying and error-prone to convert our already structured input
        # into that format (even though I already wrote a flatten function to
        # construct test ids: that is far more appropriate when a string is the
        # end goal)
        output_dir = tmp_path
        test_name = request.node.callspec.id.replace(" ", "_").replace("/", "_")
        overrides = {
            "inference": {
                "num_designs": 1,
                "output_prefix": output_dir / test_name,
                "deterministic": True,
            }
        }
        conf = OmegaConf.merge(conf, spec, overrides)

        run_inference(conf)
        handle_test_output(test_name, reference_dir, output_dir, request)

if __name__ == "__main__":
    pytest.main()
