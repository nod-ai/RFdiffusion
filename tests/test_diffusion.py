import os
import pathlib
import shlex
import shutil

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


@pytest.fixture(scope="session")
def set_torch_device(worker_idx):
    """Partition GPUs across workers in a distributed test run."""
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


@pytest.mark.usefixtures("set_torch_device", "symlink_inputs")
@pytest.mark.parametrize(
    "script", sorted(example_dir.glob("*.sh")), ids=lambda x: x.stem
)
def test_command(script, tmp_path, reference_dir, request):
    # The pytest docs say you need to create this directory, but empirically it
    # is already created.
    output_dir = tmp_path
    command_string = parse_bash_file(script)

    command_overrides = shlex.split(command_string)

    config_name = "base"
    # This has to be the first argument if it's present, as it changes the rest
    # of the argument parsing, and for the same reason it has to be passed to
    # hydra.compose explicitly rather than through overrides.
    if command_overrides[0].startswith("--config-name="):
        config_name = command_overrides.pop(0).removeprefix("--config-name=")

    os.chdir(script_dir)

    with hydra.initialize(config_path="../config/inference", version_base="1.2"):
        conf = hydra.compose(config_name=config_name, overrides=command_overrides)
        output_dir = tmp_path
        test_name = request.node.callspec.id.replace(" ", "_").replace("/", "_")
        start_step = conf.diffuser.partial_T or conf.diffuser.T
        overrides = {
            "inference": {
                "num_designs": 1,
                "output_prefix": output_dir / test_name,
                "deterministic": True,
                "final_step": start_step - 2,
            }
        }
        conf = OmegaConf.merge(conf, overrides)

        run_inference(conf)
        handle_test_output(test_name, reference_dir, output_dir, request)


def parse_bash_file(bash_file):
    """Parses a bash file from the examples folder and extracts the command line passed to run_inference.py."""
    with open(bash_file, "r") as f:
        lines = f.readlines()

    script_invocation = "../scripts/run_inference.py "
    commands = []
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        commands.append(line.strip())

    if len(commands) != 1 or not commands[0].startswith(script_invocation):
        raise ValueError(
            f"Expected exactly one non-empty non-comment line starting with '{script_invocation}' in {bash_file}, found:\n"
            + "\n".join(commands)
        )

    return commands[0].removeprefix(script_invocation).strip()


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

    if "inference.ckpt_override_path" in flattened:
        flattened["inference.ckpt_override_path"] = pathlib.Path(
            flattened["inference.ckpt_override_path"]
        ).stem.removesuffix("_ckpt")

    # This is not an exciting property of the test
    flattened.pop("inference.random_seed", None)

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

    config_name = "base"
    if "config_name" in spec:
        config_name = spec.pop("config_name")

    with hydra.initialize(config_path="../config/inference", version_base="1.2"):
        conf = hydra.compose(config_name=config_name)
        # hydra.compose has an overrides argument but it only accepts
        # dot-notation string syntax like
        # "configmap.contigs=[A151-180/70-70/A251-300]" for some reason. It
        # seems annoying and error-prone to convert our already structured input
        # into that format (even though I already wrote a flatten function to
        # construct test ids: that is far more appropriate when a string is the
        # end goal)
        output_dir = tmp_path
        test_name = request.node.callspec.id.replace(" ", "_").replace("/", "_")
        conf = OmegaConf.merge(conf, spec)
        # First resolve the diffuser configuration
        start_step = conf.diffuser.partial_T or conf.diffuser.T
        overrides = {
            "inference": {
                "num_designs": 1,
                "output_prefix": output_dir / test_name,
                "deterministic": True,
                "final_step": start_step - 2,
                "random_seed": 1337,
            }
        }
        conf = OmegaConf.merge(conf, overrides)
        print(f"Running test {test_name} with config:\n{OmegaConf.to_yaml(conf)}")
        run_inference(conf)
        handle_test_output(test_name, reference_dir, output_dir, request)


if __name__ == "__main__":
    pytest.main()
