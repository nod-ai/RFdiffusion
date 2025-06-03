import os
import pathlib
import shutil
import subprocess

import pytest

from rfdiffusion.inference import utils as iu
from rfdiffusion.util import calc_rmsd

script_dir = pathlib.Path(__file__).parent
example_dir = script_dir.parent / "examples"


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


@pytest.mark.usefixtures("symlink_inputs")
@pytest.mark.parametrize(
    "script", sorted(example_dir.glob("*.sh")), ids=lambda x: x.stem
)
def test_command(script, tmp_path, reference_dir, child_env, request):
    output_dir = tmp_path
    # The pytest docs say you need to create this directory, but empirically it
    # is already created.
    # output_dir.mkdir()
    modified_script = _write_command(script, output_dir)
    print(f"Running {modified_script}")
    # cwd is required because the scripts use relative paths like `../scripts/run_inference.py`
    subprocess.run(["bash", modified_script], check=True, cwd=script_dir, env=child_env)
    test_name = modified_script.stem
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
        test_pdb = iu.parse_pdb(test_file)
        ref_pdb = iu.parse_pdb(reference_file)
        rmsd = calc_rmsd(
            test_pdb["xyz"][:, :3].reshape(-1, 3), ref_pdb["xyz"][:, :3].reshape(-1, 3)
        )[0].item()
        request.node.user_properties.append(("rmsd", rmsd))
        print(f"RMSD={rmsd:.3}")

        assert rmsd == pytest.approx(0, rel=0, abs=0.01)


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


if __name__ == "__main__":
    pytest.main()
