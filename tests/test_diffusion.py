import datetime
import pathlib
import shutil
import subprocess

import pytest

from rfdiffusion.inference import utils as iu
from rfdiffusion.util import calc_rmsd

script_dir = pathlib.Path(__file__).parent
example_dir = script_dir.parent / "examples"


@pytest.fixture(scope="module")
def output_dir():
    now = datetime.datetime.now()
    now = now.strftime("%Y_%m_%d_%H_%M_%S")
    d = script_dir / f"tests_{now}"
    d.mkdir()
    return d


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


@pytest.fixture(
    scope="module", params=sorted(example_dir.glob("*.sh")), ids=lambda x: x.stem
)
def modified_script(request, output_dir):
    return _write_command(request.param, output_dir)


def test_command(modified_script, output_dir, reference_dir, symlink_inputs, request):
    print(f"Running {modified_script}")
    subprocess.run(["bash", modified_script], check=True)
    test_name = modified_script.stem
    test_files = list((output_dir / "example_outputs" / test_name).glob("*.pdb"))
    assert len(test_files) == 1
    test_file = test_files[0]
    test_pdb = iu.parse_pdb(test_file)
    reference_file = reference_dir / test_file.relative_to(output_dir)
    if request.config.getoption("--update-goldens"):
        reference_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(test_file, reference_file)
        print(f"Updated reference file {reference_file}")
    else:
        ref_pdb = iu.parse_pdb(reference_file)
        rmsd = calc_rmsd(
            test_pdb["xyz"][:, :3].reshape(-1, 3), ref_pdb["xyz"][:, :3].reshape(-1, 3)
        )[0]
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
            if not (line.startswith("python") or line.startswith("../")):
                out_lines.append(line)
            else:
                command = line.strip()
        if not command.startswith("python"):
            command = f"python {command}"
    # get the partial_T
    if "partial_T" in command:
        final_step = int(command.split("partial_T=")[1].split(" ")[0]) - 2
    else:
        final_step = 48

    test_name = bash_file.stem

    # Override these. It's ok if they're already specified, as last one wins
    command = f"{command} inference.num_designs=1 inference.output_prefix={output_dir}/example_outputs/{test_name}/{test_name} inference.deterministic=True inference.final_step={final_step}"

    out_lines.append(command)

    # write the new command
    output_script = output_dir / bash_file.name
    output_script.write_text(''.join(out_lines))

    return output_script


if __name__ == "__main__":
    pytest.main()
