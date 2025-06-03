import statistics

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--update-goldens",
        action="store_true",
        help="Update golden reference output files",
    )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Custom terminal summary to display RMSD values and reference file creation."""
    tr = terminalreporter
    rmsds = {}
    reference_files = {}

    for report in tr.getreports(""):
        for name, value in getattr(report, "user_properties", []):
            test_name = report.nodeid
            if name == "rmsd":
                rmsds[test_name] = value
            elif name == "reference_file":
                reference_files[test_name] = value

    if rmsds:
        tr.write_sep("=", "RMSD Report")
        if len(rmsds) > 1:
            value_sorted_rmsds = sorted(rmsds.items(), key=lambda x: x[1])
            min_rmsd = value_sorted_rmsds[0]
            max_rmsd = value_sorted_rmsds[-1]

            rmsd_values = list(rmsds.values())
            mean_rmsd = statistics.fmean(rmsd_values)
            std_rmsd = statistics.stdev(rmsd_values)

            tr.write_line(f"mean:   {mean_rmsd:.3}")
            tr.write_line(f"stddev: {std_rmsd:.3}")
            tr.write_line(f"max:    {max_rmsd[1]:.3}")
            tr.write_line(f"min:    {min_rmsd[1]:.3}")

            tr.write_line(f"Max RMSD from: {max_rmsd[0]}")
            tr.write_line(f"Min RMSD from: {min_rmsd[0]}")

            tr.write_line("")

        tr.write_line("RMSD for all tests:")

        for test_name, rmsd in sorted(rmsds.items()):
            # Make all the numbers aligned for ease of reading
            tr.write_line(f"{rmsd:6.3}  {test_name}")
        tr.write_line("")

    if reference_files:
        tr.write_sep("=", "Reference Files Report")
        tr.write_line("Reference files written to...")
        for test_name, ref_file in sorted(reference_files.items()):
            tr.write_line(f"{test_name}")
            # filepaths tend to be long and wrap poorly. Give them their own line.
            tr.write_line(f"    {ref_file}")
        tr.write_line("")


# When running with --capture=no (-s), the first line of output gets printed on
# the same line as the test name. So always print a newline as the first thing.
# It seems weird that pytest doesn't just add the linebreak by default.
@pytest.fixture(autouse=True)
def print_starting_newline():
    print()


# If pytest-xdist is installed (or theoretically anthing that provides the
# `worker_id` fixture, we make worker index available. Otherwise we just set it
# to 0.
try:

    @pytest.fixture(scope="session")
    def worker_idx(worker_id):
        if worker_id == "master":
            return 0

        return int(worker_id.removeprefix("gw"))

except:

    @pytest.fixture(scope="session")
    def worker_idx():
        return 0
