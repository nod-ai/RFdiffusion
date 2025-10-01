#!/usr/bin/env python3
"""
Script to process RFdiffusion worker log files and create a CSV tracking
design completion over elapsed time.

Usage: python process_distributed_logs.py <log_directory>

Example: python process_distributed_logs.py outputs/2025-08-13/17-44-24
"""

import sys
import re
import csv
from datetime import datetime
import pathlib


INFO_LOG_PATTERN = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) INFO:"
INFO_LOG_REGEX = re.compile(INFO_LOG_PATTERN)
LENGTH_LOG_REGEX = re.compile(INFO_LOG_PATTERN + r" Sampled contig .* with length (\d+)")
COMPLETE_LOG_REGEX = re.compile(INFO_LOG_PATTERN + r" Finished design in (\d+) seconds")

def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse timestamp from log line format: 2025-08-13 17:44:38,802"""
    return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")


def extract_completion_times(log_file) -> list[dict[str, datetime|int]]:
    """
    Extract design completion times from a single log file.

    Returns list of tuples: (completion_timestamp, design_time_seconds)
    """
    completions = []

    with open(log_file, "r") as f:
        length = None
        for line in f:
            line = line.strip()
            match_length = LENGTH_LOG_REGEX.match(line)
            if match_length:
                if length is not None:
                    print(f"ERROR: {log_file}: Expected length to be unset when parsing sampled contig line, but got '{length}' at line: {line}")
                    sys.exit(1)
                length = int(match_length.group(2))

            match_complete = COMPLETE_LOG_REGEX.match(line)
            if match_complete:
                if length is None:
                    print(f"ERROR: {log_file}: Expected length to be set when parsing design completion, but got None at line: {line}")
                timestamp_str = match_complete.group(1)
                design_time = int(match_complete.group(2))
                timestamp = parse_timestamp(timestamp_str)
                completions.append({"timestamp": timestamp, "design_time": design_time, "length": length})
                length = None

    return completions


def find_start_time(log_dir) -> datetime:
    root_logs = list(log_dir.glob("*.log"))
    if len(root_logs) != 1:
        print(
            f"ERROR: Expected exactly one root log, but got {len(root_logs)}: root_logs"
        )
    root_log = root_logs[0]

    with open(root_log) as f:
        first_line = f.readline().strip()
        match = INFO_LOG_REGEX.search(first_line)
        if not match:
            print(
                f"ERROR: did not find timestamp in first line of root log {root_log}: {first_line}"
            )
            sys.exit(1)

        return parse_timestamp(match.group(1))


def process_all_logs(log_directory: str) -> list[tuple[float, int]]:
    """
    Process all worker logs and return sorted list of (elapsed_s, completed_designs).

    Returns list sorted by elapsed time.
    """
    all_completions = []

    log_dir = pathlib.Path(log_directory)

    for subdir in log_dir.iterdir():
        if subdir.is_dir() and subdir.name.isdigit():
            worker = int(subdir.name)

            log_files = list(subdir.glob(f"worker.{worker}.*.log"))
            if len(log_files) != 1:
                print(f"Found multiple log files in {subdir}", file=sys.stderr)
                sys.exit(1)

            log_file = log_files[0]
            print(f"Processing {log_file}")
            completions = extract_completion_times(log_file)
            for c in completions:
                c["worker"] = worker
            all_completions.extend(completions)

    if not all_completions:
        print("Error: No design completions found in any log files", file=sys.stderr)
        sys.exit(1)

    # Find start time
    start_time = find_start_time(log_directory)
    if start_time is None:
        print("Error: Could not determine start time")
        return []

    print(f"Start time: {start_time}")
    print(f"Found {len(all_completions)} design completions")

    all_completions.append({"timestamp": start_time})

    for c in all_completions:
        c["elapsed_s"] = (c["timestamp"] - start_time).total_seconds()

    all_completions.sort(key=lambda x: x["elapsed_s"])

    for i, c in enumerate(all_completions):
        c["completed_designs"] = i

    return all_completions


def write_csv(data: list[dict[str, float]], output_file: str):
    """Write the results to CSV file."""
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, dialect="unix", fieldnames=data[-1].keys())
        writer.writeheader()
        writer.writerows(data)

    print(f"Results written to {output_file}")


def main():
    fn = pathlib.Path(__file__).name
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        print(f"Usage: python {fn} <log_directory>")
        print(f"Example: python {fn} outputs/2025-08-13/17-44-24")
        sys.exit(1)

    log_directory = pathlib.Path(sys.argv[1])

    if not log_directory.is_dir():
        print(f"Error: {log_directory} is not a directory", file=sys.stderr)
        sys.exit(1)

    data = process_all_logs(log_directory)

    if not data:
        print("No data to process")
        sys.exit(1)

    write_csv(data, log_directory / "design_completion_times.csv")

    print(f"\nSummary:")
    print(f"Total designs completed: {len(data)}")
    print(f"Time range: {data[0]["timestamp"]} to {data[-1]["timestamp"]}")
    total_elapsed_s = data[-1]["elapsed_s"]
    print(f"Total elapsed time: {total_elapsed_s:.1f}s ({total_elapsed_s/60:.1f} minutes)")


if __name__ == "__main__":
    main()
