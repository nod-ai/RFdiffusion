import pytest

def pytest_addoption(parser):
    parser.addoption("--update-goldens", action="store_true", help="Update golden reference output files")
