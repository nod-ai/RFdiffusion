#!/usr/bin/env python3
import copy
import pathlib

import yaml

CONFIG_INPUT = pathlib.Path(__file__).with_name("configs.base.yaml")
CONFIG_OUTPUT = CONFIG_INPUT.with_name("configs.gen.yaml")

DEFAULT_LENGTHEN_SETTINGS = (100, 1000)

LENGTHEN_SETTINGS = {
    "nickel": (120, 1200),
    "cyclic_oligos": (120, 1200),
    "tetrahedral_oligos": (120, 1200),
    "dihedral_oligos": (120, 1200),
}

LENGTH_PAD_WIDTH = len(str(1200))


def lengthen_contig(contig, lengthen_by):
    contig_list = contig[0].strip().split()

    contig_list = [contig.split("/") for contig in contig_list]

    expansion_points = []
    for i, con in enumerate(contig_list):
        for j, subcon in enumerate(con):
            # Because we use fixed lengths, these are never ranges
            if subcon.isdecimal() and subcon != "0":
                expansion_points.append((i, j))

    expansion_point_count = len(expansion_points)

    if expansion_point_count == 0:
        print(f"WARNING: No expansion points for {contig}")
        return contig

    if lengthen_by % expansion_point_count != 0:
        raise ValueError(
            f"Contig {contig} has {expansion_point_count} expansion points,"
            f" which does not evenly divide {lengthen_by=}"
        )

    point_lengthen_by = lengthen_by // expansion_point_count

    for expansion_point in expansion_points:
        i, j = expansion_point
        contig_list[i][j] = str(int(contig_list[i][j]) + point_lengthen_by)

    return [
        " ".join(["/".join([subcon for subcon in contig]) for contig in contig_list])
    ]


def expand_design(name, config):
    new_tests = {}

    length = int(config.get("contigmap").get("length"))
    lengthen_step, max_length = LENGTHEN_SETTINGS.get(name, DEFAULT_LENGTHEN_SETTINGS)

    new_length = length
    contig = config["contigmap"].get("contigs")
    for new_length in range(length + lengthen_step, max_length + 1, lengthen_step):
        new_config = copy.deepcopy(config)
        lengthen_by = new_length - length
        if contig:
            new_contig = lengthen_contig(contig, lengthen_by)
            new_config["contigmap"]["contigs"] = new_contig
        new_config["contigmap"]["length"] = str(new_length)
        new_tests[f"{name}_{new_length:0{LENGTH_PAD_WIDTH}}"] = new_config

    return new_tests


def extract_single_value_range(value):
    """Extracts a single value from a range string formatted as 'n-n'."""
    if "-" not in value:
        raise ValueError(
            f"Expected range format 'n-n', but got: {value}."
            f" Passing only a single value n implicitly samples from [0, n]"
        )
    low, high = value.split("-")
    if low != high:
        raise ValueError(f"Expected deterministic range n-n, but got: {value}")
    return int(low)


def expand_scaffoldguided(name, config):
    new_tests = {}
    length = int(config["contigmap"]["length"])
    lengthen_step, max_length = LENGTHEN_SETTINGS.get(name, DEFAULT_LENGTHEN_SETTINGS)
    assert lengthen_step % 2 == 0, "Scaffold guided lengthen must be even"

    sg = config["scaffoldguided"]
    sc = extract_single_value_range(sg["sampled_C"])
    sn = extract_single_value_range(sg["sampled_N"])
    for new_length in range(length + lengthen_step, max_length + 1, lengthen_step):
        new_config = copy.deepcopy(config)
        lengthen_by = new_length - length

        new_sc = sc + (lengthen_by // 2)
        new_sn = sn + (lengthen_by // 2)

        # Note that unlike contigs, these have to be expressed as ranges because
        # a single number n is interpretered as 0-n
        new_config["scaffoldguided"]["sampled_C"] = f"{new_sc}-{new_sc}"
        new_config["scaffoldguided"]["sampled_N"] = f"{new_sn}-{new_sn}"
        new_config["contigmap"]["length"] = str(new_length)
        new_tests[f"{name}_{new_length:0{LENGTH_PAD_WIDTH}}"] = new_config

    return new_tests


def generate_configs(tests):
    small_tests = {}
    large_tests = copy.deepcopy(tests["large"])

    for design, config in tests["small"].items():
        try:
            length = config["contigmap"]["length"]
            small_tests[f"{design}_{int(length):04}"] = config

            if design.startswith("partialdiffusion"):
                continue
            elif "contigs" in config["contigmap"]:
                new_large_tests = expand_design(design, config)
            elif "scaffoldguided" in config:
                new_large_tests = expand_scaffoldguided(design, config)
            else:
                raise ValueError(f"Don't know how to expand {design}")
            large_tests.update(new_large_tests)

        except Exception as e:
            raise RuntimeError(f"Error expanding {design}") from e

    return {"small": small_tests, "large": large_tests}


class CustomDumper(yaml.SafeDumper):
    """Custom YAML dumper to make the generated output better match existing configs."""

    def represent_str(self, data):
        # Quote strings with double quotes
        return self.represent_scalar("tag:yaml.org,2002:str", data, style='"')

    def represent_mapping(self, tag, mapping, flow_style=None):
        # Always quote string values and never quote string keys
        node = super().represent_mapping(tag, mapping, flow_style)
        for key_node, _ in node.value:
            if key_node.style == '"':
                key_node.style = None
        return node

    def represent_list(self, data):
        # Use flow style (i.e. inline json) for lists with single items or
        # simple strings. This comes up a lot with contigs in particular.
        if len(data) == 1 or all(
            isinstance(item, str) and len(item) < 50 for item in data
        ):
            return self.represent_sequence(
                "tag:yaml.org,2002:seq", data, flow_style=True
            )
        return self.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=False)

    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow=flow, indentless=False)


CustomDumper.add_representer(str, CustomDumper.represent_str)
CustomDumper.add_representer(list, CustomDumper.represent_list)


def main():
    tests = yaml.safe_load(CONFIG_INPUT.read_text())
    new_tests = generate_configs(tests)

    yaml_output = yaml.dump(
        new_tests,
        Dumper=CustomDumper,
        default_flow_style=False,
        sort_keys=False,
        indent=2,
        width=1000,
    )
    CONFIG_OUTPUT.write_text(yaml_output)


if __name__ == "__main__":
    main()
