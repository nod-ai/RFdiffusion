#!/usr/bin/env python3
import copy
import pathlib

import yaml

CONFIG_INPUT = pathlib.Path(__file__).with_name("configs.base.yaml")
CONFIG_OUTPUT = CONFIG_INPUT.with_name("configs.gen.yaml")

MAX_LENGTH = 1200
LENGTHEN_STEP = {
    "nickel": 96,
    "cyclic_oligos": 120,
    "tetrahedral_oligos": 120,
    "partialdiffusion_full": 0,
    "partialdiffusion_withseq": 0,
    "partialdiffusion_multipleseq": 0,
    "timbarrel_short": 0,
}


class CustomDumper(yaml.SafeDumper):
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
        # Use flow style for lists with single items or simple strings
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


def generate_configs():
    tests = yaml.safe_load(CONFIG_INPUT.read_text())

    new_tests = {}

    for design, config in tests.items():
        try:
            length = config.get("contigmap", {}).get("length")
            lengthen_step = LENGTHEN_STEP.get(design, 100)
            if not length or lengthen_step == 0:
                new_tests[design] = config
                continue
            length = int(config["contigmap"]["length"])
            new_length = length
            contig = config["contigmap"].get("contigs")
            while new_length <= MAX_LENGTH:
                new_config = copy.deepcopy(config)
                lengthen_by = new_length - length
                if contig:
                    new_contig = lengthen_contig(contig, lengthen_by)
                    new_config["contigmap"]["contigs"] = new_contig
                new_config["contigmap"]["length"] = str(new_length)
                new_tests[f"{design}_{new_length}"] = new_config
                new_length += lengthen_step
        except:
            print(f"Error generating {design}")
            raise

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
    generate_configs()
