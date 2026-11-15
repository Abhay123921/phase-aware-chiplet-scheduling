from pathlib import Path
import re
import csv


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_ROOT = ROOT / "timeloop" / "outputs" / "resnet50"
OUTPUT_CSV = ROOT / "results" / "resnet50_timeloop_stats.csv"


def number(value):
    return float(value)


def extract(pattern, text, cast=float, default=None):
    match = re.search(pattern, text, re.MULTILINE)

    if not match:
        return default

    try:
        return cast(match.group(1))
    except (ValueError, TypeError):
        return default


def get_section(text, start_pattern, end_pattern=None):
    match = re.search(
        start_pattern,
        text,
        re.MULTILINE | re.DOTALL,
    )

    if not match:
        return ""

    section = match.group(1)

    if end_pattern:
        end_match = re.search(
            end_pattern,
            section,
            re.MULTILINE | re.DOTALL,
        )

        if end_match:
            section = section[:end_match.start()]

    return section


def get_dataspace_section(level_text, dataspace):
    pattern = (
        rf"^\s*{re.escape(dataspace)}:\s*$"
        rf"(.*?)"
        rf"(?=^\s*(?:Weights|Inputs|Outputs):\s*$|\Z)"
    )

    match = re.search(
        pattern,
        level_text,
        re.MULTILINE | re.DOTALL,
    )

    if not match:
        return ""

    return match.group(1)


def extract_field(text, field, cast=float):
    pattern = (
        rf"^\s*{re.escape(field)}\s*:\s*"
        rf"([0-9.eE+-]+)"
    )

    return extract(pattern, text, cast)


def extract_dataspace(level_text, dataspace):
    section = get_dataspace_section(
        level_text,
        dataspace,
    )

    prefix = dataspace.lower()

    result = {}

    fields = {
        "partition_size": "Partition size",
        "utilized_capacity": "Utilized capacity",
        "reads": "Scalar reads (per-instance)",
        "fills": "Scalar fills (per-instance)",
        "updates": "Scalar updates (per-instance)",
        "shared_bw": "Shared Bandwidth (total)",
        "read_bw": "Read Bandwidth (total)",
        "write_bw": "Write Bandwidth (total)",
        "energy_pj": "Energy (total)",
    }

    for name, field in fields.items():
        result[f"{prefix}_{name}"] = extract_field(
            section,
            field,
        )

    return result


def extract_layer_stats(layer_dir):

    stats_file = layer_dir / "timeloop-mapper.stats.txt"

    if not stats_file.exists():
        return None

    text = stats_file.read_text()

    result = {
        "layer_id": layer_dir.name,
    }

    # ---------------------------------------------------------
    # MAC SECTION
    # ---------------------------------------------------------

    mac_match = re.search(
        r"=== mac ===(.*?)(?=^\s*Level 1\s*$)",
        text,
        re.MULTILINE | re.DOTALL,
    )

    mac_section = mac_match.group(1) if mac_match else ""

    result["mac_instances"] = extract_field(
        mac_section,
        "Instances",
        int,
    )

    result["utilized_mac_instances"] = extract_field(
        mac_section,
        "Utilized instances",
        int,
    )

    result["computes"] = extract_field(
        mac_section,
        "Computes (total)",
    )

    result["cycles"] = extract_field(
        mac_section,
        "Cycles",
    )

    result["mac_energy_pj"] = extract_field(
        mac_section,
        "Energy (total)",
    )

    # ---------------------------------------------------------
    # BUFFER SECTION
    # ---------------------------------------------------------

    buffer_match = re.search(
        r"=== buffer ===(.*?)(?=^\s*Level 4\s*$)",
        text,
        re.MULTILINE | re.DOTALL,
    )

    buffer_section = (
        buffer_match.group(1)
        if buffer_match
        else ""
    )

    for dataspace in ["Weights", "Inputs", "Outputs"]:

        result.update(
            extract_dataspace(
                buffer_section,
                dataspace,
            )
        )

    # ---------------------------------------------------------
    # DRAM SECTION
    # ---------------------------------------------------------

    dram_match = re.search(
        r"=== DRAM ===(.*?)(?=^\s*Networks\s*$)",
        text,
        re.MULTILINE | re.DOTALL,
    )

    dram_section = (
        dram_match.group(1)
        if dram_match
        else ""
    )

    for dataspace in ["Weights", "Inputs", "Outputs"]:

        result.update(
            {
                f"dram_{k}": v
                for k, v in extract_dataspace(
                    dram_section,
                    dataspace,
                ).items()
            }
        )

    # ---------------------------------------------------------
    # SUMMARY STATS
    # ---------------------------------------------------------

    summary_match = re.search(
        r"Summary Stats\s*-+\s*(.*?)(?=\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )

    summary = (
        summary_match.group(1)
        if summary_match
        else ""
    )

    result["gflops_at_1ghz"] = extract(
        r"GFLOPs \(@1GHz\):\s*([0-9.eE+-]+)",
        summary,
    )

    result["utilization"] = extract(
        r"Utilization:\s*([0-9.eE+-]+)",
        summary,
    )

    if result["utilization"] is not None:
        result["utilization"] /= 100.0

    result["summary_cycles"] = extract(
        r"Cycles:\s*([0-9.eE+-]+)",
        summary,
    )

    result["total_energy_uJ"] = extract(
        r"Energy:\s*([0-9.eE+-]+)\s*uJ",
        summary,
    )

    result["edp"] = extract(
        r"EDP\(J\*cycle\):\s*([0-9.eE+-]+)",
        summary,
    )

    result["summary_computes"] = extract(
        r"Computes\s*=\s*([0-9.eE+-]+)",
        summary,
    )

    result["total_fj_per_compute"] = extract(
        r"Total\s*=\s*([0-9.eE+-]+)",
        summary,
    )

    # ---------------------------------------------------------
    # ENERGY BREAKDOWN
    # ---------------------------------------------------------

    result["buffer_fj_per_compute"] = extract(
        r"buffer\s*=\s*([0-9.eE+-]+)",
        summary,
    )

    result["dram_fj_per_compute"] = extract(
        r"DRAM\s*=\s*([0-9.eE+-]+)",
        summary,
    )

    # ---------------------------------------------------------
    # OPERATIONAL INTENSITY
    # ---------------------------------------------------------

    op_match = re.search(
        r"Operational Intensity Stats"
        r"(.*?)(?=^\s*Summary Stats\s*$)",
        text,
        re.MULTILINE | re.DOTALL,
    )

    op_section = (
        op_match.group(1)
        if op_match
        else ""
    )

    result["total_elementwise_ops"] = extract(
        r"Total elementwise ops\s*:\s*([0-9.eE+-]+)",
        op_section,
    )

    result["total_reduction_ops"] = extract(
        r"Total reduction ops\s*:\s*([0-9.eE+-]+)",
        op_section,
    )

    result["total_ops"] = extract(
        r"Total ops\s*:\s*([0-9.eE+-]+)",
        op_section,
    )

    result["total_memory_accesses"] = extract(
        r"Total memory accesses required\s*:\s*([0-9.eE+-]+)",
        op_section,
    )

    result["optimal_op_per_byte"] = extract(
        r"Optimal Op per Byte\s*:\s*([0-9.eE+-]+)",
        op_section,
    )

    # ---------------------------------------------------------
    # DERIVED VALUES
    # ---------------------------------------------------------

    computes = result["computes"]
    cycles = result["cycles"]

    if computes is not None and cycles:
        result["compute_per_cycle"] = (
            computes / cycles
        )
    else:
        result["compute_per_cycle"] = None

    mac_instances = result["mac_instances"]
    utilized = result["utilized_mac_instances"]

    if mac_instances and utilized is not None:
        result["mac_utilization"] = (
            utilized / mac_instances
        )
    else:
        result["mac_utilization"] = None

    return result


def main():

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    layer_dirs = sorted(
        path
        for path in OUTPUT_ROOT.iterdir()
        if path.is_dir()
    )

    print(
        f"Found {len(layer_dirs)} Timeloop "
        f"output directories."
    )

    for layer_dir in layer_dirs:

        print(
            f"Processing: {layer_dir.name}"
        )

        stats = extract_layer_stats(
            layer_dir
        )

        if stats is not None:
            rows.append(stats)

    if not rows:
        print("No statistics found.")
        return

    fieldnames = sorted(
        {
            key
            for row in rows
            for key in row.keys()
        }
    )

    with OUTPUT_CSV.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print(
        f"Extracted {len(rows)} layers."
    )

    print(
        f"Saved to: {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()
