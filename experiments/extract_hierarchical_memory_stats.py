#!/usr/bin/env python3

"""
Extract hierarchical Timeloop statistics without modifying existing
extraction scripts or result files.

Sources:
  timeloop/outputs/resnet50/<layer>/timeloop-mapper.stats.txt

Output:
  results/resnet50_hierarchical_memory_stats.csv

Missing or unreported values are preserved as None/empty CSV cells.
Per-instance values are not multiplied into totals.
"""

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT / "timeloop" / "outputs" / "resnet50"
OUTPUT_FILE = ROOT / "results" / "resnet50_hierarchical_memory_stats.csv"

LEVELS = {
    "mac": "mac",
    "reg": "reg",
    "buffer": "buffer",
    "dram": "DRAM",
}

DATASPACES = ("Weights", "Inputs", "Outputs")


def number(value):
    """Convert a numeric string to int/float; preserve unavailable values."""
    if value is None:
        return None

    value = value.strip().replace(",", "")
    if not value or value in {"-", "N/A", "None"}:
        return None

    try:
        result = float(value)
        return int(result) if result.is_integer() else result
    except ValueError:
        return None


def extract_number(text, field):
    """Extract a numeric value while allowing trailing units or annotations."""
    if not text:
        return None

    pattern = (
        r"^\s*" + re.escape(field) +
        r"\s*:\s*([+-]?(?:[0-9,]+(?:\.[0-9]*)?|\.[0-9]+)"
        r"(?:[eE][+-]?\d+)?|-)"
    )
    match = re.search(pattern, text, flags=re.MULTILINE)
    return number(match.group(1)) if match else None


def extract_level_section(text, marker):
    """
    Extract the first hardware-level section for a marker.
    The section ends at the next Level heading or major summary heading.
    """
    match = re.search(
        r"^\s*===\s*" + re.escape(marker) + r"\s*===\s*$",
        text,
        flags=re.MULTILINE,
    )
    if not match:
        return None

    start = match.end()
    remainder = text[start:]

    end_match = re.search(
        r"^\s*(?:Level\s+\d+\s*$|Networks\s*$|"
        r"Operational Intensity Stats|Summary Stats)",
        remainder,
        flags=re.MULTILINE,
    )

    end = start + end_match.start() if end_match else len(text)
    return text[start:end]


def extract_dataspace_section(level_text, dataspace):
    """Extract one dataspace block from a hardware-level section."""
    if not level_text:
        return None

    match = re.search(
        r"^\s*" + re.escape(dataspace) + r"\s*:\s*$",
        level_text,
        flags=re.MULTILINE,
    )
    if not match:
        return None

    start = match.end()
    remainder = level_text[start:]

    next_match = re.search(
        r"^\s*(?:Weights|Inputs|Outputs)\s*:\s*$",
        remainder,
        flags=re.MULTILINE,
    )

    end = next_match.start() if next_match else len(remainder)
    return remainder[:end]


def extract_operational_intensity(text):
    """Extract global and per-level Operational Intensity statistics."""
    result = {}

    match = re.search(
        r"Operational Intensity Stats(.*?)(?=Summary Stats)",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return result

    section = match.group(1)

    global_fields = {
        "total_elementwise_ops": "Total Elementwise Ops",
        "total_reduction_ops": "Total Reduction Ops",
        "total_ops": "Total Ops",
        "total_memory_accesses_required": "Total Memory Accesses Required",
        "optimal_op_per_byte": "Optimal Op per Byte",
    }

    for output_name, source_name in global_fields.items():
        result["oi_" + output_name] = extract_number(section, source_name)

    for level in ("reg", "inter_PE_spatial", "buffer", "DRAM"):
        level_match = re.search(
            r"^\s*===\s*" + re.escape(level) + r"\s*===\s*$"
            r"(.*?)(?=^\s*===|\Z)",
            section,
            flags=re.MULTILINE | re.DOTALL,
        )

        level_text = level_match.group(1) if level_match else None

        result[f"oi_{level}_total_scalar_accesses"] = (
            extract_number(level_text, "Total scalar accesses")
            if level_text is not None else None
        )
        result[f"oi_{level}_op_per_byte"] = (
            extract_number(level_text, "Op per Byte")
            if level_text is not None else None
        )

    # Presence of an empty marker is not evidence of zero communication.
    result["oi_inter_PE_spatial_marker_present"] = (
        "=== inter_PE_spatial ===" in section
    )

    return result


def extract_summary(text):
    """Extract aggregate layer-level summary statistics using actual labels."""
    result = {}

    match = re.search(
        r"Summary Stats(.*)\Z",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    section = match.group(1) if match else ""

    def value(pattern):
        found = re.search(pattern, section, flags=re.MULTILINE)
        return number(found.group(1)) if found else None

    result["summary_gflops"] = value(
        r"^\s*GFLOPs\s*\(@1GHz\)\s*:\s*([0-9.eE+-]+)"
    )
    result["summary_utilization"] = value(
        r"^\s*Utilization\s*:\s*([0-9.eE+-]+)%?"
    )
    result["summary_cycles"] = value(
        r"^\s*Cycles\s*:\s*([0-9.eE+-]+)"
    )
    result["summary_energy_uj"] = value(
        r"^\s*Energy\s*:\s*([0-9.eE+-]+)\s*uJ"
    )
    result["summary_edp"] = value(
        r"^\s*EDP\s*\(J\*cycle\)\s*:\s*([0-9.eE+-]+)"
    )
    result["summary_area_mm2"] = value(
        r"^\s*Area\s*:\s*([0-9.eE+-]+)\s*mm"
    )
    result["summary_computes"] = value(
        r"^\s*Computes\s*=\s*([0-9.eE+-]+)"
    )

    fj_block = re.search(
        r"^\s*fJ/Compute\s*$([\\s\\S]*)",
        section,
        flags=re.MULTILINE,
    )
    fj_text = fj_block.group(1) if fj_block else ""

    for output_name, source_name in (
        ("summary_mac_fj_per_compute", "mac"),
        ("summary_buffer_fj_per_compute", "buffer"),
        ("summary_dram_fj_per_compute", "DRAM"),
        ("summary_total_fj_per_compute", "Total"),
    ):
        found = re.search(
            r"^\s*" + re.escape(source_name) +
            r"\s*=\s*([0-9.eE+-]+)",
            fj_text,
            flags=re.MULTILINE,
        )
        result[output_name] = number(found.group(1)) if found else None

    return result

def extract_mac_stats(text, section):
    result = {}

    result["mac_hardware_instances"] = extract_number(section, "Instances")
    result["mac_utilized_instances"] = extract_number(
        section, "Utilized instances"
    )
    result["mac_computes_total"] = extract_number(section, "Computes (total)")
    result["mac_cycles"] = extract_number(section, "Cycles")
    result["mac_energy_total_pj"] = extract_number(section, "Energy (total)")

    return result


def extract_memory_level(text, level_name, marker):
    """Extract level specifications and per-dataspace statistics."""
    result = {}
    section = extract_level_section(text, marker)

    result[f"{level_name}_section_present"] = section is not None

    if section is None:
        return result

    result[f"{level_name}_hardware_instances"] = extract_number(
        section, "Instances"
    )
    result[f"{level_name}_cycles"] = extract_number(section, "Cycles")

    fields = {
        "partition_size": "Partition size",
        "utilized_capacity": "Utilized capacity",
        "utilized_instances_max": "Utilized instances (max)",
        "utilized_clusters_max": "Utilized clusters (max)",

        "scalar_reads_per_instance": "Scalar reads (per-instance)",
        "scalar_fills_per_instance": "Scalar fills (per-instance)",
        "scalar_updates_per_instance": "Scalar updates (per-instance)",
        "temporal_reductions_per_instance": "Temporal reductions (per-instance)",
        "address_generations_per_cluster": "Address generations (per-cluster)",

        "energy_per_scalar_access_pj": "Energy (per-scalar-access)",
        "energy_per_instance_pj": "Energy (per-instance)",
        "energy_total_pj": "Energy (total)",

        "temporal_reduction_energy_per_instance_pj":
            "Temporal Reduction Energy (per-instance)",
        "temporal_reduction_energy_total_pj":
            "Temporal Reduction Energy (total)",

        "address_generation_energy_per_cluster_pj":
            "Address Generation Energy (per-cluster)",
        "address_generation_energy_total_pj":
            "Address Generation Energy (total)",

        "bandwidth_consumption_scale": "Bandwidth Consumption Scale",

        "shared_bandwidth_per_instance_words_per_cycle":
            "Shared Bandwidth (per-instance)",
        "shared_bandwidth_total_words_per_cycle":
            "Shared Bandwidth (total)",
        "read_bandwidth_per_instance_words_per_cycle":
            "Read Bandwidth (per-instance)",
        "read_bandwidth_total_words_per_cycle":
            "Read Bandwidth (total)",
        "write_bandwidth_per_instance_words_per_cycle":
            "Write Bandwidth (per-instance)",
        "write_bandwidth_total_words_per_cycle":
            "Write Bandwidth (total)",
    }

    for dataspace in DATASPACES:
        block = extract_dataspace_section(section, dataspace)

        for output_field, source_field in fields.items():
            key = f"{level_name}_{dataspace.lower()}_{output_field}"
            result[key] = (
                extract_number(block, source_field)
                if block is not None else None
            )

    return result


def extract_layer(stats_path):
    text = stats_path.read_text(errors="replace")
    row = {"layer": stats_path.parent.name}

    # MAC statistics
    mac_section = extract_level_section(text, LEVELS["mac"])
    row["mac_section_present"] = mac_section is not None

    if mac_section is not None:
        row.update(extract_mac_stats(text, mac_section))
    else:
        row.update({
            "mac_hardware_instances": None,
            "mac_utilized_instances": None,
            "mac_computes_total": None,
            "mac_cycles": None,
            "mac_energy_total_pj": None,
        })

    # Register, buffer and DRAM statistics
    for output_name in ("reg", "buffer", "dram"):
        row.update(
            extract_memory_level(
                text,
                output_name,
                LEVELS[output_name],
            )
        )

    # Operational Intensity and aggregate summary
    row.update(extract_operational_intensity(text))
    row.update(extract_summary(text))

    # Consistency checks: report differences, do not silently correct values.
    mac_cycles = row.get("mac_cycles")
    summary_cycles = row.get("summary_cycles")

    row["mac_summary_cycles_match"] = (
        mac_cycles == summary_cycles
        if mac_cycles is not None and summary_cycles is not None
        else None
    )

    mac_computes = row.get("mac_computes_total")
    summary_computes = row.get("summary_computes")

    row["mac_summary_computes_match"] = (
        mac_computes == summary_computes
        if mac_computes is not None and summary_computes is not None
        else None
    )

    return row


def main():
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_ROOT}")

    stats_files = sorted(INPUT_ROOT.glob("*/timeloop-mapper.stats.txt"))

    if not stats_files:
        raise FileNotFoundError(
            f"No timeloop-mapper.stats.txt files found under {INPUT_ROOT}"
        )

    rows = []
    failures = []

    for stats_path in stats_files:
        try:
            rows.append(extract_layer(stats_path))
        except Exception as exc:
            failures.append((stats_path.parent.name, str(exc)))

    if not rows:
        raise RuntimeError("No layers could be parsed.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    columns = sorted({key for row in rows for key in row})

    with OUTPUT_FILE.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 72)
    print("HIERARCHICAL TIMELOOP STATISTICS EXTRACTION")
    print("=" * 72)
    print(f"Input layers discovered : {len(stats_files)}")
    print(f"Layers successfully parsed: {len(rows)}")
    print(f"Parsing failures        : {len(failures)}")
    print(f"CSV columns             : {len(columns)}")
    print(f"Output                  : {OUTPUT_FILE}")
    print("-" * 72)

    for level in ("mac", "reg", "buffer", "dram"):
        count = sum(
            bool(row.get(f"{level}_section_present"))
            for row in rows
        )
        print(f"{level.upper():8s} sections parsed: {count}/{len(rows)}")

    oi_interpe_present = sum(
        bool(row.get("oi_inter_PE_spatial_marker_present"))
        for row in rows
    )
    oi_interpe_values = sum(
        row.get("oi_inter_PE_spatial_total_scalar_accesses") is not None
        for row in rows
    )

    print("-" * 72)
    print(
        "Operational Intensity inter_PE_spatial markers: "
        f"{oi_interpe_present}/{len(rows)}"
    )
    print(
        "inter_PE_spatial scalar-access values reported: "
        f"{oi_interpe_values}/{len(rows)}"
    )

    cycles_matches = sum(
        row.get("mac_summary_cycles_match") is True for row in rows
    )
    computes_matches = sum(
        row.get("mac_summary_computes_match") is True for row in rows
    )

    print("-" * 72)
    print(f"MAC vs summary cycle matches   : {cycles_matches}/{len(rows)}")
    print(f"MAC vs summary compute matches : {computes_matches}/{len(rows)}")

    if failures:
        print("-" * 72)
        print("FAILURES:")
        for layer, error in failures:
            print(f"  {layer}: {error}")

    print("=" * 72)


if __name__ == "__main__":
    main()
