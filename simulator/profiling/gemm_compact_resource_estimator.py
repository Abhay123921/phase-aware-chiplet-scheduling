"""Estimated resource profiles for mapping-derived GEMM execution groups.

Timeloop supplies layer-level measurements. Group-level resource values are
allocated estimates based on logical MAC and tensor-access shares; they are
not per-group hardware measurements.
"""

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class GEMMGroupResourceProfile:
    group_id: str
    layer_id: str
    estimated_compute_work: float
    reference_cycles: float
    estimated_input_accesses: float
    estimated_weight_accesses: float
    estimated_output_accesses: float
    input_bw: float
    weight_bw: float
    output_bw: float
    dram_input_bw: float
    dram_weight_bw: float
    dram_output_bw: float
    estimation_method: str = (
        "estimated group metrics from layer-level Timeloop measurements allocated by logical GEMM access shares"
    )

    def to_dict(self):
        return asdict(self)


def _metric(profile, name):
    if name not in profile:
        raise ValueError(f"Missing layer metric: {name}")
    value = float(profile[name])
    if value < 0:
        raise ValueError(f"Negative layer metric: {name}")
    return value


def _access_map(group_accesses, group_id):
    if group_id not in group_accesses:
        raise ValueError(f"Missing tensor accesses for group: {group_id}")
    entries = group_accesses[group_id].get("tensor_accesses")
    if entries is None:
        raise ValueError(f"Missing tensor_accesses for group: {group_id}")
    return {entry.tensor_name: entry for entry in entries}


def estimate_gemm_execution_group_resources(
    *,
    execution_group_graph,
    group_profiles,
    group_accesses,
    layer_profile,
):
    """Allocate layer-level Timeloop measurements across GEMM groups.

    Required layer_profile fields:
      reference_compute_work, reference_execution_cycles,
      buffer_input_shared_bw, buffer_weight_shared_bw,
      buffer_output_shared_bw,
      dram_input_transfers, dram_weight_transfers,
      dram_output_transfers.

    Buffer bandwidths are Timeloop Level-3 shared-buffer bandwidths
    in words/cycle. DRAM transfers are layer-level word counts.

    Input and weight allocations use logical read-element shares.
    Output allocations use unique output-element shares because the LDG
    does not specify the reduction implementation or output read/write split.
    DRAM bandwidth estimates are allocated transfers divided by group cycles.
    """
    groups = execution_group_graph.groups
    if not groups:
        raise ValueError("Execution-group graph is empty")

    if set(group_profiles) != set(groups):
        raise ValueError("Group profile IDs do not match execution groups")

    compute_total = _metric(layer_profile, "reference_compute_work")
    cycles_total = _metric(layer_profile, "reference_execution_cycles")
    if compute_total <= 0 or cycles_total <= 0:
        raise ValueError("Reference compute work and cycles must be positive")

    buffer_totals = {
        "input": _metric(layer_profile, "buffer_input_shared_bw"),
        "weight": _metric(layer_profile, "buffer_weight_shared_bw"),
        "output": _metric(layer_profile, "buffer_output_shared_bw"),
    }
    dram_totals = {
        "input": _metric(layer_profile, "dram_input_transfers"),
        "weight": _metric(layer_profile, "dram_weight_transfers"),
        "output": _metric(layer_profile, "dram_output_transfers"),
    }

    records = {}
    totals = {
        "mac": 0.0,
        "input_reads": 0.0,
        "weight_reads": 0.0,
        "output_unique": 0.0,
    }

    for group_id, group in groups.items():
        profile = group_profiles[group_id]
        if profile.work_count is None or profile.work_count <= 0:
            raise ValueError(f"{group_id}: missing positive logical MAC count")

        accesses = _access_map(group_accesses, group_id)
        if len(group.input_tensors) != 2 or len(group.output_tensors) != 1:
            raise ValueError(f"{group_id}: expected two inputs and one output")

        input_access = accesses[group.input_tensors[0]]
        weight_access = accesses[group.input_tensors[1]]
        output_access = accesses[group.output_tensors[0]]

        if input_access.read_elements is None or weight_access.read_elements is None:
            raise ValueError(f"{group_id}: input/weight read counts are required")
        if output_access.unique_elements is None:
            raise ValueError(f"{group_id}: unique output count is required")

        values = {
            "mac": float(profile.work_count),
            "input_reads": float(input_access.read_elements),
            "weight_reads": float(weight_access.read_elements),
            "output_unique": float(output_access.unique_elements),
        }
        if any(value < 0 for value in values.values()):
            raise ValueError(f"{group_id}: negative logical access count")

        records[group_id] = values
        for key, value in values.items():
            totals[key] += value

    if any(value <= 0 for value in totals.values()):
        raise ValueError("Invalid aggregate logical work/access totals")

    results = []
    for group_id, group in groups.items():
        values = records[group_id]
        mac_share = values["mac"] / totals["mac"]
        input_share = values["input_reads"] / totals["input_reads"]
        weight_share = values["weight_reads"] / totals["weight_reads"]
        output_share = values["output_unique"] / totals["output_unique"]

        group_cycles = cycles_total * mac_share
        if group_cycles <= 0:
            raise ValueError(f"{group_id}: estimated cycles must be positive")

        results.append(GEMMGroupResourceProfile(
            group_id=group_id,
            layer_id=execution_group_graph.workload_id,
            estimated_compute_work=compute_total * mac_share,
            reference_cycles=group_cycles,
            estimated_input_accesses=values["input_reads"],
            estimated_weight_accesses=values["weight_reads"],
            estimated_output_accesses=values["output_unique"],
            input_bw=buffer_totals["input"] * input_share / mac_share,
            weight_bw=buffer_totals["weight"] * weight_share / mac_share,
            output_bw=buffer_totals["output"] * output_share / mac_share,
            dram_input_bw=dram_totals["input"] * input_share / group_cycles,
            dram_weight_bw=dram_totals["weight"] * weight_share / group_cycles,
            dram_output_bw=dram_totals["output"] * output_share / group_cycles,
        ))

    tolerance = 1e-8

    def conserved(actual, expected, label):
        if abs(actual - expected) > max(1e-8, abs(expected) * tolerance):
            raise ValueError(f"{label} conservation failed: {actual} != {expected}")

    conserved(
        sum(item.estimated_compute_work for item in results),
        compute_total,
        "Compute work",
    )
    conserved(
        sum(item.reference_cycles for item in results),
        cycles_total,
        "Reference cycles",
    )

    # Time-weighted group buffer bandwidths must reproduce Timeloop totals.
    for field, expected in (
        ("input_bw", buffer_totals["input"]),
        ("weight_bw", buffer_totals["weight"]),
        ("output_bw", buffer_totals["output"]),
    ):
        actual = sum(
            getattr(item, field) * item.reference_cycles / cycles_total
            for item in results
        )
        conserved(actual, expected, field)

    # Allocated DRAM transfers must reproduce layer-level transfer counts.
    for field, expected in (
        ("dram_input_bw", dram_totals["input"]),
        ("dram_weight_bw", dram_totals["weight"]),
        ("dram_output_bw", dram_totals["output"]),
    ):
        actual = sum(
            getattr(item, field) * item.reference_cycles
            for item in results
        )
        conserved(actual, expected, field)

    return tuple(results)
