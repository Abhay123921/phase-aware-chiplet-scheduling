"""Estimated resource profiles for compact convolution execution groups.

Layer-level Timeloop measurements are allocated to compact groups using
logical access proxies. Group-level values are estimates, not measurements.
"""

from dataclasses import dataclass, asdict
from math import prod


RESOURCE_COLUMNS = {
    "input_bw": "inputs_shared_bw",
    "weight_bw": "weights_shared_bw",
    "output_bw": "outputs_shared_bw",
    "dram_input_bw": "dram_inputs_shared_bw",
    "dram_weight_bw": "dram_weights_shared_bw",
    "dram_output_bw": "dram_outputs_shared_bw",
}


@dataclass(frozen=True)
class CompactGroupResourceProfile:
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
    estimation_method: str = "layer-bandwidth allocation using logical access proxies"

    def to_dict(self):
        return asdict(self)


def estimate_compact_group_resources(group_set, layer_profile):
    """Allocate layer-level work and bandwidth to compact groups.

    Compute work and reference cycles are distributed by logical-slot share.
    Input bandwidth is distributed by valid-contribution share.
    Weight and output bandwidth use per-group logical access proxies.

    For each bandwidth dimension, the time-weighted group average equals
    the original layer-level Timeloop bandwidth.
    """

    if not group_set.groups:
        raise ValueError("Compact group set is empty")

    def metric(name):
        value = float(layer_profile[name])
        if value < 0:
            raise ValueError(f"Negative layer metric: {name}")
        return value

    total_slots = float(group_set.total_logical_slots)
    total_valid = float(group_set.total_valid_contributions)

    if total_slots <= 0 or total_valid <= 0:
        raise ValueError("Invalid total logical or valid work")

    compute_work = metric("reference_compute_work")
    reference_cycles = metric("reference_execution_cycles")

    if compute_work <= 0 or reference_cycles <= 0:
        raise ValueError("Reference compute work and cycles must be positive")

    dimensions = dict(group_set.dimensions)
    reduction_size = prod(dimensions[a] for a in ("C", "R", "S"))

    # Logical proxy: each output channel owns one filter reduction.
    weight_accesses = [
        float(group.output_shape[1] * reduction_size)
        for group in group_set.groups
    ]

    # Logical proxy: each output coordinate is written once.
    output_accesses = [
        float(prod(group.output_shape))
        for group in group_set.groups
    ]

    total_weight_accesses = sum(weight_accesses)
    total_output_accesses = sum(output_accesses)

    if total_weight_accesses <= 0 or total_output_accesses <= 0:
        raise ValueError("Invalid weight/output access proxy")

    layer_bandwidths = {
        key: metric(column)
        for key, column in RESOURCE_COLUMNS.items()
    }

    profiles = []

    for index, group in enumerate(group_set.groups):
        time_share = group.logical_slots / total_slots
        input_share = group.valid_contributions / total_valid
        weight_share = weight_accesses[index] / total_weight_accesses
        output_share = output_accesses[index] / total_output_accesses

        profiles.append(CompactGroupResourceProfile(
            group_id=group.group_id,
            layer_id=group_set.layer_id,
            estimated_compute_work=compute_work * time_share,
            reference_cycles=reference_cycles * time_share,
            estimated_input_accesses=float(group.valid_contributions),
            estimated_weight_accesses=weight_accesses[index],
            estimated_output_accesses=output_accesses[index],
            input_bw=layer_bandwidths["input_bw"] * input_share / time_share,
            weight_bw=layer_bandwidths["weight_bw"] * weight_share / time_share,
            output_bw=layer_bandwidths["output_bw"] * output_share / time_share,
            dram_input_bw=layer_bandwidths["dram_input_bw"] * input_share / time_share,
            dram_weight_bw=layer_bandwidths["dram_weight_bw"] * weight_share / time_share,
            dram_output_bw=layer_bandwidths["dram_output_bw"] * output_share / time_share,
        ))

    # Verify conservation of reference work and time.
    tolerance = 1e-8
    if abs(sum(p.estimated_compute_work for p in profiles) - compute_work) > max(1.0, compute_work * tolerance):
        raise ValueError("Estimated group compute work does not conserve layer total")

    if abs(sum(p.reference_cycles for p in profiles) - reference_cycles) > max(1.0, reference_cycles * tolerance):
        raise ValueError("Estimated group cycles do not conserve layer total")

    # Verify that time-weighted group bandwidth preserves each layer metric.
    for resource, layer_bw in layer_bandwidths.items():
        weighted = sum(
            getattr(profile, resource) * group.logical_slots / total_slots
            for profile, group in zip(profiles, group_set.groups)
        )
        if abs(weighted - layer_bw) > max(1e-8, abs(layer_bw) * tolerance):
            raise ValueError(f"Bandwidth conservation failed for {resource}")

    return tuple(profiles)
