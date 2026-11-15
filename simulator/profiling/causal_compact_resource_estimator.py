"""Pre-execution metadata-based resource estimates for compact convolution groups."""

from math import prod

from simulator.profiling.compact_resource_estimator import (
    CompactGroupResourceProfile,
)

TARGET_TO_PROFILE = {
    "input_bw": "inputs_shared_bw",
    "weight_bw": "weights_shared_bw",
    "output_bw": "outputs_shared_bw",
    "dram_input_bw": "dram_inputs_shared_bw",
    "dram_weight_bw": "dram_weights_shared_bw",
    "dram_output_bw": "dram_outputs_shared_bw",
}


def estimate_causal_compact_group_resources(
    group_set,
    calibration_baselines,
    nominal_compute_rate=16.0,
):
    """Estimate group resources using metadata and fixed calibration."""

    if not group_set.groups:
        raise ValueError("Compact group set is empty")

    if nominal_compute_rate <= 0:
        raise ValueError("Nominal compute rate must be positive")

    if not group_set.validate():
        raise ValueError("Compact group set failed validation")

    total_slots = float(group_set.total_logical_slots)
    total_valid = float(group_set.total_valid_contributions)

    if total_slots <= 0 or total_valid <= 0:
        raise ValueError("Invalid logical or valid work")

    dimensions = dict(group_set.dimensions)
    reduction_size = prod(dimensions[a] for a in ("C", "R", "S"))

    weight_accesses = [
        float(group.output_shape[1] * reduction_size)
        for group in group_set.groups
    ]
    output_accesses = [
        float(prod(group.output_shape))
        for group in group_set.groups
    ]

    total_weight = sum(weight_accesses)
    total_output = sum(output_accesses)

    if total_weight <= 0 or total_output <= 0:
        raise ValueError("Invalid weight/output access proxy")

    baselines = {}
    for profile_name, target_name in TARGET_TO_PROFILE.items():
        if target_name not in calibration_baselines:
            raise KeyError(f"Missing calibration baseline: {target_name}")

        value = float(calibration_baselines[target_name])
        if value < 0:
            raise ValueError(f"Negative calibration baseline: {target_name}")

        baselines[profile_name] = value

    profiles = []

    for index, group in enumerate(group_set.groups):
        time_share = group.logical_slots / total_slots
        input_share = group.valid_contributions / total_valid
        weight_share = weight_accesses[index] / total_weight
        output_share = output_accesses[index] / total_output

        estimated_work = float(group.logical_slots)
        estimated_cycles = estimated_work / nominal_compute_rate

        profiles.append(
            CompactGroupResourceProfile(
                group_id=group.group_id,
                layer_id=group_set.layer_id,
                estimated_compute_work=estimated_work,
                reference_cycles=estimated_cycles,
                estimated_input_accesses=float(group.valid_contributions),
                estimated_weight_accesses=weight_accesses[index],
                estimated_output_accesses=output_accesses[index],
                input_bw=baselines["input_bw"] * input_share / time_share,
                weight_bw=baselines["weight_bw"] * weight_share / time_share,
                output_bw=baselines["output_bw"] * output_share / time_share,
                dram_input_bw=baselines["dram_input_bw"] * input_share / time_share,
                dram_weight_bw=baselines["dram_weight_bw"] * weight_share / time_share,
                dram_output_bw=baselines["dram_output_bw"] * output_share / time_share,
                estimation_method=(
                    "pre-execution metadata proxies with V2 calibration "
                    "medians and nominal 16 logical-operations/cycle"
                ),
            )
        )

    estimated_total_work = sum(p.estimated_compute_work for p in profiles)
    if abs(estimated_total_work - total_slots) > max(1.0, total_slots * 1e-8):
        raise ValueError("Estimated compute work does not conserve")

    estimated_total_cycles = sum(p.reference_cycles for p in profiles)
    expected_cycles = total_slots / nominal_compute_rate

    if abs(estimated_total_cycles - expected_cycles) > max(1.0, expected_cycles * 1e-8):
        raise ValueError("Estimated cycles do not conserve")

    return tuple(profiles)
