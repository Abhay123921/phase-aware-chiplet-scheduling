"""Event-driven scheduler for compact execution groups.

This is an abstract resource-aware scheduling model. Group bandwidth values
are estimates allocated from layer-level Timeloop measurements.
"""

from dataclasses import dataclass
import heapq
import math
from itertools import product as cartesian_product

from simulator.hardware.model import load_hardware
from simulator.scheduler.locality_policy import LocalityHistory


LOCAL = ("input_bw", "weight_bw", "output_bw")
DRAM = ("dram_input_bw", "dram_weight_bw", "dram_output_bw")

REMAP_COST_FRACTION = 0.05


@dataclass
class CompactScheduleResult:
    policy: str
    makespan: float
    completed_groups: int
    remaps: int
    chiplet_busy_time: dict
    chiplet_group_counts: dict
    chiplet_utilization: dict
    average_group_duration: float
    workload_chiplet_counts: dict
    layer_placement_counts: dict = None


def _duration(
    profile,
    chiplet,
    scales,
    active_dram_pressure,
    resource_aggregation="max",
):
    """Estimate abstract group duration under a selectable pressure model.

    max: bottleneck pressure across resource dimensions.
    mean: average normalized pressure across resource dimensions.
    """
    cycles = float(profile.reference_cycles)

    if resource_aggregation == "max":
        local_value = max(
            getattr(profile, key) / scales[key] for key in LOCAL
        )
        dram_pressure = max(
            getattr(profile, key) / scales[key] for key in DRAM
        )
    elif resource_aggregation == "mean":
        local_value = sum(
            getattr(profile, key) / scales[key] for key in LOCAL
        ) / len(LOCAL)
        dram_pressure = sum(
            getattr(profile, key) / scales[key] for key in DRAM
        ) / len(DRAM)
    else:
        raise ValueError(
            "resource_aggregation must be 'max' or 'mean'"
        )

    local_pressure = local_value / chiplet.memory_bandwidth
    compute = cycles / chiplet.compute_capacity
    memory = cycles * local_pressure
    dram = cycles * dram_pressure * (
        1.0 + 0.25 * active_dram_pressure
    )

    return max(compute, memory, dram)


def _classify_resource_phase(
    profile,
    scales,
    active_dram_pressure,
    resource_aggregation="max",
    mixed_threshold=0.10,
):
    """Classify estimated dominant resource on a balanced C4 reference."""
    if resource_aggregation == "max":
        local_pressure = max(
            getattr(profile, key) / scales[key] for key in LOCAL
        )
        dram_pressure = max(
            getattr(profile, key) / scales[key] for key in DRAM
        )
    elif resource_aggregation == "mean":
        local_pressure = sum(
            getattr(profile, key) / scales[key] for key in LOCAL
        ) / len(LOCAL)
        dram_pressure = sum(
            getattr(profile, key) / scales[key] for key in DRAM
        ) / len(DRAM)
    else:
        raise ValueError("resource_aggregation must be max or mean")

    if mixed_threshold < 0:
        raise ValueError("mixed_threshold must be non-negative")

    components = {
        "compute": 1.0,
        "local_memory": local_pressure,
        "dram": dram_pressure * (
            1.0 + 0.25 * active_dram_pressure
        ),
    }
    ranked = sorted(components.items(), key=lambda item: item[1], reverse=True)
    largest = ranked[0][1]
    second_largest = ranked[1][1]
    if largest <= 0 or largest <= second_largest * (1.0 + mixed_threshold):
        return "mixed"
    return ranked[0][0]


def simulate_compact_groups(
    workload_layers,
    policy,
    hardware=None,
    static_chiplet="C1",
    remap_fraction=REMAP_COST_FRACTION,
    lookahead_groups=4,
    resource_aggregation="max",
    calibration_scales=None,
    allow_legacy_profile_max=False,
    locality_features=None,
    locality_feature_provider=None,
    locality_window_size=16,
    locality_decay=0.8,
    locality_tolerance=0.05,
):
    """Simulate concurrent workloads with layer-level completion barriers.

    workload_layers:
        {workload_name: [(group_set, group_profiles), ...]}

    Each compact group is an independent scheduling unit within its layer.
    A workload's next layer becomes ready after all groups in its current
    layer complete.
    """

    if policy not in (
        "static", "static_rr", "reactive",
        "causal_phase_aware", "phase_aware", "locality_aware",
    ):
        raise ValueError(f"Unknown policy: {policy}")
    if lookahead_groups < 1:
        raise ValueError("lookahead_groups must be at least 1")
    if resource_aggregation not in ("max", "mean"):
        raise ValueError("resource_aggregation must be max or mean")
    if policy == "locality_aware":
        if locality_features is None and locality_feature_provider is None:
            raise ValueError(
                "locality_features are required unless "
                "locality_feature_provider is supplied"
            )
        if locality_features is not None and locality_feature_provider is not None:
            raise ValueError(
                "Provide either locality_features or "
                "locality_feature_provider, not both"
            )
        if locality_feature_provider is not None and not callable(
            locality_feature_provider
        ):
            raise TypeError("locality_feature_provider must be callable")
        if not math.isfinite(locality_tolerance) or not 0 <= locality_tolerance <= 1:
            raise ValueError("locality_tolerance must be in [0, 1]")

    hardware = hardware or load_hardware()
    chiplets = hardware.chiplets

    static_targets = (
        list(static_chiplet.values())
        if isinstance(static_chiplet, dict)
        else [static_chiplet]
    )
    if any(target not in chiplets for target in static_targets):
        raise ValueError(f"Unknown static chiplet in: {static_chiplet}")

    if not workload_layers:
        raise ValueError("No workloads supplied")

    # Validate group/profile alignment and prepare per-workload state.
    states = {}
    all_profiles = []

    for name, layers in workload_layers.items():
        if not layers:
            raise ValueError(f"Workload {name} has no layers")

        normalized_layers = []
        for group_set, profiles in layers:
            if len(group_set.groups) != len(profiles):
                raise ValueError(f"Group/profile count mismatch in {name}")
            normalized_layers.append((group_set, tuple(profiles)))
            all_profiles.extend(profiles)

        states[name] = {
            "layers": normalized_layers,
            "layer": 0,
            "next_group": 0,
            "completed_in_layer": 0,
            "previous_layer_dominant": None,
            "layer_placement_counts": {name: 0 for name in chiplets},
            "layer_placement_history": [],
            "transition_remap_charged": False,
            "remaps": 0,
            "completed": 0,
        }

    required_scales = set(LOCAL + DRAM)

    if calibration_scales is None:
        if not allow_legacy_profile_max:
            raise ValueError(
                "Fixed calibration_scales are required. "
                "Set allow_legacy_profile_max=True only to reproduce "
                "legacy experiments using evaluation-wide normalization."
            )

        # Legacy transductive normalization; not valid for causal evaluation.
        scales = {
            key: max(
                float(getattr(profile, key)) for profile in all_profiles
            )
            for key in LOCAL + DRAM
        }
        scales = {
            key: max(value, 1e-12) for key, value in scales.items()
        }
    else:
        missing = required_scales - set(calibration_scales)
        if missing:
            raise ValueError(
                f"Missing calibration scales: {sorted(missing)}"
            )

        scales = {
            key: float(calibration_scales[key])
            for key in LOCAL + DRAM
        }

        invalid = {
            key: value for key, value in scales.items()
            if not math.isfinite(value) or value <= 0
        }
        if invalid:
            raise ValueError(
                f"Calibration scales must be finite and positive: {invalid}"
            )

    if policy == "locality_aware":
        if locality_features is not None:
            if set(locality_features) != set(states):
                raise ValueError(
                    "locality_features workload keys must match workload_layers"
                )
            for workload_name, state in states.items():
                feature_layers = locality_features[workload_name]
                if len(feature_layers) != len(state["layers"]):
                    raise ValueError(
                        f"Locality layer count mismatch in {workload_name}"
                    )
                normalized_features = []
                for layer_index, ((group_set, profiles), layer_features) in enumerate(
                    zip(state["layers"], feature_layers)
                ):
                    if len(layer_features) != len(group_set.groups):
                        raise ValueError(
                            f"Locality group count mismatch in {workload_name}, "
                            f"layer {layer_index}"
                        )
                    layer_features = tuple(layer_features)
                    for group, feature in zip(group_set.groups, layer_features):
                        if str(feature.layer_id) != str(group_set.layer_id):
                            raise ValueError("Locality feature layer_id mismatch")
                        if str(feature.group_id) != str(group.group_id):
                            raise ValueError("Locality feature group_id mismatch")
                    normalized_features.append(layer_features)
                state["locality_features"] = tuple(normalized_features)

            state["locality_feature_provider"] = None
        else:
            for state in states.values():
                state["locality_features"] = None
                state["locality_feature_provider"] = locality_feature_provider

        for state in states.values():
            state["locality_history"] = LocalityHistory(
                chiplets,
                window_size=locality_window_size,
                decay=locality_decay,
            )

    busy_until = {name: 0.0 for name in chiplets}
    busy_time = {name: 0.0 for name in chiplets}
    group_counts = {name: 0 for name in chiplets}
    workload_chiplet_counts = {
        workload_name: {chiplet_name: 0 for chiplet_name in chiplets}
        for workload_name in states
    }
    free = set(chiplets)
    events = []
    event_counter = 0
    now = 0.0
    active_dram_pressure = 0.0
    total_duration = 0.0
    total_groups = sum(
        len(group_set.groups)
        for layers in workload_layers.values()
        for group_set, _ in layers
    )

    def ready_workloads():
        return [
            name for name, state in states.items()
            if state["layer"] < len(state["layers"])
            and state["next_group"] < len(
                state["layers"][state["layer"]][1]
            )
        ]

    def next_profile(name):
        state = states[name]
        layer_index = state["layer"]
        group_index = state["next_group"]
        return state["layers"][layer_index][1][group_index]

    # A provider-backed feature is computed once per pending group and reused
    # while all candidate chiplets are scored for the current dispatch.
    locality_feature_cache = {}

    def get_locality_feature(name):
        state = states[name]
        layer_index = state["layer"]
        group_index = state["next_group"]
        cache_key = (name, layer_index, group_index)

        if cache_key not in locality_feature_cache:
            group_set, _ = state["layers"][layer_index]
            group = group_set.groups[group_index]

            if state["locality_features"] is not None:
                feature = state["locality_features"][layer_index][group_index]
            else:
                feature = state["locality_feature_provider"](
                    name, layer_index, group_index, group_set, group
                )

            if str(feature.layer_id) != str(group_set.layer_id):
                raise ValueError("Locality feature layer_id mismatch")
            if str(feature.group_id) != str(group.group_id):
                raise ValueError("Locality feature group_id mismatch")

            locality_feature_cache[cache_key] = feature

        return locality_feature_cache[cache_key]

    def candidate_score(name, chiplet_name):
        state = states[name]
        profile = next_profile(name)
        chiplet = chiplets[chiplet_name]

        duration = _duration(
            profile, chiplet, scales, active_dram_pressure, resource_aggregation
        )

        remap = (
            remap_fraction * profile.reference_cycles
            if (
                not state["transition_remap_charged"]
                and state["previous_layer_dominant"] is not None
                and state["previous_layer_dominant"] != chiplet_name
            )
            else 0.0
        )

        if policy == "causal_phase_aware":
            phase = _classify_resource_phase(
                profile,
                scales,
                active_dram_pressure,
                resource_aggregation,
            )

            if phase == "compute":
                preferred = {
                    name for name, item in chiplets.items()
                    if item.compute_capacity == max(
                        chiplet.compute_capacity
                        for chiplet in chiplets.values()
                    )
                }
            elif phase == "local_memory":
                preferred = {
                    name for name, item in chiplets.items()
                    if item.memory_bandwidth == max(
                        chiplet.memory_bandwidth
                        for chiplet in chiplets.values()
                    )
                }
            else:
                preferred = set(chiplets)

            # Lexicographic preference: phase affinity first, then
            # the existing duration and remapping estimate.
            affinity_penalty = 0 if chiplet_name in preferred else 1
            return (affinity_penalty, duration + remap)

        if policy != "phase_aware":
            return duration + remap

        # Phase-aware: bounded lookahead within the current layer only.
        # Existing chiplet occupancy is represented by residual busy time.
        loads = {
            cname: max(0.0, busy_until[cname] - now)
            for cname in chiplets
        }

        loads[chiplet_name] += duration + remap

        current_profiles = state["layers"][state["layer"]][1]
        future_start = state["next_group"] + 1
        future_end = min(
            len(current_profiles),
            state["next_group"] + lookahead_groups,
        )
        future_profiles = current_profiles[future_start:future_end]

        if not future_profiles:
            return max(loads.values())

        chiplet_names = tuple(sorted(chiplets))
        best_objective = None

        # Exhaustively evaluate assignments for the small lookahead window.
        # Future groups are independent within this layer. No group-to-group
        # migration penalty is added; layer-transition remapping is separate.
        for assignment in cartesian_product(
            chiplet_names, repeat=len(future_profiles)
        ):
            projected = dict(loads)

            for future_profile, future_chiplet_name in zip(
                future_profiles, assignment
            ):
                future_duration = _duration(
                    future_profile,
                    chiplets[future_chiplet_name],
                    scales,
                    active_dram_pressure,
                    resource_aggregation,
                )
                projected[future_chiplet_name] += future_duration

            objective = (
                max(projected.values()),
                sum(projected.values()),
            )

            if best_objective is None or objective < best_objective:
                best_objective = objective

        return best_objective[0]

    while any(
        state["layer"] < len(state["layers"])
        for state in states.values()
    ) or events:

        # Dispatch onto every currently free chiplet.
        dispatched = True
        while free and dispatched:
            dispatched = False
            ready = ready_workloads()
            if not ready:
                break

            # Select the best workload-chiplet pair globally.
            # This allows reactive and phase-aware policies to influence
            # the actual chiplet placement decision.
            candidates = []

            for chiplet_name in sorted(free):
                for name in ready:
                    if policy in ("static", "static_rr"):
                        if policy == "static_rr":
                            ordered_chiplets = sorted(chiplets)
                            target = ordered_chiplets[
                                states[name]["next_group"] % len(ordered_chiplets)
                            ]
                        else:
                            target = (
                                static_chiplet.get(name)
                                if isinstance(static_chiplet, dict)
                                else static_chiplet
                            )

                        if target != chiplet_name:
                            continue

                    candidates.append((
                        candidate_score(name, chiplet_name),
                        name,
                        chiplet_name,
                    ))

            if not candidates:
                break

            if policy == "locality_aware":
                best_score = min(item[0] for item in candidates)
                threshold = best_score + locality_tolerance * max(
                    abs(best_score), 1e-12
                )
                eligible = [item for item in candidates if item[0] <= threshold]
                _, chosen_workload, chiplet_name = min(
                    eligible,
                    key=lambda item: (
                        -states[item[1]]["locality_history"].score(
                            item[2],
                            get_locality_feature(item[1]),
                        ),
                        item[0], item[1], item[2],
                    ),
                )
            else:
                _, chosen_workload, chiplet_name = min(candidates)

            state = states[chosen_workload]
            group_set, profiles = state["layers"][state["layer"]]
            profile = profiles[state["next_group"]]
            if policy == "locality_aware":
                locality_feature = get_locality_feature(chosen_workload)
                state["locality_history"].record(
                    chiplet_name, locality_feature
                )
                # Evict only the dispatched group; retain other workloads'
                # pending features for reuse during the next candidate scan.
                locality_feature_cache.pop(
                    (chosen_workload, state["layer"], state["next_group"]),
                    None,
                )
            chiplet = chiplets[chiplet_name]

            duration = _duration(
                profile, chiplet, scales, active_dram_pressure, resource_aggregation
            )

            if not state["transition_remap_charged"]:
                previous = state["previous_layer_dominant"]
                if previous is not None and previous != chiplet_name:
                    duration += remap_fraction * profile.reference_cycles
                    state["remaps"] += 1
                state["transition_remap_charged"] = True

            state["layer_placement_counts"][chiplet_name] += 1
            workload_chiplet_counts[chosen_workload][chiplet_name] += 1

            # Existing running groups are not retroactively slowed.
            group_dram_pressure = max(
                getattr(profile, key) / scales[key] for key in DRAM
            )
            active_dram_pressure += group_dram_pressure

            finish = now + duration
            event_counter += 1
            heapq.heappush(
                events,
                (
                    finish,
                    event_counter,
                    chiplet_name,
                    chosen_workload,
                    group_dram_pressure,
                    duration,
                ),
            )

            busy_until[chiplet_name] = finish
            busy_time[chiplet_name] += duration
            group_counts[chiplet_name] += 1
            total_duration += duration
            state["next_group"] += 1
            free.remove(chiplet_name)
            dispatched = True

        if not events:
            if ready_workloads():
                raise RuntimeError("Ready work exists but no event can run")
            break

        # Process all completions at the earliest timestamp together.
        first_event = heapq.heappop(events)
        batch_finish = first_event[0]
        tolerance = 1e-12 * max(1.0, abs(batch_finish))

        completion_batch = [first_event]
        while events and events[0][0] <= batch_finish + tolerance:
            completion_batch.append(heapq.heappop(events))

        now = max(event[0] for event in completion_batch)
        affected_workloads = set()

        # First release every completed chiplet and update all counters.
        for event in completion_batch:
            (
                finish,
                _,
                chiplet_name,
                workload_name,
                dram_pressure,
                duration,
            ) = event

            active_dram_pressure = max(
                0.0, active_dram_pressure - dram_pressure
            )
            free.add(chiplet_name)

            state = states[workload_name]
            state["completed"] += 1
            state["completed_in_layer"] += 1
            affected_workloads.add(workload_name)

        # Only after the whole batch is accounted for, process layer transitions.
        for workload_name in sorted(affected_workloads):
            state = states[workload_name]
            current_profiles = state["layers"][state["layer"]][1]

            if state["completed_in_layer"] == len(current_profiles):
                placements = state["layer_placement_counts"]
                layer_group_set = state["layers"][state["layer"]][0]
                state["layer_placement_history"].append({
                    "layer_id": str(layer_group_set.layer_id),
                    "group_counts": dict(placements),
                    "dominant_chiplet": max(
                        placements, key=lambda name: (placements[name], name)
                    ),
                })
                state["previous_layer_dominant"] = max(
                    placements, key=lambda name: (placements[name], name)
                )
                state["layer"] += 1
                state["next_group"] = 0
                state["completed_in_layer"] = 0
                state["layer_placement_counts"] = {
                    name: 0 for name in chiplets
                }
                state["transition_remap_charged"] = False

    completed = sum(state["completed"] for state in states.values())
    if completed != total_groups:
        raise RuntimeError(
            f"Completed {completed} groups, expected {total_groups}"
        )

    remaps = sum(state["remaps"] for state in states.values())
    utilization = {
        name: (busy_time[name] / now if now > 0 else 0.0)
        for name in chiplets
    }

    return CompactScheduleResult(
        policy=policy,
        makespan=now,
        completed_groups=completed,
        remaps=remaps,
        chiplet_busy_time=busy_time,
        chiplet_group_counts=group_counts,
        chiplet_utilization=utilization,
        average_group_duration=(
            total_duration / completed if completed else 0.0
        ),
        workload_chiplet_counts=workload_chiplet_counts,
        layer_placement_counts={
            name: list(state["layer_placement_history"])
            for name, state in states.items()
        },
    )
