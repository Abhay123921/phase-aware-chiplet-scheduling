"""Dependency-aware discrete-event simulation of logical execution groups.

Abstract execution durations and global execution slots are used initially.
This is not a cycle-accurate or physical chiplet/network simulator.
"""

from dataclasses import dataclass, field
import math
from typing import Mapping, Optional

from simulator.profiling.execution_group_reduction import (
    ExecutionGroupReductionRequirement,
    ReductionPartialResult,
    ReductionReadinessTracker,
)


@dataclass(frozen=True)
class WorkloadExecutionSpec:
    """Validated graph bundle and abstract costs for one workload."""

    workload_id: str
    operator_graph: object
    execution_group_graph: object
    ldg_graphs: tuple
    reduction_requirements: tuple
    group_durations: Mapping[str, float]
    arrival_time: float = 0.0
    reduction_consumers: Mapping[str, tuple] = field(default_factory=dict)

    def validate(self):
        if not isinstance(self.workload_id, str) or not self.workload_id.strip():
            raise ValueError("workload_id cannot be empty")

        if self.operator_graph.workload_id != self.workload_id:
            raise ValueError("Operator graph workload mismatch")
        if self.execution_group_graph.workload_id != self.workload_id:
            raise ValueError("Execution-group graph workload mismatch")

        if (
            isinstance(self.arrival_time, bool)
            or not isinstance(self.arrival_time, (int, float))
            or not math.isfinite(self.arrival_time)
            or self.arrival_time < 0
        ):
            raise ValueError("arrival_time must be finite and nonnegative")

        self.execution_group_graph.validate(
            operator_graph=self.operator_graph,
            ldg_graphs=self.ldg_graphs,
        )

        group_ids = set(self.execution_group_graph.groups)
        if set(self.group_durations) != group_ids:
            raise ValueError(
                "group_durations must define every execution group exactly once"
            )

        for group_id, duration in self.group_durations.items():
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration)
                or duration <= 0
            ):
                raise ValueError(
                    "Duration for {} must be finite and positive".format(group_id)
                )

        requirement_ids = set()
        for requirement in self.reduction_requirements:
            if not isinstance(
                requirement, ExecutionGroupReductionRequirement
            ):
                raise TypeError("Invalid reduction requirement type")
            if requirement.requirement_id in requirement_ids:
                raise ValueError("Duplicate reduction requirement ID")
            requirement_ids.add(requirement.requirement_id)
            requirement.validate_against(
                self.execution_group_graph, self.ldg_graphs
            )

        requirements = {
            item.requirement_id: item
            for item in self.reduction_requirements
        }

        for requirement_id, consumers in self.reduction_consumers.items():
            if requirement_id not in requirements:
                raise ValueError("Unknown reduction requirement: {}".format(
                    requirement_id
                ))
            if (
                not isinstance(consumers, tuple)
                or len(consumers) != len(set(consumers))
            ):
                raise ValueError("Reduction consumers must be a unique tuple")

            tensor_id = requirements[requirement_id].output_tensor
            for group_id in consumers:
                if group_id not in group_ids:
                    raise ValueError("Unknown reduction consumer: {}".format(
                        group_id
                    ))
                if tensor_id not in self.execution_group_graph.groups[
                    group_id
                ].input_tensors:
                    raise ValueError(
                        "Reduction consumer does not declare required input tensor"
                    )

        return True


@dataclass(frozen=True)
class SimulationEvent:
    event_type: str
    timestamp: float
    workload_id: str
    group_id: Optional[str] = None
    detail: str = ""


@dataclass(frozen=True)
class SimulationResult:
    makespan: float
    completion_times: Mapping[str, float]
    group_start_times: Mapping[tuple, float]
    group_completion_times: Mapping[tuple, float]
    events: tuple
    reduction_ready_times: Mapping[tuple, float]


class DependencyAwareEventSimulator:
    """Discrete-event engine with deterministic ready-group dispatch."""

    def __init__(self, workloads, execution_slots=1):
        if (
            not isinstance(execution_slots, int)
            or isinstance(execution_slots, bool)
            or execution_slots <= 0
        ):
            raise ValueError("execution_slots must be a positive integer")

        self.workloads = tuple(workloads)
        if not self.workloads:
            raise ValueError("At least one workload is required")

        for spec in self.workloads:
            if not isinstance(spec, WorkloadExecutionSpec):
                raise TypeError("Expected WorkloadExecutionSpec")
            spec.validate()

        workload_ids = [spec.workload_id for spec in self.workloads]
        if len(workload_ids) != len(set(workload_ids)):
            raise ValueError("Workload IDs must be unique")

        self.execution_slots = execution_slots

    def run(self):
        specs = {spec.workload_id: spec for spec in self.workloads}

        # Namespace group identities across workloads.
        states = {
            (spec.workload_id, group_id): "pending"
            for spec in self.workloads
            for group_id in spec.execution_group_graph.groups
        }

        predecessors = {}
        reduction_gates = {}
        trackers = {}
        participant_by_group = {}

        for spec in self.workloads:
            wid = spec.workload_id

            for group_id in spec.execution_group_graph.groups:
                predecessors[(wid, group_id)] = {
                    edge.source_group_id
                    for edge in spec.execution_group_graph.dependencies
                    if edge.target_group_id == group_id
                }
                reduction_gates[(wid, group_id)] = {
                    requirement_id
                    for requirement_id, consumers
                    in spec.reduction_consumers.items()
                    if group_id in consumers
                }

            for requirement in spec.reduction_requirements:
                tracker_key = (wid, requirement.requirement_id)
                trackers[tracker_key] = ReductionReadinessTracker(requirement)

                for participant in requirement.participants:
                    key = (wid, participant.execution_group_id)
                    participant_by_group.setdefault(key, []).append(
                        (requirement, participant)
                    )

        events = []
        starts = {}
        completions = {}
        completion_times = {}
        reduction_ready_times = {}
        running = {}
        arrived = set()

        total_groups = len(states)
        now = min(spec.arrival_time for spec in self.workloads)

        def emit(event_type, key, detail=""):
            events.append(
                SimulationEvent(event_type, now, key[0], key[1], detail)
            )

        while len(completions) < total_groups:

            # 1. Register workload arrivals.
            for spec in self.workloads:
                if (
                    spec.workload_id not in arrived
                    and spec.arrival_time <= now
                ):
                    arrived.add(spec.workload_id)
                    events.append(
                        SimulationEvent("ARRIVAL", now, spec.workload_id)
                    )

            # 2. Process all completions at this timestamp.
            finished = sorted(
                key for key, end_time in running.items()
                if end_time <= now
            )

            for key in finished:
                del running[key]
                wid, group_id = key
                states[key] = "completed"
                completions[key] = now
                emit("GROUP_COMPLETE", key)

                # Submit completed logical partial results.
                for requirement, participant in participant_by_group.get(
                    key, []
                ):
                    tracker_key = (wid, requirement.requirement_id)
                    ready = trackers[tracker_key].submit(
                        ReductionPartialResult(
                            requirement_id=requirement.requirement_id,
                            execution_group_id=group_id,
                            work_item_ids=participant.work_item_ids,
                        )
                    )

                    if ready:
                        reduction_ready_times[tracker_key] = now
                        emit(
                            "REDUCTION_READY",
                            key,
                            "requirement={}".format(
                                requirement.requirement_id
                            ),
                        )

                spec = specs[wid]
                if all(
                    states[(wid, candidate)] == "completed"
                    for candidate in spec.execution_group_graph.groups
                ):
                    completion_times[wid] = now
                    emit("WORKLOAD_COMPLETE", key)

            # 3. Find groups whose dependencies and reduction gates are ready.
            candidates = []

            for key, status in states.items():
                if status != "pending":
                    continue

                wid, group_id = key
                spec = specs[wid]

                if wid not in arrived:
                    continue

                if not all(
                    states[(wid, predecessor)] == "completed"
                    for predecessor in predecessors[key]
                ):
                    continue

                if not all(
                    trackers[(wid, requirement_id)].is_ready
                    for requirement_id in reduction_gates[key]
                ):
                    continue

                candidates.append((spec.arrival_time, wid, group_id))

            # Deterministic tie-breaking; scheduling policies come later.
            candidates.sort()

            # 4. Dispatch into available abstract execution slots.
            while candidates and len(running) < self.execution_slots:
                _, wid, group_id = candidates.pop(0)
                key = (wid, group_id)

                if states[key] != "pending":
                    continue

                states[key] = "running"
                starts[key] = now
                running[key] = (
                    now + float(specs[wid].group_durations[group_id])
                )
                emit("GROUP_START", key)

            if len(completions) == total_groups:
                break

            # 5. Advance to the next completion or workload arrival.
            future_times = list(running.values())
            future_times.extend(
                spec.arrival_time
                for spec in self.workloads
                if spec.workload_id not in arrived
            )

            if not future_times:
                blocked = [
                    key for key, state in states.items()
                    if state != "completed"
                ]
                raise RuntimeError(
                    "Simulation deadlock: blocked groups={}".format(blocked)
                )

            next_time = min(future_times)
            if next_time <= now:
                raise RuntimeError("Simulation failed to advance time")

            now = next_time

        return SimulationResult(
            makespan=max(completion_times.values()),
            completion_times=dict(completion_times),
            group_start_times=dict(starts),
            group_completion_times=dict(completions),
            events=tuple(events),
            reduction_ready_times=dict(reduction_ready_times),
        )
