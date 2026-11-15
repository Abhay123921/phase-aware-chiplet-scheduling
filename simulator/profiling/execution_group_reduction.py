"""Reduction-aware completion contracts for candidate execution groups.

This module describes logical contributor ownership and partial-result
readiness. It does not model physical reduction topology, communication,
synchronization latency, placement, or execution timing.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping


@dataclass(frozen=True)
class ReductionParticipant:
    """One execution group's complete logical contribution to a reduction."""

    execution_group_id: str
    work_item_ids: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.execution_group_id, str) or not self.execution_group_id.strip():
            raise ValueError("execution_group_id cannot be empty")

        if not isinstance(self.work_item_ids, tuple) or not self.work_item_ids:
            raise ValueError("A participant must reference at least one work item")

        if any(not isinstance(item, str) or not item.strip()
               for item in self.work_item_ids):
            raise ValueError("Participant work-item IDs cannot be empty")

        if len(set(self.work_item_ids)) != len(self.work_item_ids):
            raise ValueError("Participant work-item IDs must be unique")


@dataclass(frozen=True)
class ExecutionGroupReductionRequirement:
    """Completion contract for one logical output reduction."""

    requirement_id: str
    workload_id: str
    operator_id: str
    ldg_group_id: str
    output_tensor: str
    output_coordinates: Mapping[str, int]
    reduction_axes: tuple[str, ...]
    reduction_operator: str
    expected_contributors: int
    participants: tuple[ReductionParticipant, ...]

    def __post_init__(self):
        for name in (
            "requirement_id",
            "workload_id",
            "operator_id",
            "ldg_group_id",
            "output_tensor",
            "reduction_operator",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} cannot be empty")

        if not isinstance(self.output_coordinates, Mapping):
            raise ValueError("output_coordinates must be a mapping")

        # Copy and freeze the mapping so external mutations cannot change
        # the identity of an already-created reduction requirement.
        coordinates = dict(self.output_coordinates)
        object.__setattr__(
            self, "output_coordinates", MappingProxyType(coordinates)
        )

        for axis, coordinate in self.output_coordinates.items():
            if (
                not isinstance(axis, str)
                or not axis.strip()
                or not isinstance(coordinate, int)
                or isinstance(coordinate, bool)
                or coordinate < 0
            ):
                raise ValueError("Invalid output coordinate")

        if not isinstance(self.reduction_axes, tuple) or not self.reduction_axes:
            raise ValueError("reduction_axes must be a non-empty tuple")

        if any(not isinstance(axis, str) or not axis.strip()
               for axis in self.reduction_axes):
            raise ValueError("Reduction axis names cannot be empty")

        if len(set(self.reduction_axes)) != len(self.reduction_axes):
            raise ValueError("reduction_axes must be unique")

        if (
            not isinstance(self.expected_contributors, int)
            or isinstance(self.expected_contributors, bool)
            or self.expected_contributors <= 0
        ):
            raise ValueError("expected_contributors must be positive")

        if not isinstance(self.participants, tuple) or not self.participants:
            raise ValueError("At least one reduction participant is required")

        if any(
            not isinstance(participant, ReductionParticipant)
            for participant in self.participants
        ):
            raise TypeError(
                "participants must contain only ReductionParticipant objects"
            )

        group_ids = [p.execution_group_id for p in self.participants]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("Duplicate execution-group participant")

        assigned_ids = [
            work_id
            for participant in self.participants
            for work_id in participant.work_item_ids
        ]
        if len(assigned_ids) != len(set(assigned_ids)):
            raise ValueError("A work item is assigned to multiple participants")

        if len(assigned_ids) != self.expected_contributors:
            raise ValueError(
                "Participant contributor count does not match "
                "expected_contributors"
            )

    @property
    def participant_group_ids(self) -> tuple[str, ...]:
        return tuple(p.execution_group_id for p in self.participants)

    @property
    def assigned_work_item_ids(self) -> frozenset[str]:
        return frozenset(
            work_id
            for participant in self.participants
            for work_id in participant.work_item_ids
        )

    def validate_against(self, execution_group_graph, ldg_graphs: Iterable) -> bool:
        """Validate the contract against its source execution groups and LDG."""

        if execution_group_graph.workload_id != self.workload_id:
            raise ValueError("Reduction requirement workload does not match graph")

        ldg_identity = (
            self.workload_id,
            self.operator_id,
            self.ldg_group_id,
        )

        matching_ldgs = [
            ldg for ldg in ldg_graphs
            if (
                ldg.workload_id,
                ldg.operator_id,
                ldg.group_id,
            ) == ldg_identity
        ]

        if len(matching_ldgs) != 1:
            raise ValueError(
                "Reduction requirement must resolve to exactly one source LDG"
            )

        ldg = matching_ldgs[0]
        ldg.validate()

        # Every group referencing this LDG must participate in this
        # output-reduction contract. This prevents silently omitting a
        # group even when the declared participants cover all work items.
        graph_references = [
            group_id
            for group_id, group in execution_group_graph.groups.items()
            for reference in group.ldg_references
            if (
                reference.workload_id,
                reference.operator_id,
                reference.ldg_group_id,
            ) == ldg_identity
        ]
        if (
            len(graph_references) != len(set(graph_references))
            or set(graph_references) != set(self.participant_group_ids)
        ):
            raise ValueError(
                "Reduction participants must exactly match all execution "
                "groups referencing the source LDG"
            )

        if ldg.output.output_tensor != self.output_tensor:
            raise ValueError("Reduction output tensor does not match source LDG")

        if ldg.output.output_coordinates != self.output_coordinates:
            raise ValueError("Reduction output coordinates do not match source LDG")

        if tuple(ldg.output.reduction_axes) != self.reduction_axes:
            raise ValueError("Reduction axes do not match source LDG")

        if ldg.output.reduction_operator != self.reduction_operator:
            raise ValueError("Reduction operator does not match source LDG")

        if ldg.output.expected_contributors != self.expected_contributors:
            raise ValueError("Expected contributor count does not match source LDG")

        expected_valid_ids = {
            work_id
            for work_id, item in ldg.work_items.items()
            if item.is_valid
        }

        assigned_ids = set()

        for participant in self.participants:
            group_id = participant.execution_group_id

            if group_id not in execution_group_graph.groups:
                raise ValueError(f"Unknown reduction participant group: {group_id}")

            group = execution_group_graph.groups[group_id]

            if group.operator_id != self.operator_id:
                raise ValueError(
                    f"Reduction participant {group_id} belongs to another operator"
                )

            if self.output_tensor not in group.output_tensors:
                raise ValueError(
                    f"Reduction output is not declared by participant {group_id}"
                )

            matching_references = [
                reference
                for reference in group.ldg_references
                if (
                    reference.workload_id,
                    reference.operator_id,
                    reference.ldg_group_id,
                ) == ldg_identity
            ]

            if len(matching_references) != 1:
                raise ValueError(
                    f"Participant {group_id} must contain exactly one "
                    "reference to the required LDG"
                )

            referenced_ids = set(matching_references[0].work_item_ids)
            participant_ids = set(participant.work_item_ids)

            if referenced_ids != participant_ids:
                raise ValueError(
                    f"Participant {group_id} work items do not match "
                    "its execution-group LDG reference"
                )

            for work_id in participant.work_item_ids:
                if work_id not in ldg.work_items:
                    raise ValueError(f"Unknown reduction work item: {work_id}")

                if not ldg.work_items[work_id].is_valid:
                    raise ValueError(
                        f"Invalid/padding work item in reduction participant: {work_id}"
                    )

                if work_id in assigned_ids:
                    raise ValueError(
                        f"Reduction work item assigned more than once: {work_id}"
                    )

                assigned_ids.add(work_id)

        missing = expected_valid_ids - assigned_ids
        unexpected = assigned_ids - expected_valid_ids

        if missing or unexpected:
            raise ValueError(
                "Reduction participant coverage mismatch: "
                f"{len(missing)} valid contributors missing, "
                f"{len(unexpected)} unexpected contributors"
            )

        if len(assigned_ids) != self.expected_contributors:
            raise ValueError("Reduction contributor total is inconsistent")

        return True


@dataclass(frozen=True)
class ReductionPartialResult:
    """Logical completion record for one participant's complete contribution."""

    requirement_id: str
    execution_group_id: str
    work_item_ids: tuple[str, ...]

    def __post_init__(self):
        for name in ("requirement_id", "execution_group_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} cannot be empty")

        if not isinstance(self.work_item_ids, tuple) or not self.work_item_ids:
            raise ValueError("A partial result must contain work-item IDs")

        if any(not isinstance(item, str) or not item.strip()
               for item in self.work_item_ids):
            raise ValueError("Partial-result work-item IDs cannot be empty")

        if len(set(self.work_item_ids)) != len(self.work_item_ids):
            raise ValueError("Partial-result work-item IDs must be unique")


class ReductionReadinessTracker:
    """Track logical partial-result completion for one reduction requirement."""

    def __init__(self, requirement: ExecutionGroupReductionRequirement):
        if not isinstance(requirement, ExecutionGroupReductionRequirement):
            raise TypeError("requirement must be an ExecutionGroupReductionRequirement")

        self.requirement = requirement
        self._expected_by_group = {
            participant.execution_group_id: frozenset(participant.work_item_ids)
            for participant in requirement.participants
        }
        self._completed_by_group = {}

    @property
    def is_ready(self) -> bool:
        return set(self._completed_by_group) == set(self._expected_by_group)

    @property
    def completed_group_count(self) -> int:
        return len(self._completed_by_group)

    @property
    def expected_group_count(self) -> int:
        return len(self._expected_by_group)

    @property
    def completed_work_item_ids(self) -> frozenset[str]:
        return frozenset(
            work_id
            for work_ids in self._completed_by_group.values()
            for work_id in work_ids
        )

    @property
    def missing_group_ids(self) -> tuple[str, ...]:
        return tuple(
            group_id
            for group_id in self._expected_by_group
            if group_id not in self._completed_by_group
        )

    def submit(self, partial_result: ReductionPartialResult) -> bool:
        """Accept one complete participant result; return readiness state."""

        if not isinstance(partial_result, ReductionPartialResult):
            raise TypeError("partial_result must be a ReductionPartialResult")

        if partial_result.requirement_id != self.requirement.requirement_id:
            raise ValueError("Partial result belongs to another requirement")

        group_id = partial_result.execution_group_id

        if group_id not in self._expected_by_group:
            raise ValueError(f"Unexpected reduction participant: {group_id}")

        if group_id in self._completed_by_group:
            raise ValueError(f"Duplicate partial result from participant: {group_id}")

        received = frozenset(partial_result.work_item_ids)
        expected = self._expected_by_group[group_id]

        if received != expected:
            missing = expected - received
            unexpected = received - expected
            raise ValueError(
                f"Incomplete or invalid partial result from {group_id}: "
                f"{len(missing)} missing, {len(unexpected)} unexpected work items"
            )

        self._completed_by_group[group_id] = received
        return self.is_ready
