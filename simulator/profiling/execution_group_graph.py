"""Execution-group dependency DAG.

Execution groups are candidate logical scheduling units. They do not imply
physical execution, placement, concurrency, communication cost, or timing.
"""

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional
import json


EXECUTION_GROUP_GRAPH_SCHEMA_VERSION = "1.1"


@dataclass(frozen=True)
class LDGWorkReference:
    """Reference to selected work items in one logical dependency graph."""

    workload_id: str
    operator_id: str
    ldg_group_id: str
    work_item_ids: tuple[str, ...]

    def __post_init__(self):
        for name in ("workload_id", "operator_id", "ldg_group_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} cannot be empty")

        if not self.work_item_ids:
            raise ValueError("At least one work item must be referenced")

        if any(not isinstance(x, str) or not x.strip()
               for x in self.work_item_ids):
            raise ValueError("Work-item IDs cannot be empty")

        if len(set(self.work_item_ids)) != len(self.work_item_ids):
            raise ValueError("Work-item references must be unique")


@dataclass(frozen=True)
class ExecutionGroup:
    """Candidate logical scheduling unit owned by one operator."""

    execution_group_id: str
    operator_id: str
    ldg_references: tuple[LDGWorkReference, ...]
    grouping_level: str
    input_tensors: tuple[str, ...] = ()
    output_tensors: tuple[str, ...] = ()
    profile_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for name in ("execution_group_id", "operator_id", "grouping_level"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} cannot be empty")

        if not self.ldg_references:
            raise ValueError("Execution group must reference at least one LDG")

        if self.profile_id is not None:
            if not isinstance(self.profile_id, str) or not self.profile_id.strip():
                raise ValueError("profile_id cannot be empty")

        for name in ("input_tensors", "output_tensors"):
            values = getattr(self, name)
            if any(not isinstance(x, str) or not x.strip() for x in values):
                raise ValueError(f"{name} cannot contain empty IDs")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must contain unique IDs")

        if len({
            (ref.workload_id, ref.operator_id, ref.ldg_group_id)
            for ref in self.ldg_references
        }) != len(self.ldg_references):
            raise ValueError("Duplicate LDG references in execution group")

        if any(ref.operator_id != self.operator_id
               for ref in self.ldg_references):
            raise ValueError("All LDG references must belong to the group operator")


@dataclass(frozen=True)
class ExecutionGroupDependency:
    """Tensor-specific data dependency between two execution groups."""

    source_group_id: str
    target_group_id: str
    tensor_id: str
    relation: str = "data"
    tensor_region: Optional[dict[str, tuple[int, int]]] = None

    def __post_init__(self):
        for name in (
            "source_group_id", "target_group_id", "tensor_id", "relation"
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} cannot be empty")

        if self.source_group_id == self.target_group_id:
            raise ValueError("An execution group cannot depend on itself")

        if self.relation != "data":
            raise ValueError(f"Unsupported dependency relation: {self.relation}")

        if self.tensor_region is not None:
            if not isinstance(self.tensor_region, dict):
                raise ValueError("tensor_region must be a dictionary")

            for axis, bounds in self.tensor_region.items():
                if not isinstance(axis, str) or not axis.strip():
                    raise ValueError("Tensor region axis cannot be empty")

                if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                    raise ValueError("Tensor region bounds must contain two values")

                start, end = bounds
                if (
                    not isinstance(start, int) or isinstance(start, bool)
                    or not isinstance(end, int) or isinstance(end, bool)
                    or start < 0 or end < start
                ):
                    raise ValueError("Invalid tensor region bounds")


@dataclass
class ExecutionGroupDependencyGraph:
    """DAG of candidate execution groups for one grouping configuration."""

    workload_id: str
    grouping_id: str
    groups: dict[str, ExecutionGroup]
    dependencies: tuple[ExecutionGroupDependency, ...]
    full_coverage_ldgs: tuple[tuple[str, str, str], ...] = ()
    schema_version: str = EXECUTION_GROUP_GRAPH_SCHEMA_VERSION

    def __post_init__(self):
        for name in ("workload_id", "grouping_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} cannot be empty")

        if not isinstance(self.full_coverage_ldgs, tuple):
            raise ValueError("full_coverage_ldgs must be a tuple")
        for identity in self.full_coverage_ldgs:
            if (
                not isinstance(identity, tuple)
                or len(identity) != 3
                or any(not isinstance(part, str) or not part.strip()
                       for part in identity)
            ):
                raise ValueError(
                    "Each full-coverage LDG identity must be a tuple of "
                    "(workload_id, operator_id, group_id)"
                )
        if len(set(self.full_coverage_ldgs)) != len(self.full_coverage_ldgs):
            raise ValueError("full_coverage_ldgs contains duplicate identities")

    @property
    def group_count(self) -> int:
        return len(self.groups)

    @property
    def dependency_count(self) -> int:
        return len(self.dependencies)

    def topological_order(self) -> tuple[str, ...]:
        """Return a valid group order or reject cycles."""

        indegree = {group_id: 0 for group_id in self.groups}
        adjacency = {group_id: [] for group_id in self.groups}

        for edge in self.dependencies:
            adjacency[edge.source_group_id].append(edge.target_group_id)
            indegree[edge.target_group_id] += 1

        ready = [
            group_id for group_id, degree in indegree.items()
            if degree == 0
        ]
        order = []

        while ready:
            current = ready.pop(0)
            order.append(current)

            for target in adjacency[current]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)

        if len(order) != len(self.groups):
            raise ValueError("Execution-group dependency graph contains a cycle")

        return tuple(order)

    def validate(self, *, operator_graph, ldg_graphs: Iterable) -> bool:
        """Validate references and dependencies against source graphs.

        ldg_graphs may contain multiple LDGs for the same operator. Each LDG
        is identified by (workload_id, operator_id, group_id).
        """

        if not self.groups:
            raise ValueError("Execution-group graph cannot be empty")

        operator_graph.validate()

        if operator_graph.workload_id != self.workload_id:
            raise ValueError("Operator graph workload does not match")

        if set(self.groups) != {
            group.execution_group_id for group in self.groups.values()
        }:
            raise ValueError("Execution-group dictionary keys do not match IDs")

        ldg_map = {}
        for ldg in ldg_graphs:
            key = (ldg.workload_id, ldg.operator_id, ldg.group_id)
            if key in ldg_map:
                raise ValueError(f"Duplicate source LDG identity: {key}")
            ldg.validate()
            ldg_map[key] = ldg

        # Each logical work item can be assigned to at most one candidate
        # execution group within this grouping configuration.
        ownership = set()
        assigned_by_ldg = {}

        for group in self.groups.values():
            if group.operator_id not in operator_graph.operators:
                raise ValueError(
                    f"Unknown operator for execution group: {group.operator_id}"
                )

            operator = operator_graph.operators[group.operator_id]

            if not set(group.input_tensors).issubset(set(operator.input_tensors)):
                raise ValueError(
                    f"Execution group {group.execution_group_id} declares "
                    "an input not declared by its operator"
                )

            if not set(group.output_tensors).issubset(set(operator.output_tensors)):
                raise ValueError(
                    f"Execution group {group.execution_group_id} declares "
                    "an output not declared by its operator"
                )

            for ref in group.ldg_references:
                if ref.workload_id != self.workload_id:
                    raise ValueError("LDG workload does not match graph workload")

                key = (ref.workload_id, ref.operator_id, ref.ldg_group_id)
                if key not in ldg_map:
                    raise ValueError(f"Unknown LDG reference: {key}")

                ldg = ldg_map[key]

                if ldg.operator_id != group.operator_id:
                    raise ValueError("LDG operator does not match execution group")

                if ldg.output.output_tensor not in group.output_tensors:
                    raise ValueError(
                        "Referenced LDG output tensor must be declared "
                        "by its execution group"
                    )

                for work_item_id in ref.work_item_ids:
                    if work_item_id not in ldg.work_items:
                        raise ValueError(
                            f"Unknown LDG work item: {work_item_id}"
                        )

                    if not ldg.work_items[work_item_id].is_valid:
                        raise ValueError(
                            f"Invalid/padding work item cannot be scheduled: "
                            f"{work_item_id}"
                        )

                    owner_key = (key, work_item_id)
                    if owner_key in ownership:
                        raise ValueError(
                            f"Work item assigned to multiple groups: {owner_key}"
                        )
                    ownership.add(owner_key)
                    assigned_by_ldg.setdefault(key, set()).add(work_item_id)

        # Full coverage is opt-in per LDG. Partial candidate groupings remain
        # valid unless their configuration explicitly declares full coverage.
        for key in self.full_coverage_ldgs:
            if key not in ldg_map:
                raise ValueError(f"Unknown full-coverage LDG identity: {key}")
            expected_valid = {
                work_item_id
                for work_item_id, item in ldg_map[key].work_items.items()
                if item.is_valid
            }
            assigned = assigned_by_ldg.get(key, set())
            missing = expected_valid - assigned
            unexpected = assigned - expected_valid
            if missing or unexpected:
                raise ValueError(
                    f"Full coverage failed for LDG {key}: "
                    f"{len(missing)} valid work items missing, "
                    f"{len(unexpected)} unexpected work items"
                )

        expected_operator_dependencies = {
            (
                edge.tensor_id,
                edge.producer_operator_id,
                edge.consumer_operator_id,
            )
            for edge in operator_graph.dependencies
        }

        seen_edges = set()

        for edge in self.dependencies:
            if edge.source_group_id not in self.groups:
                raise ValueError("Unknown source execution group")

            if edge.target_group_id not in self.groups:
                raise ValueError("Unknown target execution group")

            edge_key = (
                edge.source_group_id,
                edge.target_group_id,
                edge.tensor_id,
                edge.relation,
            )
            if edge_key in seen_edges:
                raise ValueError("Duplicate execution-group dependency")
            seen_edges.add(edge_key)

            source = self.groups[edge.source_group_id]
            target = self.groups[edge.target_group_id]

            if source.operator_id == target.operator_id:
                raise ValueError(
                    "Intra-operator dependencies are not supported in schema v1"
                )

            if edge.tensor_id not in source.output_tensors:
                raise ValueError("Dependency tensor is not a source-group output")

            if edge.tensor_id not in target.input_tensors:
                raise ValueError("Dependency tensor is not a target-group input")

            operator_edge = (
                edge.tensor_id,
                source.operator_id,
                target.operator_id,
            )
            if operator_edge not in expected_operator_dependencies:
                raise ValueError(
                    "Execution-group dependency has no matching "
                    "operator-level tensor dependency"
                )

        self.topological_order()
        return True

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "workload_id": self.workload_id,
            "grouping_id": self.grouping_id,
            "group_count": self.group_count,
            "dependency_count": self.dependency_count,
            "full_coverage_ldg_count": len(self.full_coverage_ldgs),
            "full_coverage_ldgs": [
                list(identity) for identity in self.full_coverage_ldgs
            ],
            "topological_order": list(self.topological_order()),
            "operators": sorted({
                group.operator_id for group in self.groups.values()
            }),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "workload_id": self.workload_id,
            "grouping_id": self.grouping_id,
            "groups": [
                {
                    "execution_group_id": group.execution_group_id,
                    "operator_id": group.operator_id,
                    "grouping_level": group.grouping_level,
                    "ldg_references": [
                        {
                            "workload_id": ref.workload_id,
                            "operator_id": ref.operator_id,
                            "ldg_group_id": ref.ldg_group_id,
                            "work_item_ids": list(ref.work_item_ids),
                        }
                        for ref in group.ldg_references
                    ],
                    "input_tensors": list(group.input_tensors),
                    "output_tensors": list(group.output_tensors),
                    "profile_id": group.profile_id,
                    "metadata": group.metadata,
                }
                for group in self.groups.values()
            ],
            "full_coverage_ldgs": [
                list(identity) for identity in self.full_coverage_ldgs
            ],
            "dependencies": [
                {
                    "source_group_id": edge.source_group_id,
                    "target_group_id": edge.target_group_id,
                    "tensor_id": edge.tensor_id,
                    "relation": edge.relation,
                    "tensor_region": edge.tensor_region,
                }
                for edge in self.dependencies
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def build_execution_group_dependency_graph(
    *,
    workload_id: str,
    grouping_id: str,
    groups: Iterable[ExecutionGroup],
    dependencies: Iterable[ExecutionGroupDependency] = (),
    full_coverage_ldgs: Iterable[tuple[str, str, str]] = (),
) -> ExecutionGroupDependencyGraph:
    """Build a graph and validate its internal identity and DAG structure."""

    group_map = {}

    for group in groups:
        if group.execution_group_id in group_map:
            raise ValueError(
                f"Duplicate execution-group ID: {group.execution_group_id}"
            )
        group_map[group.execution_group_id] = group

    graph = ExecutionGroupDependencyGraph(
        workload_id=workload_id,
        grouping_id=grouping_id,
        groups=group_map,
        dependencies=tuple(dependencies),
        full_coverage_ldgs=tuple(full_coverage_ldgs),
    )

    if not graph.groups:
        raise ValueError("At least one execution group is required")

    # Validate internal edge endpoints and cycles now. Source-graph validation
    # is performed separately through graph.validate(...).
    for edge in graph.dependencies:
        if edge.source_group_id not in group_map:
            raise ValueError("Unknown source execution group")
        if edge.target_group_id not in group_map:
            raise ValueError("Unknown target execution group")

    graph.topological_order()
    return graph
