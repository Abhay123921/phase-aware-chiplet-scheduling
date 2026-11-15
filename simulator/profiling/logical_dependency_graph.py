"""Operator-agnostic Logical Dependency Graph (LDG).

The graph represents:
1. Hierarchical refinement of logical partitions.
2. Ownership of logical work items by leaf partitions.
3. Valid logical contributions to an output reduction.

It does not prescribe physical execution order, communication,
synchronization, reduction topology, or timing.
"""

from dataclasses import dataclass, field
from typing import Iterable, Optional
import json


LDG_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class LogicalWorkItem:
    """One logical slot supplied by a workload adapter."""

    work_item_id: str
    hierarchy_indices: dict[str, int]
    reduction_coordinates: dict[str, int]
    is_valid: bool

    work_count: Optional[float] = None
    invalid_reason: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.work_item_id.strip():
            raise ValueError("work_item_id cannot be empty")

        if not self.hierarchy_indices:
            raise ValueError("hierarchy_indices cannot be empty")

        if not self.reduction_coordinates:
            raise ValueError("reduction_coordinates cannot be empty")

        for name, value in self.hierarchy_indices.items():
            if not name.strip() or not isinstance(value, int) or value < 0:
                raise ValueError(f"Invalid hierarchy index: {name}")

        for name, value in self.reduction_coordinates.items():
            if not name.strip() or not isinstance(value, int) or value < 0:
                raise ValueError(f"Invalid reduction coordinate: {name}")

        if self.work_count is not None and self.work_count < 0:
            raise ValueError("work_count cannot be negative")

        if self.is_valid and self.invalid_reason is not None:
            raise ValueError(
                "A valid work item cannot have an invalid_reason"
            )

        if not self.is_valid and not self.invalid_reason:
            raise ValueError(
                "An invalid work item must specify invalid_reason"
            )


@dataclass(frozen=True)
class PartitionNode:
    """One logical partition identified by a hierarchy prefix."""

    partition_id: str
    level: str
    prefix: tuple[int, ...]
    parent_id: Optional[str]

    logical_slots: int
    valid_slots: int
    invalid_slots: int


@dataclass(frozen=True)
class OutputNode:
    """One logical output requiring reduction of valid contributions."""

    output_id: str
    output_tensor: str
    output_coordinates: dict[str, int]
    reduction_axes: tuple[str, ...]
    reduction_operator: str
    expected_contributors: int


@dataclass(frozen=True)
class LDGEdge:
    """A typed logical relationship between graph entities."""

    source_id: str
    target_id: str
    relation: str


@dataclass
class LogicalDependencyGraph:
    """Generic graph for one logical output reduction group."""

    workload_id: str
    operator_id: str
    group_id: str

    hierarchy_levels: tuple[str, ...]
    reduction_axes: tuple[str, ...]

    partitions: dict[str, PartitionNode]
    work_items: dict[str, LogicalWorkItem]
    output: OutputNode
    edges: tuple[LDGEdge, ...]

    schema_version: str = LDG_SCHEMA_VERSION

    @property
    def logical_slot_count(self):
        return len(self.work_items)

    @property
    def valid_work_item_count(self):
        return sum(item.is_valid for item in self.work_items.values())

    @property
    def invalid_work_item_count(self):
        return self.logical_slot_count - self.valid_work_item_count

    def validate(self):
        """Validate complete graph structure and all typed relationships."""

        if not self.hierarchy_levels:
            raise ValueError("Graph must have hierarchy levels")
        if len(set(self.hierarchy_levels)) != len(self.hierarchy_levels):
            raise ValueError("Hierarchy levels must be unique")
        if any(not isinstance(x, str) or not x.strip()
               for x in self.hierarchy_levels):
            raise ValueError("Hierarchy level names cannot be empty")

        if not self.reduction_axes:
            raise ValueError("Graph must have reduction axes")
        if len(set(self.reduction_axes)) != len(self.reduction_axes):
            raise ValueError("Reduction axes must be unique")
        if any(not isinstance(x, str) or not x.strip()
               for x in self.reduction_axes):
            raise ValueError("Reduction axis names cannot be empty")

        if not self.work_items:
            raise ValueError("Graph cannot be empty")

        if not self.output.output_id.strip():
            raise ValueError("Output ID cannot be empty")

        if self.output.expected_contributors != self.valid_work_item_count:
            raise ValueError("Output contributor count mismatch")

        if set(self.work_items) != {
            item.work_item_id for item in self.work_items.values()
        }:
            raise ValueError("Work-item dictionary keys do not match IDs")

        partition_records = {}
        seen_reduction_coordinates = set()
        expected_contains_edges = set()
        expected_contribution_edges = set()

        for work_item in self.work_items.values():
            if set(work_item.hierarchy_indices) != set(
                self.hierarchy_levels
            ):
                raise ValueError(
                    f"Incomplete hierarchy indices: {work_item.work_item_id}"
                )

            if set(work_item.reduction_coordinates) != set(
                self.reduction_axes
            ):
                raise ValueError(
                    f"Reduction axes mismatch: {work_item.work_item_id}"
                )

            reduction_key = tuple(
                work_item.reduction_coordinates[axis]
                for axis in self.reduction_axes
            )

            if reduction_key in seen_reduction_coordinates:
                raise ValueError(
                    f"Duplicate reduction coordinates: {reduction_key}"
                )
            seen_reduction_coordinates.add(reduction_key)

            previous_prefix = ()

            for depth, level in enumerate(self.hierarchy_levels, start=1):
                prefix = tuple(
                    work_item.hierarchy_indices[name]
                    for name in self.hierarchy_levels[:depth]
                )
                partition_id = self._partition_id(depth, prefix)

                parent_id = (
                    self._partition_id(depth - 1, previous_prefix)
                    if depth > 1 else None
                )

                record = partition_records.setdefault(
                    partition_id,
                    {
                        "level": level,
                        "prefix": prefix,
                        "parent_id": parent_id,
                        "counts": [0, 0, 0],
                    },
                )

                if (
                    record["level"] != level
                    or record["prefix"] != prefix
                    or record["parent_id"] != parent_id
                ):
                    raise ValueError(
                        f"Inconsistent partition structure: {partition_id}"
                    )

                record["counts"][0] += 1
                if work_item.is_valid:
                    record["counts"][1] += 1
                else:
                    record["counts"][2] += 1

                previous_prefix = prefix

            leaf_id = self._partition_id(
                len(self.hierarchy_levels),
                previous_prefix,
            )

            expected_contains_edges.add(
                (leaf_id, work_item.work_item_id, "contains")
            )

            if work_item.is_valid:
                expected_contribution_edges.add(
                    (
                        work_item.work_item_id,
                        self.output.output_id,
                        "contributes_to",
                    )
                )

        # The partition dictionary must contain exactly the partitions
        # implied by the work items: neither missing nor additional nodes.
        if set(self.partitions) != set(partition_records):
            raise ValueError(
                "Partition set does not match work-item hierarchy"
            )

        expected_refinement_edges = set()

        for partition_id, record in partition_records.items():
            partition = self.partitions[partition_id]

            if partition.partition_id != partition_id:
                raise ValueError(
                    f"Partition dictionary key mismatch: {partition_id}"
                )

            if (
                partition.level != record["level"]
                or partition.prefix != record["prefix"]
                or partition.parent_id != record["parent_id"]
            ):
                raise ValueError(
                    f"Partition identity or parent mismatch: {partition_id}"
                )

            if (
                partition.logical_slots,
                partition.valid_slots,
                partition.invalid_slots,
            ) != tuple(record["counts"]):
                raise ValueError(
                    f"Partition occupancy mismatch: {partition_id}"
                )

            if record["parent_id"] is not None:
                expected_refinement_edges.add(
                    (
                        record["parent_id"],
                        partition_id,
                        "refines",
                    )
                )

        # Validate edge uniqueness, supported relations and nonempty IDs.
        edge_set = set()
        allowed_relations = {
            "refines",
            "contains",
            "contributes_to",
        }

        for edge in self.edges:
            if edge.relation not in allowed_relations:
                raise ValueError(
                    f"Unsupported graph relation: {edge.relation}"
                )

            if (
                not edge.source_id
                or not edge.target_id
                or not edge.source_id.strip()
                or not edge.target_id.strip()
            ):
                raise ValueError("Graph edge IDs cannot be empty")

            key = (edge.source_id, edge.target_id, edge.relation)

            if key in edge_set:
                raise ValueError("Duplicate graph edges")
            edge_set.add(key)

        # Graph entity identifiers must be globally unambiguous.
        partition_ids = set(self.partitions)
        work_item_ids = set(self.work_items)
        output_ids = {self.output.output_id}

        if partition_ids & work_item_ids:
            raise ValueError("Partition and work-item IDs collide")
        if partition_ids & output_ids or work_item_ids & output_ids:
            raise ValueError("Output ID collides with another graph node")

        actual_refinement_edges = {
            edge for edge in edge_set if edge[2] == "refines"
        }
        actual_contains_edges = {
            edge for edge in edge_set if edge[2] == "contains"
        }
        actual_contribution_edges = {
            edge for edge in edge_set if edge[2] == "contributes_to"
        }

        if actual_refinement_edges != expected_refinement_edges:
            raise ValueError(
                "Refinement edges do not match partition hierarchy"
            )

        if actual_contains_edges != expected_contains_edges:
            raise ValueError(
                "Ownership edges do not match logical work items"
            )

        if actual_contribution_edges != expected_contribution_edges:
            raise ValueError(
                "Contribution edges do not match valid work items"
            )

        return True

    @staticmethod
    def _partition_id(depth, prefix):
        encoded = json.dumps(prefix, separators=(",", ":"))
        return f"partition:{depth}:{encoded}"

    def summary(self):
        """Return compact structural statistics."""

        level_counts = {
            level: sum(
                node.level == level
                for node in self.partitions.values()
            )
            for level in self.hierarchy_levels
        }

        return {
            "schema_version": self.schema_version,
            "workload_id": self.workload_id,
            "operator_id": self.operator_id,
            "group_id": self.group_id,
            "hierarchy_levels": list(self.hierarchy_levels),
            "reduction_axes": list(self.reduction_axes),
            "logical_slots": self.logical_slot_count,
            "valid_work_items": self.valid_work_item_count,
            "invalid_work_items": self.invalid_work_item_count,
            "output_contributors": self.output.expected_contributors,
            "partition_counts": level_counts,
            "edge_counts": {
                relation: sum(
                    edge.relation == relation for edge in self.edges
                )
                for relation in (
                    "refines",
                    "contains",
                    "contributes_to",
                )
            },
        }

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "workload_id": self.workload_id,
            "operator_id": self.operator_id,
            "group_id": self.group_id,
            "hierarchy_levels": list(self.hierarchy_levels),
            "reduction_axes": list(self.reduction_axes),
            "partitions": [
                {
                    "partition_id": node.partition_id,
                    "level": node.level,
                    "prefix": list(node.prefix),
                    "parent_id": node.parent_id,
                    "logical_slots": node.logical_slots,
                    "valid_slots": node.valid_slots,
                    "invalid_slots": node.invalid_slots,
                }
                for node in self.partitions.values()
            ],
            "work_items": [
                {
                    "work_item_id": item.work_item_id,
                    "hierarchy_indices": item.hierarchy_indices,
                    "reduction_coordinates": item.reduction_coordinates,
                    "is_valid": item.is_valid,
                    "work_count": item.work_count,
                    "invalid_reason": item.invalid_reason,
                    "metadata": item.metadata,
                }
                for item in self.work_items.values()
            ],
            "output": {
                "output_id": self.output.output_id,
                "output_tensor": self.output.output_tensor,
                "output_coordinates": self.output.output_coordinates,
                "reduction_axes": list(self.output.reduction_axes),
                "reduction_operator": self.output.reduction_operator,
                "expected_contributors": self.output.expected_contributors,
            },
            "edges": [
                {
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "relation": edge.relation,
                }
                for edge in self.edges
            ],
        }


def build_logical_dependency_graph(
    *,
    workload_id: str,
    operator_id: str,
    group_id: str,
    output_tensor: str,
    output_coordinates: dict[str, int],
    reduction_axes: tuple[str, ...],
    reduction_operator: str,
    hierarchy_levels: tuple[str, ...],
    work_items: Iterable[LogicalWorkItem],
):
    """Build a generic logical dependency graph from adapter-produced items."""

    if not workload_id.strip() or not operator_id.strip():
        raise ValueError("workload_id and operator_id are required")

    if not group_id.strip() or not output_tensor.strip():
        raise ValueError("group_id and output_tensor are required")

    if not reduction_operator.strip():
        raise ValueError("reduction_operator cannot be empty")

    if not hierarchy_levels or len(set(hierarchy_levels)) != len(
        hierarchy_levels
    ):
        raise ValueError("Invalid hierarchy_levels")

    if not reduction_axes or len(set(reduction_axes)) != len(reduction_axes):
        raise ValueError("Invalid reduction_axes")

    if any(not name.strip() for name in hierarchy_levels + reduction_axes):
        raise ValueError("Hierarchy and reduction names cannot be empty")

    for name, value in output_coordinates.items():
        if not name.strip() or not isinstance(value, int) or value < 0:
            raise ValueError(f"Invalid output coordinate: {name}")

    items = list(work_items)

    if not items:
        raise ValueError("At least one logical work item is required")

    item_map = {}

    for item in items:
        if item.work_item_id in item_map:
            raise ValueError(
                f"Duplicate work item ID: {item.work_item_id}"
            )

        if set(item.hierarchy_indices) != set(hierarchy_levels):
            raise ValueError(
                f"Hierarchy schema mismatch: {item.work_item_id}"
            )

        if set(item.reduction_coordinates) != set(reduction_axes):
            raise ValueError(
                f"Reduction schema mismatch: {item.work_item_id}"
            )

        item_map[item.work_item_id] = item

    output_id = f"output:{group_id}"

    valid_count = sum(item.is_valid for item in items)

    output = OutputNode(
        output_id=output_id,
        output_tensor=output_tensor,
        output_coordinates=dict(output_coordinates),
        reduction_axes=tuple(reduction_axes),
        reduction_operator=reduction_operator,
        expected_contributors=valid_count,
    )

    partitions = {}
    edges = []

    for item in items:
        previous_prefix = ()

        for depth, level in enumerate(hierarchy_levels, start=1):
            prefix = tuple(
                item.hierarchy_indices[name]
                for name in hierarchy_levels[:depth]
            )

            partition_id = LogicalDependencyGraph._partition_id(
                depth, prefix
            )

            parent_id = (
                LogicalDependencyGraph._partition_id(
                    depth - 1, previous_prefix
                )
                if depth > 1
                else None
            )

            if partition_id not in partitions:
                partitions[partition_id] = {
                    "level": level,
                    "prefix": prefix,
                    "parent_id": parent_id,
                    "logical_slots": 0,
                    "valid_slots": 0,
                    "invalid_slots": 0,
                }

            record = partitions[partition_id]

            if (
                record["level"] != level
                or record["parent_id"] != parent_id
            ):
                raise ValueError(
                    f"Inconsistent partition parent: {partition_id}"
                )

            record["logical_slots"] += 1

            if item.is_valid:
                record["valid_slots"] += 1
            else:
                record["invalid_slots"] += 1

            if parent_id is not None:
                edges.append(
                    LDGEdge(
                        source_id=parent_id,
                        target_id=partition_id,
                        relation="refines",
                    )
                )

            previous_prefix = prefix

        leaf_id = LogicalDependencyGraph._partition_id(
            len(hierarchy_levels),
            previous_prefix,
        )

        edges.append(
            LDGEdge(
                source_id=leaf_id,
                target_id=item.work_item_id,
                relation="contains",
            )
        )

        if item.is_valid:
            edges.append(
                LDGEdge(
                    source_id=item.work_item_id,
                    target_id=output_id,
                    relation="contributes_to",
                )
            )

    # Deduplicate shared hierarchy refinement relationships.
    # Preserve insertion order and all distinct graph relationships.
    unique_edges = {}
    for edge in edges:
        edge_key = (edge.source_id, edge.target_id, edge.relation)
        unique_edges.setdefault(edge_key, edge)
    edges = list(unique_edges.values())

    partition_nodes = {
        partition_id: PartitionNode(
            partition_id=partition_id,
            **record,
        )
        for partition_id, record in partitions.items()
    }

    graph = LogicalDependencyGraph(
        workload_id=workload_id,
        operator_id=operator_id,
        group_id=group_id,
        hierarchy_levels=tuple(hierarchy_levels),
        reduction_axes=tuple(reduction_axes),
        partitions=partition_nodes,
        work_items=item_map,
        output=output,
        edges=tuple(edges),
    )

    graph.validate()
    return graph
