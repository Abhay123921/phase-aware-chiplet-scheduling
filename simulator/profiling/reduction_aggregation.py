"""Reconstruct logical reduction aggregation trees from TLEP groups."""

from collections import Counter, defaultdict
from dataclasses import dataclass

from simulator.profiling.reduction_dependency_analyzer import HIERARCHY_LEVELS
from simulator.profiling.convolution_reduction import REDUCTION_AXES


@dataclass
class AggregationNode:
    level: str
    prefix: tuple
    valid_contributors: int = 0
    padding_slots: int = 0

    @property
    def logical_slots(self):
        return self.valid_contributors + self.padding_slots

    @property
    def padding_only(self):
        return self.valid_contributors == 0 and self.padding_slots > 0


@dataclass
class AggregationBoundary:
    parent_level: str
    child_level: str
    parent_count: int
    child_count: int
    fanout_distribution: dict
    min_fanout: int
    max_fanout: int
    mean_fanout: float
    parents_with_valid_children: int
    children_with_valid_contributors: int
    padding_only_children: int
    identity_partition: bool


@dataclass
class ReductionAggregationTree:
    workload_id: str
    operator_id: str
    group_id: str
    logical_reduction_slots: int
    valid_contributors: int
    padding_slots: int
    level_node_counts: dict
    boundaries: dict
    root_count: int

    def validate(self):
        if self.valid_contributors + self.padding_slots != (
            self.logical_reduction_slots
        ):
            raise ValueError("Logical reduction coverage mismatch")

        if self.level_node_counts["dram"] != self.root_count:
            raise ValueError("Root count mismatch")

        for boundary_name, boundary in self.boundaries.items():
            if boundary.child_count < boundary.parent_count:
                if not boundary.identity_partition:
                    raise ValueError(
                        f"Unexpected partition decrease at {boundary_name}"
                    )

            if sum(boundary.fanout_distribution.values()) != (
                boundary.parent_count
            ):
                raise ValueError(
                    f"Fanout distribution does not cover all parents: "
                    f"{boundary_name}"
                )

            weighted_children = sum(
                fanout * parent_count
                for fanout, parent_count
                in boundary.fanout_distribution.items()
            )

            if weighted_children != boundary.child_count:
                raise ValueError(
                    f"Fanout does not reconstruct child count: "
                    f"{boundary_name}"
                )

        return True

    def to_dict(self):
        return {
            "workload_id": self.workload_id,
            "operator_id": self.operator_id,
            "group_id": self.group_id,
            "logical_reduction_slots": self.logical_reduction_slots,
            "valid_contributors": self.valid_contributors,
            "padding_slots": self.padding_slots,
            "level_node_counts": self.level_node_counts,
            "root_count": self.root_count,
            "boundaries": {
                name: {
                    "parent_level": b.parent_level,
                    "child_level": b.child_level,
                    "parent_count": b.parent_count,
                    "child_count": b.child_count,
                    "fanout_distribution": b.fanout_distribution,
                    "min_fanout": b.min_fanout,
                    "max_fanout": b.max_fanout,
                    "mean_fanout": b.mean_fanout,
                    "parents_with_valid_children": (
                        b.parents_with_valid_children
                    ),
                    "children_with_valid_contributors": (
                        b.children_with_valid_contributors
                    ),
                    "padding_only_children": b.padding_only_children,
                    "identity_partition": b.identity_partition,
                }
                for name, b in self.boundaries.items()
            },
        }


def build_reduction_aggregation_tree(padding_aware_group):
    """Build a logical hierarchy tree for one output reduction group.

    Nodes and edges represent logical contributor partitions only.
    They do not represent physical messages, reduction instructions,
    synchronization events, or execution timing.
    """

    reduction_group = padding_aware_group.reduction_group

    nodes = {
        level: {}
        for level in HIERARCHY_LEVELS
    }

    seen_reduction_coordinates = set()

    def add_slot(reduction_coordinates, hierarchy_indices, is_padding):
        reduction_key = tuple(
            reduction_coordinates[axis]
            for axis in REDUCTION_AXES
        )

        if reduction_key in seen_reduction_coordinates:
            raise ValueError(
                f"Duplicate reduction coordinate: {reduction_key}"
            )
        seen_reduction_coordinates.add(reduction_key)

        if set(hierarchy_indices) != set(HIERARCHY_LEVELS):
            raise ValueError("Incomplete hierarchy indices")

        for depth, level in enumerate(HIERARCHY_LEVELS, start=1):
            prefix = tuple(
                hierarchy_indices[name]
                for name in HIERARCHY_LEVELS[:depth]
            )

            if prefix not in nodes[level]:
                nodes[level][prefix] = AggregationNode(
                    level=level,
                    prefix=prefix,
                )

            node = nodes[level][prefix]

            if is_padding:
                node.padding_slots += 1
            else:
                node.valid_contributors += 1

    for contributor in reduction_group.contributors:
        indices = {
            level: contributor.logical_coordinates[f"{level}_index"]
            for level in HIERARCHY_LEVELS
        }

        add_slot(
            contributor.reduction_coordinates,
            indices,
            is_padding=False,
        )

    for slot in padding_aware_group.padding_slots:
        add_slot(
            slot.reduction_coordinates,
            slot.hierarchy_indices,
            is_padding=True,
        )

    expected_slots = padding_aware_group.logical_reduction_slots

    if len(seen_reduction_coordinates) != expected_slots:
        raise ValueError(
            f"Expected {expected_slots} unique logical slots, "
            f"found {len(seen_reduction_coordinates)}"
        )

    for level in HIERARCHY_LEVELS:
        if not nodes[level]:
            raise ValueError(f"No aggregation nodes at level {level}")

    boundaries = {}

    for parent_level, child_level in zip(
        HIERARCHY_LEVELS[:-1],
        HIERARCHY_LEVELS[1:],
    ):
        children_by_parent = defaultdict(list)

        for child_prefix, child in nodes[child_level].items():
            parent_prefix = child_prefix[:-1]

            if parent_prefix not in nodes[parent_level]:
                raise ValueError(
                    f"Missing parent for {child_level}: {child_prefix}"
                )

            children_by_parent[parent_prefix].append(child)

        fanouts = {
            parent_prefix: len(children)
            for parent_prefix, children in children_by_parent.items()
        }

        if set(fanouts) != set(nodes[parent_level]):
            raise ValueError(
                f"Some {parent_level} nodes have no child partitions"
            )

        distribution = Counter(fanouts.values())

        fanout_values = list(fanouts.values())
        child_nodes = list(nodes[child_level].values())

        boundary_name = f"{parent_level}_to_{child_level}"

        boundaries[boundary_name] = AggregationBoundary(
            parent_level=parent_level,
            child_level=child_level,
            parent_count=len(nodes[parent_level]),
            child_count=len(nodes[child_level]),
            fanout_distribution=dict(sorted(distribution.items())),
            min_fanout=min(fanout_values),
            max_fanout=max(fanout_values),
            mean_fanout=sum(fanout_values) / len(fanout_values),
            parents_with_valid_children=sum(
                any(child.valid_contributors > 0 for child in children)
                for children in children_by_parent.values()
            ),
            children_with_valid_contributors=sum(
                child.valid_contributors > 0
                for child in child_nodes
            ),
            padding_only_children=sum(
                child.padding_only for child in child_nodes
            ),
            identity_partition=(
                len(nodes[parent_level]) == len(nodes[child_level])
                and distribution == {1: len(nodes[parent_level])}
            ),
        )

    tree = ReductionAggregationTree(
        workload_id=reduction_group.workload_id,
        operator_id=reduction_group.operator_id,
        group_id=reduction_group.group_id,
        logical_reduction_slots=expected_slots,
        valid_contributors=len(reduction_group.contributors),
        padding_slots=len(padding_aware_group.padding_slots),
        level_node_counts={
            level: len(nodes[level])
            for level in HIERARCHY_LEVELS
        },
        boundaries=boundaries,
        root_count=len(nodes["dram"]),
    )

    tree.validate()
    return tree
