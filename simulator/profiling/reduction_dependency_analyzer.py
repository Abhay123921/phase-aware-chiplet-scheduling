"""Analyze logical reduction partitions across hierarchy levels."""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import mean

from simulator.profiling.convolution_reduction import REDUCTION_AXES
from simulator.profiling.tlep_schema import (
    MetricProvenance,
    ProvenanceKind,
)


HIERARCHY_LEVELS = (
    "dram",
    "buffer",
    "inter_pe_temporal",
    "spatial",
    "reg",
)


@dataclass(frozen=True)
class ReductionDependencyAnalysis:
    workload_id: str
    operator_id: str
    group_id: str

    logical_reduction_slots: int
    valid_contributors: int
    padding_slots: int

    # One summary per hierarchy prefix.
    level_summaries: dict[str, dict]

    full_hierarchy_partition_count: int

    provenance: dict[str, MetricProvenance] = field(default_factory=dict)

    def validate(self):
        if self.valid_contributors + self.padding_slots != (
            self.logical_reduction_slots
        ):
            raise ValueError("Logical reduction coverage mismatch")

        previous_count = 0

        for level in HIERARCHY_LEVELS:
            summary = self.level_summaries[level]
            count = summary["partition_count"]

            if count < previous_count:
                raise ValueError(
                    f"Hierarchy prefix partition count decreased at {level}"
                )

            previous_count = count

            if summary["valid_contributors"] + summary["padding_slots"] < 0:
                raise ValueError("Negative partition count")

        if previous_count != self.full_hierarchy_partition_count:
            raise ValueError("Full hierarchy partition count mismatch")

        return True

    def to_dict(self):
        return {
            "workload_id": self.workload_id,
            "operator_id": self.operator_id,
            "group_id": self.group_id,
            "logical_reduction_slots": self.logical_reduction_slots,
            "valid_contributors": self.valid_contributors,
            "padding_slots": self.padding_slots,
            "level_summaries": self.level_summaries,
            "full_hierarchy_partition_count": (
                self.full_hierarchy_partition_count
            ),
            "provenance": {
                name: {
                    "kind": record.kind.value,
                    "source": record.source,
                    "method": record.method,
                    "confidence": record.confidence,
                    "notes": record.notes,
                }
                for name, record in self.provenance.items()
            },
        }


def _summarize_buckets(valid_buckets, padding_buckets):
    """Summarize contributor occupancy across logical prefix partitions."""

    keys = set(valid_buckets) | set(padding_buckets)

    valid_counts = [
        len(valid_buckets.get(key, ()))
        for key in keys
    ]
    padding_counts = [
        len(padding_buckets.get(key, ()))
        for key in keys
    ]

    occupancy_distribution = Counter(valid_counts)

    return {
        "partition_count": len(keys),
        "valid_contributors": sum(valid_counts),
        "padding_slots": sum(padding_counts),
        "min_valid_contributors": min(valid_counts, default=0),
        "max_valid_contributors": max(valid_counts, default=0),
        "mean_valid_contributors": (
            mean(valid_counts) if valid_counts else 0.0
        ),
        "valid_occupancy_distribution": dict(
            sorted(occupancy_distribution.items())
        ),
        "partitions_with_valid_contributors": sum(
            count > 0 for count in valid_counts
        ),
        "partitions_with_padding": sum(
            padding_counts[index] > 0
            for index in range(len(padding_counts))
        ),
    }


def analyze_reduction_group(padding_aware_group):
    """Analyze one padding-aware reduction group.

    Prefix partitions are logical groupings of contributors. They are
    not claims about physical messages, synchronization or execution events.
    """

    group = padding_aware_group.reduction_group

    valid_buckets = {
        level: defaultdict(list)
        for level in HIERARCHY_LEVELS
    }
    padding_buckets = {
        level: defaultdict(list)
        for level in HIERARCHY_LEVELS
    }

    seen_reduction_coordinates = set()
    seen_full_hierarchy = set()

    def register_slot(reduction_coordinates, hierarchy_indices, is_padding):
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
            raise ValueError("Incomplete hierarchy index information")

        hierarchy_key = tuple(
            hierarchy_indices[level]
            for level in HIERARCHY_LEVELS
        )

        if hierarchy_key in seen_full_hierarchy:
            raise ValueError(
                f"Duplicate full hierarchy ownership: {hierarchy_key}"
            )
        seen_full_hierarchy.add(hierarchy_key)

        for depth, level in enumerate(HIERARCHY_LEVELS, start=1):
            prefix = tuple(
                hierarchy_indices[name]
                for name in HIERARCHY_LEVELS[:depth]
            )

            bucket = (
                padding_buckets[level]
                if is_padding
                else valid_buckets[level]
            )
            bucket[prefix].append(reduction_key)

    for contributor in group.contributors:
        hierarchy_indices = {
            level: contributor.logical_coordinates[f"{level}_index"]
            for level in HIERARCHY_LEVELS
        }

        register_slot(
            contributor.reduction_coordinates,
            hierarchy_indices,
            is_padding=False,
        )

    for slot in padding_aware_group.padding_slots:
        register_slot(
            slot.reduction_coordinates,
            slot.hierarchy_indices,
            is_padding=True,
        )

    if len(seen_reduction_coordinates) != (
        padding_aware_group.logical_reduction_slots
    ):
        raise ValueError("Incomplete or duplicated logical reduction space")

    if len(seen_full_hierarchy) != (
        padding_aware_group.logical_reduction_slots
    ):
        raise ValueError("Incomplete or duplicated hierarchy ownership")

    summaries = {}

    for level in HIERARCHY_LEVELS:
        summaries[level] = _summarize_buckets(
            valid_buckets[level],
            padding_buckets[level],
        )

    analysis = ReductionDependencyAnalysis(
        workload_id=group.workload_id,
        operator_id=group.operator_id,
        group_id=group.group_id,
        logical_reduction_slots=(
            padding_aware_group.logical_reduction_slots
        ),
        valid_contributors=len(group.contributors),
        padding_slots=len(padding_aware_group.padding_slots),
        level_summaries=summaries,
        full_hierarchy_partition_count=len(seen_full_hierarchy),
        provenance={
            "hierarchy_partitioning": MetricProvenance(
                kind=ProvenanceKind.RECONSTRUCTED,
                source=(
                    "PaddingAwareReductionGroup and Timeloop "
                    "mapping hierarchy indices"
                ),
                method=(
                    "Group valid contributors and padding slots by "
                    "successive hierarchy-index prefixes"
                ),
                confidence=1.0,
                notes=(
                    "Logical partitions only; not physical messages, "
                    "reduction events or synchronization operations."
                ),
            ),
        },
    )

    analysis.validate()
    return analysis
