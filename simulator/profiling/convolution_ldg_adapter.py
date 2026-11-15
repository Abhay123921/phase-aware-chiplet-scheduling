"""Convolution adapter for the operator-agnostic Logical Dependency Graph."""

from simulator.profiling.logical_dependency_graph import (
    LogicalWorkItem,
    build_logical_dependency_graph,
)


HIERARCHY_LEVELS = (
    "dram",
    "buffer",
    "inter_pe_temporal",
    "spatial",
    "reg",
)


def _work_item_id(operator_id, hierarchy_indices):
    """Create a unique identity using every hierarchy level."""
    suffix = ":".join(
        str(hierarchy_indices[level])
        for level in HIERARCHY_LEVELS
    )
    return f"{operator_id}:logical-slot:{suffix}"


def build_convolution_ldg(padding_aware_group):
    """Translate one padding-aware convolution reduction group into LDG."""

    padding_aware_group.validate_complete()
    reduction_group = padding_aware_group.reduction_group

    work_items = []

    # Translate valid logical contributors.
    for contributor in reduction_group.contributors:
        hierarchy_indices = {}

        for level in HIERARCHY_LEVELS:
            key = f"{level}_index"
            if key not in contributor.logical_coordinates:
                raise ValueError(
                    f"Contributor {contributor.contributor_id} "
                    f"is missing hierarchy index {key}"
                )
            hierarchy_indices[level] = int(
                contributor.logical_coordinates[key]
            )

        metadata = {
            "adapter": "convolution",
            "source_contributor_id": contributor.contributor_id,
            "output_coordinates": dict(
                reduction_group.output_coordinates
            ),
        }

        work_items.append(
            LogicalWorkItem(
                work_item_id=_work_item_id(
                    reduction_group.operator_id,
                    hierarchy_indices,
                ),
                hierarchy_indices=hierarchy_indices,
                reduction_coordinates=dict(
                    contributor.reduction_coordinates
                ),
                is_valid=True,
                work_count=contributor.partial_work_count,
                metadata=metadata,
            )
        )

    # Preserve padding slots as invalid logical slots.
    for slot in padding_aware_group.padding_slots:
        hierarchy_indices = {
            level: int(slot.hierarchy_indices[level])
            for level in HIERARCHY_LEVELS
        }

        metadata = {
            "adapter": "convolution",
            "output_coordinates": dict(
                reduction_group.output_coordinates
            ),
            "projected_input_coordinates": list(
                slot.projected_input_coordinates
            ),
        }

        work_items.append(
            LogicalWorkItem(
                work_item_id=_work_item_id(
                    reduction_group.operator_id,
                    hierarchy_indices,
                ),
                hierarchy_indices=hierarchy_indices,
                reduction_coordinates=dict(
                    slot.reduction_coordinates
                ),
                is_valid=False,
                work_count=None,
                invalid_reason=slot.invalid_reason,
                metadata=metadata,
            )
        )

    if len(work_items) != padding_aware_group.logical_reduction_slots:
        raise ValueError("LDG work-item count does not match logical slots")

    graph = build_logical_dependency_graph(
        workload_id=reduction_group.workload_id,
        operator_id=reduction_group.operator_id,
        group_id=reduction_group.group_id,
        output_tensor=reduction_group.output_tensor,
        output_coordinates=dict(reduction_group.output_coordinates),
        reduction_axes=tuple(reduction_group.reduction_axes),
        reduction_operator=reduction_group.reduction_operator,
        hierarchy_levels=HIERARCHY_LEVELS,
        work_items=work_items,
    )

    graph.validate()

    if graph.logical_slot_count != padding_aware_group.logical_reduction_slots:
        raise ValueError("LDG logical slot count mismatch")

    if graph.valid_work_item_count != padding_aware_group.valid_contributor_count:
        raise ValueError("LDG valid contributor count mismatch")

    if graph.invalid_work_item_count != padding_aware_group.padding_slot_count:
        raise ValueError("LDG padding slot count mismatch")

    return graph
