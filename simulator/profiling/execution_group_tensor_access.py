"""Reconstruct logical tensor accesses for execution groups.

This module reconstructs coordinate-level logical accesses. It does not
claim physical cache hits, DRAM transactions, bandwidth, or transfer timing.
"""

from collections import defaultdict

from simulator.profiling.tlep_schema import (
    MemoryFootprint,
    MetricProvenance,
    ProvenanceKind,
    TensorAccess,
)


def build_execution_group_tensor_accesses(
    *,
    execution_group_graph,
    operator_graph,
    ldg_graphs,
    convolution_projection_adapters=None,
):
    """Build logical tensor accesses and footprints per execution group.

    Convolution requires a projection adapter keyed by
    (workload_id, operator_id).

    GEMM uses the operand coordinates already stored in LDG metadata.

    Returns:
        dict[group_id, dict]:
            tensor_accesses, memory_footprints, provenance, diagnostics
    """

    convolution_projection_adapters = (
        convolution_projection_adapters or {}
    )
    ldg_graphs = tuple(ldg_graphs)

    execution_group_graph.validate(
        operator_graph=operator_graph,
        ldg_graphs=ldg_graphs,
    )

    ldg_map = {
        (ldg.workload_id, ldg.operator_id, ldg.group_id): ldg
        for ldg in ldg_graphs
    }

    results = {}

    for group_id, group in execution_group_graph.groups.items():
        operator = operator_graph.operators[group.operator_id]

        if len(operator.input_tensors) != 2:
            raise ValueError(
                f"{group_id}: expected exactly two input tensors"
            )

        input_a, input_b = operator.input_tensors
        output_tensor = operator.output_tensors[0]

        read_counts = defaultdict(int)
        unique_coordinates = defaultdict(set)
        source_ldgs = {
            (ref.workload_id, ref.operator_id, ref.ldg_group_id):
                ldg_map[
                    (ref.workload_id, ref.operator_id, ref.ldg_group_id)
                ]
            for ref in group.ldg_references
        }

        diagnostics = {
            "valid_work_items": 0,
            "source_ldg_logical_slots": sum(
                ldg.logical_slot_count for ldg in source_ldgs.values()
            ),
            "source_ldg_invalid_work_items": sum(
                ldg.invalid_work_item_count for ldg in source_ldgs.values()
            ),
            "source_ldg_padding_slots": sum(
                1
                for ldg in source_ldgs.values()
                for item in ldg.work_items.values()
                if not item.is_valid and item.invalid_reason == "padding"
            ),
        }

        for ref in group.ldg_references:
            ldg = ldg_map[
                (ref.workload_id, ref.operator_id, ref.ldg_group_id)
            ]

            for work_item_id in ref.work_item_ids:
                item = ldg.work_items[work_item_id]

                if not item.is_valid:
                    raise ValueError(
                        f"{group_id}: invalid work item {work_item_id}"
                    )

                metadata = item.metadata
                adapter = metadata.get("adapter")

                if item.work_count is None:
                    raise ValueError(
                        f"{group_id}: missing work_count for {work_item_id}"
                    )

                count = int(item.work_count)
                if count <= 0 or count != item.work_count:
                    raise ValueError(
                        f"{group_id}: expected positive integer work_count "
                        f"for {work_item_id}"
                    )

                if adapter == "convolution":
                    adapter_key = (ref.workload_id, ref.operator_id)
                    projection_adapter = (
                        convolution_projection_adapters.get(adapter_key)
                    )

                    if projection_adapter is None:
                        raise ValueError(
                            "Missing convolution projection adapter for "
                            f"{adapter_key}"
                        )

                    coordinates = {
                        **metadata["output_coordinates"],
                        **item.reduction_coordinates,
                    }
                    projected = projection_adapter.project(coordinates)

                    operand_coordinates = {
                        input_a: projected["Inputs"],
                        input_b: projected["Weights"],
                    }
                    output_projection = projected["Outputs"]

                    for tensor_name, access in projected.items():
                        if not access.valid:
                            raise ValueError(
                                f"Valid LDG item {work_item_id} projects to "
                                f"invalid {tensor_name} coordinates: "
                                f"{access.coordinates} ({access.invalid_reason})"
                            )

                elif adapter == "gemm":
                    operand_coordinates = {
                        input_a: metadata["input_a_coordinates"],
                        input_b: metadata["input_b_coordinates"],
                    }
                    output_projection = None

                else:
                    raise ValueError(
                        f"Unsupported adapter {adapter!r} in {work_item_id}"
                    )

                for tensor_name, coordinate_data in operand_coordinates.items():
                    if adapter == "convolution":
                        if not projected[tensor_name].valid:
                            continue
                        coordinate = tuple(coordinate_data.coordinates)
                    else:
                        axis_order = {
                            input_a: ("M", "K"),
                            input_b: ("K", "N"),
                        }[tensor_name]
                        coordinate = tuple(
                            coordinate_data[key] for key in axis_order
                        )

                    read_counts[tensor_name] += count
                    unique_coordinates[tensor_name].add(coordinate)

                if adapter == "convolution":
                    output_coordinate = tuple(
                        output_projection.coordinates
                    )
                else:
                    output_coordinate = tuple(
                        metadata["output_coordinates"][key]
                        for key in sorted(metadata["output_coordinates"])
                    )

                unique_coordinates[output_tensor].add(output_coordinate)
                diagnostics["valid_work_items"] += 1

        tensor_accesses = []
        memory_footprints = []

        for tensor_name, role in (
            (input_a, "input"),
            (input_b, "weight"),
            (output_tensor, "reduction_target"),
        ):
            unique_count = len(unique_coordinates[tensor_name])

            if role == "reduction_target":
                # Reduction implementation is not specified by the LDG.
                read_elements = None
                write_elements = None
            else:
                read_elements = read_counts[tensor_name]
                write_elements = 0

            tensor_accesses.append(
                TensorAccess(
                    tensor_name=tensor_name,
                    role=role,
                    read_elements=read_elements,
                    write_elements=write_elements,
                    unique_elements=unique_count,
                    element_size_bytes=None,
                    read_bytes=None,
                    write_bytes=None,
                    memory_level=None,
                )
            )

            memory_footprints.append(
                MemoryFootprint(
                    tensor_name=tensor_name,
                    memory_level="logical_tensor",
                    elements=unique_count,
                    bytes=None,
                    capacity_unit="elements",
                )
            )

        provenance = {
            "tensor_accesses": MetricProvenance(
                kind=ProvenanceKind.RECONSTRUCTED,
                source="LDG work-item coordinates and operator projection",
                method=(
                    "Count operand coordinates per valid MAC contribution; "
                    "deduplicate coordinates for unique-element counts"
                ),
                notes=(
                    "Logical accesses only. Does not represent physical "
                    "memory transactions, cache behavior, or bandwidth."
                ),
            ),
            "memory_footprints": MetricProvenance(
                kind=ProvenanceKind.RECONSTRUCTED,
                source="LDG tensor coordinates",
                method="Count distinct logical coordinates per tensor",
                notes=(
                    "Logical tensor footprint in elements, not measured "
                    "SRAM occupancy or DRAM capacity."
                ),
            ),
        }

        results[group_id] = {
            "tensor_accesses": tensor_accesses,
            "memory_footprints": memory_footprints,
            "provenance": provenance,
            "diagnostics": diagnostics,
        }

    return results
