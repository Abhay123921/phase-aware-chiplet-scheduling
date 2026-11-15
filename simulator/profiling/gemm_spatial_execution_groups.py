"""Build mapping-derived candidate execution groups for GEMM workloads.

Groups follow the M x N spatial tile dimensions in a Timeloop mapping.
They represent logical scheduling candidates, not asserted physical
execution units or chiplet placements.
"""

from collections import defaultdict

from simulator.profiling.gemm_ldg_adapter import build_gemm_ldg_from_mapping
from simulator.profiling.execution_group_graph import (
    ExecutionGroup,
    LDGWorkReference,
    build_execution_group_dependency_graph,
)


def build_gemm_spatial_execution_groups(
    *,
    mapping,
    workload_dimensions,
    workload_id,
    operator_graph,
    operator_id,
    grouping_id="gemm_inter_pe_spatial",
    spatial_target="inter_PE_spatial",
):
    """Build GEMM LDGs and group output elements by mapped spatial M x N tiles.

    Returns:
        (execution_group_graph, tuple_of_ldg_graphs)

    Every valid work item from every output-element LDG is assigned exactly
    once. Reduction padding, if present, remains in the LDG but is not
    scheduled.
    """
    if set(workload_dimensions) != {"M", "N", "K"}:
        raise ValueError("GEMM dimensions must contain exactly M, N, and K")

    mapping.validate_workload_dimensions(workload_dimensions)
    operator_graph.validate()

    if operator_graph.workload_id != workload_id:
        raise ValueError("Operator graph workload does not match workload_id")

    if operator_id not in operator_graph.operators:
        raise ValueError(f"Unknown GEMM operator: {operator_id}")

    operator = operator_graph.operators[operator_id]
    if operator.operator_type.lower() != "gemm":
        raise ValueError("Selected operator must have operator_type='gemm'")

    if "C" not in operator.output_tensors:
        raise ValueError("GEMM operator must declare output tensor 'C'")

    spatial_nest = mapping.get_nest(spatial_target, "spatial")
    spatial_m = spatial_nest.factors.get("M", 1)
    spatial_n = spatial_nest.factors.get("N", 1)

    for name, factor, extent in (
        ("M", spatial_m, workload_dimensions["M"]),
        ("N", spatial_n, workload_dimensions["N"]),
    ):
        if factor <= 0 or extent % factor != 0:
            raise ValueError(
                f"Spatial {name} factor {factor} must divide workload extent {extent}"
            )

    ldgs = []
    grouped_references = defaultdict(list)

    for m in range(workload_dimensions["M"]):
        for n in range(workload_dimensions["N"]):
            ldg_group_id = f"C_m{m}_n{n}"

            ldg = build_gemm_ldg_from_mapping(
                mapping=mapping,
                workload_dimensions=workload_dimensions,
                workload_id=workload_id,
                operator_id=operator_id,
                group_id=ldg_group_id,
                output_coordinates={"M": m, "N": n},
            )
            ldgs.append(ldg)

            tile_m = m // spatial_m
            tile_n = n // spatial_n
            execution_group_id = f"{operator_id}:tile_m{tile_m}_n{tile_n}"

            valid_work_item_ids = tuple(
                work_item_id
                for work_item_id, item in ldg.work_items.items()
                if item.is_valid
            )

            grouped_references[execution_group_id].append(
                LDGWorkReference(
                    workload_id=workload_id,
                    operator_id=operator_id,
                    ldg_group_id=ldg.group_id,
                    work_item_ids=valid_work_item_ids,
                )
            )

    groups = []
    for execution_group_id, references in sorted(grouped_references.items()):
        tile_m = int(execution_group_id.rsplit("tile_m", 1)[1].split("_n")[0])
        tile_n = int(execution_group_id.rsplit("_n", 1)[1])

        groups.append(
            ExecutionGroup(
                execution_group_id=execution_group_id,
                operator_id=operator_id,
                ldg_references=tuple(references),
                grouping_level="timeloop_inter_pe_spatial_output_tile",
                input_tensors=tuple(operator.input_tensors),
                output_tensors=("C",),
                metadata={
                    "interpretation": "candidate logical scheduling unit",
                    "physical_execution_claim": False,
                    "spatial_target": spatial_target,
                    "spatial_factors": {"M": spatial_m, "N": spatial_n},
                    "spatial_tile_coordinates": {"M": tile_m, "N": tile_n},
                    "output_element_count": len(references),
                    "reduction_extent": workload_dimensions["K"],
                },
            )
        )

    full_coverage_ldgs = tuple(
        (ldg.workload_id, ldg.operator_id, ldg.group_id)
        for ldg in ldgs
    )

    graph = build_execution_group_dependency_graph(
        workload_id=workload_id,
        grouping_id=grouping_id,
        groups=groups,
        full_coverage_ldgs=full_coverage_ldgs,
    )

    graph.validate(operator_graph=operator_graph, ldg_graphs=tuple(ldgs))

    return graph, tuple(ldgs)
