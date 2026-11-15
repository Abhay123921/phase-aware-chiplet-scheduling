"""GEMM adapter for the operator-agnostic Logical Dependency Graph."""

from simulator.profiling.logical_dependency_graph import (
    LogicalWorkItem,
    build_logical_dependency_graph,
)
from simulator.profiling.iteration_space import (
    SymbolicIterationSpace,
    encode_workload_coordinates,
)


HIERARCHY_LEVELS = (
    "dram",
    "buffer",
    "inter_pe_temporal",
    "spatial",
    "reg",
)


def build_gemm_ldg(
    *,
    workload_id,
    operator_id,
    group_id,
    M,
    N,
    K,
    output_coordinates,
    hierarchy_indices_by_k,
    padded_K=None,
):
    """Build an LDG for one GEMM output element C[m, n].

    Each valid work item represents A[m, k] * B[k, n].
    Positions K <= k < padded_K are retained as invalid padding slots.

    Hierarchy indices are caller-supplied logical groupings. They do not
    imply physical execution order or hardware placement.
    """

    dimensions = {"M": M, "N": N, "K": K}
    for name, extent in dimensions.items():
        if not isinstance(extent, int) or isinstance(extent, bool) or extent <= 0:
            raise ValueError(f"{name} must be a positive integer")

    if padded_K is None:
        padded_K = K

    if (
        not isinstance(padded_K, int)
        or isinstance(padded_K, bool)
        or padded_K < K
    ):
        raise ValueError("padded_K must be an integer greater than or equal to K")

    if set(output_coordinates) != {"M", "N"}:
        raise ValueError("Output coordinates must contain exactly M and N")

    m = output_coordinates["M"]
    n = output_coordinates["N"]

    if not isinstance(m, int) or isinstance(m, bool) or not 0 <= m < M:
        raise ValueError("M output coordinate is out of bounds")
    if not isinstance(n, int) or isinstance(n, bool) or not 0 <= n < N:
        raise ValueError("N output coordinate is out of bounds")

    if set(hierarchy_indices_by_k) != set(range(padded_K)):
        raise ValueError(
            "Hierarchy mapping must provide exactly one entry for every "
            "logical and padded k position"
        )

    work_items = []

    for k in range(padded_K):
        hierarchy_indices = hierarchy_indices_by_k[k]

        if set(hierarchy_indices) != set(HIERARCHY_LEVELS):
            raise ValueError(
                f"Hierarchy indices for k={k} must contain {HIERARCHY_LEVELS}"
            )

        normalized_indices = {}
        for level in HIERARCHY_LEVELS:
            value = hierarchy_indices[level]
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(
                    f"Invalid hierarchy index for k={k}, level={level}"
                )
            normalized_indices[level] = value

        is_valid = k < K

        metadata = {
            "adapter": "gemm",
            "output_coordinates": {"M": m, "N": n},
            "reduction_extent": K,
            "padded_reduction_extent": padded_K,
            "is_padding": not is_valid,
        }

        if is_valid:
            metadata["input_a_coordinates"] = {"M": m, "K": k}
            metadata["input_b_coordinates"] = {"K": k, "N": n}

        work_items.append(
            LogicalWorkItem(
                work_item_id=f"{operator_id}:m{m}:n{n}:k{k}",
                hierarchy_indices=normalized_indices,
                reduction_coordinates={"K": k},
                is_valid=is_valid,
                work_count=1 if is_valid else None,
                invalid_reason=None if is_valid else "reduction_padding",
                metadata=metadata,
            )
        )

    graph = build_logical_dependency_graph(
        workload_id=workload_id,
        operator_id=operator_id,
        group_id=group_id,
        output_tensor="C",
        output_coordinates={"M": m, "N": n},
        reduction_axes=("K",),
        reduction_operator="sum",
        hierarchy_levels=HIERARCHY_LEVELS,
        work_items=work_items,
    )

    if graph.logical_slot_count != padded_K:
        raise ValueError("GEMM logical slot count mismatch")

    if graph.valid_work_item_count != K:
        raise ValueError("GEMM valid contributor count mismatch")

    if graph.invalid_work_item_count != padded_K - K:
        raise ValueError("GEMM padding slot count mismatch")

    return graph

def build_gemm_ldg_from_mapping(
    *,
    mapping,
    workload_dimensions,
    workload_id,
    operator_id,
    group_id,
    output_coordinates,
    logical_K=None,
):
    """Build a GEMM LDG using hierarchy indices derived from a Timeloop mapping.

    workload_dimensions describes the mapped GEMM iteration space and must
    contain exactly M, N, and K.

    logical_K may be smaller than the mapped K extent. In that case, the
    additional mapped positions are retained as logical padding slots.

    This reconstructs mapping-derived logical coordinates. It does not
    establish physical execution order, placement, timing, or masking.
    """
    if not isinstance(workload_dimensions, dict):
        raise TypeError("workload_dimensions must be a dictionary")

    if set(workload_dimensions) != {"M", "N", "K"}:
        raise ValueError(
            "GEMM workload dimensions must contain exactly M, N, and K"
        )

    dimensions = {}
    for name, extent in workload_dimensions.items():
        if (
            not isinstance(extent, int)
            or isinstance(extent, bool)
            or extent <= 0
        ):
            raise ValueError(
                f"Mapped GEMM extent {name} must be a positive integer"
            )
        dimensions[name] = extent

    mapped_K = dimensions["K"]

    if logical_K is None:
        logical_K = mapped_K

    if (
        not isinstance(logical_K, int)
        or isinstance(logical_K, bool)
        or not 0 < logical_K <= mapped_K
    ):
        raise ValueError(
            "logical_K must be a positive integer no greater than mapped K"
        )

    if (
        not isinstance(output_coordinates, dict)
        or set(output_coordinates) != {"M", "N"}
    ):
        raise ValueError(
            "GEMM output coordinates must contain exactly M and N"
        )

    for name, coordinate in output_coordinates.items():
        if (
            not isinstance(coordinate, int)
            or isinstance(coordinate, bool)
            or not 0 <= coordinate < dimensions[name]
        ):
            raise ValueError(
                f"Output coordinate {name} must be an integer in "
                f"[0, {dimensions[name]})"
            )

    space = SymbolicIterationSpace.from_mapping(
        mapping,
        dimensions,
    )

    hierarchy_indices_by_k = {}

    for k in range(mapped_K):
        coordinates = {
            "M": output_coordinates["M"],
            "N": output_coordinates["N"],
            "K": k,
        }

        hierarchy_indices_by_k[k] = encode_workload_coordinates(
            space,
            coordinates,
        )

    graph = build_gemm_ldg(
        workload_id=workload_id,
        operator_id=operator_id,
        group_id=group_id,
        M=dimensions["M"],
        N=dimensions["N"],
        K=logical_K,
        padded_K=mapped_K,
        output_coordinates=output_coordinates,
        hierarchy_indices_by_k=hierarchy_indices_by_k,
    )

    return graph
