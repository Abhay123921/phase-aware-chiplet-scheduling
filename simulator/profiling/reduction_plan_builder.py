"""Build provenance-aware ReductionPlans from Timeloop mapping files."""

from pathlib import Path

from simulator.profiling.mapping_parser import load_mapping_and_workload
from simulator.profiling.iteration_space import SymbolicIterationSpace
from simulator.profiling.reduction_plan import ReductionPlan
from simulator.profiling.tlep_schema import (
    MetricProvenance,
    ProvenanceKind,
)


REDUCTION_AXES = ("C", "R", "S")
OUTPUT_AXES = ("N", "M", "P", "Q", "G")
ALL_AXES = REDUCTION_AXES + OUTPUT_AXES

HIERARCHY_LEVELS = (
    "dram",
    "buffer",
    "inter_pe_temporal",
    "spatial",
    "reg",
)


def build_reduction_plan(
    mapping_path,
    workload_path,
    workload_id="resnet50",
    operator_id=None,
):
    """Construct one validated hierarchical reduction plan."""

    mapping_path = Path(mapping_path)
    workload_path = Path(workload_path)

    if operator_id is None:
        operator_id = mapping_path.parent.name

    mapping, workload_dimensions = load_mapping_and_workload(
        mapping_path,
        workload_path,
    )

    missing_axes = set(ALL_AXES) - set(workload_dimensions)
    if missing_axes:
        raise ValueError(
            f"Workload is missing required axes: {sorted(missing_axes)}"
        )

    extents = {
        axis: int(workload_dimensions[axis])
        for axis in ALL_AXES
    }

    iteration_space = SymbolicIterationSpace.from_mapping(
        mapping,
        extents,
    )

    level_factors = {
        level_name: {
            axis: int(
                iteration_space.get_level(level_name).factors.get(axis, 1)
            )
            for axis in ALL_AXES
        }
        for level_name in HIERARCHY_LEVELS
    }

    source = (
        f"mapping={mapping_path}; workload={workload_path}"
    )

    provenance = {
        "axis_extents": MetricProvenance(
            kind=ProvenanceKind.RECONSTRUCTED,
            source=str(workload_path),
            method="Read parsed Timeloop workload problem instance",
            confidence=1.0,
            notes="Logical workload extents; not measured execution events.",
        ),
        "level_factors": MetricProvenance(
            kind=ProvenanceKind.RECONSTRUCTED,
            source=str(mapping_path),
            method=(
                "Parse Timeloop mapping nests and reconstruct "
                "canonical hierarchy-level factors"
            ),
            confidence=1.0,
            notes=(
                "Mapping-derived coordinate decomposition; "
                "does not establish physical execution timing."
            ),
        ),
        "reduction_factor_by_level": MetricProvenance(
            kind=ProvenanceKind.RECONSTRUCTED,
            source=source,
            method="Product of C, R, S factors at each hierarchy level",
            confidence=1.0,
        ),
        "output_factor_by_level": MetricProvenance(
            kind=ProvenanceKind.RECONSTRUCTED,
            source=source,
            method="Product of N, M, P, Q, G factors at each level",
            confidence=1.0,
        ),
    }

    return ReductionPlan(
        workload_id=workload_id,
        operator_id=operator_id,
        reduction_axes=REDUCTION_AXES,
        output_axes=OUTPUT_AXES,
        axis_extents=extents,
        level_factors=level_factors,
        provenance=provenance,
    )


def build_all_reduction_plans(
    root,
    workload_id="resnet50",
):
    """Build plans for all valid mapping/workload pairs under root."""

    root = Path(root)
    plans = []

    for directory in sorted(root.iterdir()):
        mapping_path = directory / "timeloop-mapper.map.yaml"
        workload_path = directory / "parsed-processed-input.yaml"

        if not mapping_path.exists() or not workload_path.exists():
            continue

        plan = build_reduction_plan(
            mapping_path=mapping_path,
            workload_path=workload_path,
            workload_id=workload_id,
            operator_id=directory.name,
        )
        plans.append(plan)

    return plans
