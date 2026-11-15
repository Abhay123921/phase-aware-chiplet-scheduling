"""Reconstruct logical spatial reduction groups for convolution mappings."""

from math import prod
from pathlib import Path

import yaml

from simulator.profiling.convolution_adapter import ConvolutionProjectionAdapter
from simulator.profiling.iteration_space import SymbolicIterationSpace
from simulator.profiling.mapping_parser import parse_mapping_file
from simulator.profiling.reduction import (
    ReductionContributor,
    ReductionGroup,
)
from simulator.profiling.tlep_schema import MetricProvenance, ProvenanceKind


REDUCTION_AXES = ("C", "R", "S")
WORKLOAD_DIMENSIONS = ("C", "M", "R", "S", "N", "P", "Q", "G")
OUTPUT_AXES = ("N", "M", "Q", "P", "G")


def build_spatial_reduction_groups(
    mapping,
    problem: dict,
    workload_id: str,
    operator_id: str,
) -> tuple[ReductionGroup, ...]:
    """Build reduction groups for spatial lanes at temporal index zero."""

    if not workload_id.strip() or not operator_id.strip():
        raise ValueError("workload_id and operator_id must be non-empty")

    instance = problem["instance"]
    dimensions = {name: instance[name] for name in WORKLOAD_DIMENSIONS}

    space = SymbolicIterationSpace.from_mapping(mapping, dimensions)
    adapter = ConvolutionProjectionAdapter.from_problem(problem)

    spatial_level = space.get_level("spatial")
    expected_reduction_factor = prod(
        spatial_level.factors.get(axis, 1)
        for axis in REDUCTION_AXES
    )

    grouped = {}

    for lane in range(space.spatial_parallelism):
        iteration = space.decode(
            dram_index=0,
            buffer_index=0,
            inter_pe_temporal_index=0,
            spatial_index=lane,
            reg_index=0,
        )

        coordinates = iteration.workload_coordinates
        output_access = adapter.project(coordinates)["Outputs"]

        if not output_access.valid:
            raise ValueError(
                f"Invalid output projection in {workload_id}, "
                f"spatial lane {lane}"
            )

        output_coordinates = dict(
            zip(OUTPUT_AXES, output_access.coordinates)
        )
        output_key = tuple(
            output_coordinates[axis] for axis in OUTPUT_AXES
        )

        contributor = ReductionContributor(
            contributor_id=f"spatial_lane:{lane}",
            logical_coordinates={
                "spatial_lane": lane,
                **output_coordinates,
            },
            reduction_coordinates={
                axis: coordinates[axis] for axis in REDUCTION_AXES
            },
        )

        grouped.setdefault(output_key, []).append(contributor)

    provenance = {
        "spatial_contributors": MetricProvenance(
            kind=ProvenanceKind.RECONSTRUCTED,
            source="Timeloop mapping YAML and parsed-processed-input.yaml",
            method=(
                "Decode spatial lanes, project to logical output "
                "coordinates, and group by C/R/S coordinates"
            ),
            notes=(
                "Reconstruction for temporal indices fixed at zero. "
                "Not a physical reduction-network or timing measurement."
            ),
        )
    }

    groups = []

    for index, (output_key, contributors) in enumerate(
        sorted(grouped.items())
    ):
        group = ReductionGroup(
            group_id=f"{operator_id}:spatial_output:{index}",
            workload_id=workload_id,
            operator_id=operator_id,
            output_tensor="Outputs",
            output_coordinates=dict(zip(OUTPUT_AXES, output_key)),
            reduction_axes=REDUCTION_AXES,
            reduction_operator="sum",
            contributors=tuple(contributors),
            expected_contributors=expected_reduction_factor,
            communication_bytes=None,
            reduction_cycles=None,
            provenance=provenance,
        )

        group.validate_complete()
        groups.append(group)

    actual_contributors = sum(
        group.received_contributors for group in groups
    )

    if actual_contributors != space.spatial_parallelism:
        raise ValueError(
            f"{workload_id}: grouped contributor count "
            "does not match spatial parallelism"
        )

    return tuple(groups)


def build_from_layer_directory(
    layer_directory: Path,
) -> tuple[ReductionGroup, ...]:
    """Load one Timeloop layer directory and reconstruct its groups."""

    mapping = parse_mapping_file(
        layer_directory / "timeloop-mapper.map.yaml"
    )

    problem = yaml.safe_load(
        (layer_directory / "parsed-processed-input.yaml").read_text()
    )["problem"]

    return build_spatial_reduction_groups(
        mapping=mapping,
        problem=problem,
        workload_id="resnet50",
        operator_id=layer_directory.name,
    )


def build_all_resnet50_groups(
    root: Path,
) -> dict[str, tuple[ReductionGroup, ...]]:
    """Build spatial reduction groups for every mapped layer."""

    result = {}

    for directory in sorted(root.iterdir()):
        mapping_path = directory / "timeloop-mapper.map.yaml"
        problem_path = directory / "parsed-processed-input.yaml"

        if mapping_path.exists() and problem_path.exists():
            result[directory.name] = build_from_layer_directory(directory)

    return result
