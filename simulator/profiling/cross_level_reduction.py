"""Padding-aware cross-level convolution reduction reconstruction."""

from dataclasses import dataclass, field
from itertools import product
from math import prod

from simulator.profiling.iteration_space import (
    SymbolicIterationSpace,
    encode_level,
    encode_workload_coordinates,
)
from simulator.profiling.convolution_adapter import ConvolutionProjectionAdapter
from simulator.profiling.convolution_reduction import (
    REDUCTION_AXES,
    OUTPUT_AXES,
)
from simulator.profiling.reduction import (
    ReductionContributor,
    ReductionGroup,
)
from simulator.profiling.tlep_schema import (
    MetricProvenance,
    ProvenanceKind,
)


@dataclass(frozen=True)
class PaddingSlot:
    """Logical reduction slot that does not reference a valid input element."""

    reduction_coordinates: dict[str, int]
    hierarchy_indices: dict[str, int]
    projected_input_coordinates: tuple[int, ...]
    invalid_reason: str


@dataclass(frozen=True)
class PaddingAwareReductionGroup:
    """Valid reduction contributors plus explicitly tracked padding slots."""

    reduction_group: ReductionGroup
    logical_reduction_slots: int
    padding_slots: tuple[PaddingSlot, ...]
    provenance: dict[str, MetricProvenance] = field(default_factory=dict)

    @property
    def valid_contributors(self):
        return self.reduction_group.contributors

    @property
    def valid_contributor_count(self):
        return len(self.valid_contributors)

    @property
    def padding_slot_count(self):
        return len(self.padding_slots)

    def validate_complete(self):
        self.reduction_group.validate_complete()

        if (
            self.valid_contributor_count + self.padding_slot_count
            != self.logical_reduction_slots
        ):
            raise ValueError("Valid and padding slots do not cover logical space")

        tuples = [
            tuple(c.reduction_coordinates[a] for a in REDUCTION_AXES)
            for c in self.valid_contributors
        ]
        tuples.extend(
            tuple(slot.reduction_coordinates[a] for a in REDUCTION_AXES)
            for slot in self.padding_slots
        )

        if len(tuples) != len(set(tuples)):
            raise ValueError("Duplicate logical reduction coordinates")

        if len(tuples) != self.logical_reduction_slots:
            raise ValueError("Missing logical reduction coordinates")

        return True



def build_padding_aware_reduction_group(
    mapping,
    problem,
    workload_id,
    operator_id,
    output_coordinates,
):
    """Reconstruct all logical contributors for one convolution output."""

    instance = problem["instance"]
    dimensions = {
        name: int(instance[name])
        for name in ("C", "M", "R", "S", "N", "P", "Q", "G")
    }

    if set(output_coordinates) != set(OUTPUT_AXES):
        raise ValueError(
            f"Expected output axes {OUTPUT_AXES}, "
            f"received {tuple(output_coordinates)}"
        )

    for axis, coordinate in output_coordinates.items():
        if not 0 <= coordinate < dimensions[axis]:
            raise ValueError(f"Output coordinate out of range: {axis}")

    space = SymbolicIterationSpace.from_mapping(mapping, dimensions)
    adapter = ConvolutionProjectionAdapter.from_problem(problem)

    # Confirm that the selected output is a valid projected output.
    base_coordinates = {
        **output_coordinates,
        "C": 0,
        "R": 0,
        "S": 0,
    }
    output_access = adapter.project(base_coordinates)["Outputs"]

    if not output_access.valid:
        raise ValueError("Selected output coordinate is invalid")

    projected_output = dict(
        zip(OUTPUT_AXES, output_access.coordinates)
    )
    if projected_output != output_coordinates:
        raise ValueError(
            f"Output projection mismatch: {projected_output}"
        )

    expected_slots = prod(
        dimensions[axis] for axis in REDUCTION_AXES
    )

    valid_contributors = []
    padding_slots = []
    seen_hierarchy_indices = set()

    for c, r, s in product(
        range(dimensions["C"]),
        range(dimensions["R"]),
        range(dimensions["S"]),
    ):
        coordinates = {
            **output_coordinates,
            "C": c,
            "R": r,
            "S": s,
        }

        indices = encode_workload_coordinates(space, coordinates)

        decoded = space.decode(
            dram_index=indices["dram"],
            buffer_index=indices["buffer"],
            inter_pe_temporal_index=indices["inter_pe_temporal"],
            spatial_index=indices["spatial"],
            reg_index=indices["reg"],
        )

        if decoded.workload_coordinates != coordinates:
            raise ValueError(
                f"Coordinate round-trip failed: {coordinates}"
            )

        hierarchy_key = tuple(
            indices[level.name] for level in space.levels
        )
        if hierarchy_key in seen_hierarchy_indices:
            raise ValueError("Duplicate hierarchy index combination")
        seen_hierarchy_indices.add(hierarchy_key)

        input_access = adapter.project(coordinates)["Inputs"]

        reduction_coordinates = {
            "C": c,
            "R": r,
            "S": s,
        }

        if input_access.valid:
            logical_coordinates = {
                **{
                    f"output_{axis}": value
                    for axis, value in output_coordinates.items()
                },
                **{
                    f"{level}_index": index
                    for level, index in indices.items()
                },
            }

            valid_contributors.append(
                ReductionContributor(
                    contributor_id=(
                        f"{operator_id}:"
                        f"{indices['dram']}:"
                        f"{indices['buffer']}:"
                        f"{indices['spatial']}:"
                        f"{indices['reg']}"
                    ),
                    logical_coordinates=logical_coordinates,
                    reduction_coordinates=reduction_coordinates,
                    partial_work_count=1.0,
                )
            )
        else:
            if input_access.invalid_reason != "padding":
                raise ValueError(
                    "Unexpected invalid input projection: "
                    f"{input_access.invalid_reason}"
                )

            padding_slots.append(
                PaddingSlot(
                    reduction_coordinates=reduction_coordinates,
                    hierarchy_indices=indices,
                    projected_input_coordinates=input_access.coordinates,
                    invalid_reason=input_access.invalid_reason,
                )
            )

    group_provenance = {
        "valid_contributors": MetricProvenance(
            kind=ProvenanceKind.RECONSTRUCTED,
            source="Timeloop mapping YAML and parsed-processed-input.yaml",
            method=(
                "Enumerate reduction coordinates, encode hierarchy indices, "
                "decode for validation, and retain valid input projections"
            ),
            confidence=1.0,
            notes="Logical contributors, not measured execution events.",
        ),
    }

    padding_provenance = MetricProvenance(
        kind=ProvenanceKind.RECONSTRUCTED,
        source="Timeloop workload projections and tensor extents",
        method="Classify input projections outside tensor bounds as padding",
        confidence=1.0,
        notes="Padding slots are not counted as actual input accesses or MACs.",
    )

    reduction_group = ReductionGroup(
        group_id=(
            f"{operator_id}:output:"
            + ",".join(str(output_coordinates[a]) for a in OUTPUT_AXES)
        ),
        workload_id=workload_id,
        operator_id=operator_id,
        output_tensor="Outputs",
        output_coordinates=dict(output_coordinates),
        reduction_axes=REDUCTION_AXES,
        reduction_operator="sum",
        contributors=tuple(valid_contributors),
        expected_contributors=len(valid_contributors),
        communication_bytes=None,
        reduction_cycles=None,
        provenance=group_provenance,
    )

    result = PaddingAwareReductionGroup(
        reduction_group=reduction_group,
        logical_reduction_slots=expected_slots,
        padding_slots=tuple(padding_slots),
        provenance={"padding_slots": padding_provenance},
    )

    result.validate_complete()

    if len(seen_hierarchy_indices) != expected_slots:
        raise ValueError("Hierarchy index coverage is incomplete")

    return result
