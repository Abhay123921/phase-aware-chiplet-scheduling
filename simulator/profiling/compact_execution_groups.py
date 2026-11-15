"""Compact mapping-derived convolution execution groups."""

from dataclasses import dataclass
from itertools import product
from math import prod

from simulator.profiling.convolution_adapter import ConvolutionProjectionAdapter


OUTPUT_AXES = ("N", "M", "Q", "P", "G")
REDUCTION_AXES = ("C", "R", "S")


@dataclass(frozen=True)
class CompactExecutionGroup:
    group_id: str
    output_origin: tuple
    output_shape: tuple
    spatial_factors: tuple
    logical_slots: int
    valid_contributions: int
    padding_slots: int


@dataclass(frozen=True)
class CompactExecutionGroupSet:
    layer_id: str
    output_axes: tuple
    dimensions: tuple
    spatial_factors: tuple
    groups: tuple
    total_output_coordinates: int
    total_logical_slots: int
    total_valid_contributions: int
    total_padding_slots: int

    def validate(self):
        if not self.groups:
            raise ValueError("No compact groups generated")

        origins = [g.output_origin for g in self.groups]
        if len(origins) != len(set(origins)):
            raise ValueError("Duplicate output tile origins")

        if sum(prod(g.output_shape) for g in self.groups) != self.total_output_coordinates:
            raise ValueError("Output coverage mismatch")

        if sum(g.logical_slots for g in self.groups) != self.total_logical_slots:
            raise ValueError("Logical-slot conservation failed")

        if sum(g.valid_contributions for g in self.groups) != self.total_valid_contributions:
            raise ValueError("Valid-work conservation failed")

        if sum(g.padding_slots for g in self.groups) != self.total_padding_slots:
            raise ValueError("Padding conservation failed")

        for group in self.groups:
            if group.valid_contributions + group.padding_slots != group.logical_slots:
                raise ValueError(f"Incomplete group: {group.group_id}")

        if self.total_valid_contributions + self.total_padding_slots != self.total_logical_slots:
            raise ValueError("Layer-level conservation failed")

        return True


def build_compact_convolution_groups(mapping, problem, layer_id):
    """Build compact output tiles without materializing individual LDG items."""

    instance = problem["instance"]
    dims = {
        key: int(instance[key])
        for key in ("C", "M", "R", "S", "N", "P", "Q", "G")
    }

    if dims["N"] != 1 or dims["G"] != 1:
        raise NotImplementedError("Current implementation requires N=G=1")

    mapping.validate_workload_dimensions(dims)

    spatial = mapping.get_nest("inter_PE_spatial", "spatial").factors
    factors = {axis: int(spatial.get(axis, 1)) for axis in OUTPUT_AXES}

    for axis in OUTPUT_AXES:
        if dims[axis] % factors[axis]:
            raise ValueError(f"{axis} extent is not divisible by its spatial factor")

    adapter = ConvolutionProjectionAdapter.from_problem(problem)

    # Count valid C,R,S contributors for each P,Q output position.
    valid_by_pq = {}
    for p in range(dims["P"]):
        for q in range(dims["Q"]):
            valid = 0
            for c, r, s in product(
                range(dims["C"]), range(dims["R"]), range(dims["S"])
            ):
                access = adapter.project({
                    "N": 0, "M": 0, "Q": q, "P": p, "G": 0,
                    "C": c, "R": r, "S": s,
                })["Inputs"]

                if access.valid:
                    valid += 1
                elif access.invalid_reason != "padding":
                    raise ValueError(
                        f"Unexpected invalid input: {access.invalid_reason}"
                    )
            valid_by_pq[(p, q)] = valid

    origin_ranges = [
        range(0, dims[axis], factors[axis])
        for axis in OUTPUT_AXES
    ]

    groups = []
    for index, origin in enumerate(product(*origin_ranges)):
        shape = tuple(factors[axis] for axis in OUTPUT_AXES)
        origin_map = dict(zip(OUTPUT_AXES, origin))

        valid_one_channel = sum(
            valid_by_pq[(p, q)]
            for p in range(origin_map["P"], origin_map["P"] + factors["P"])
            for q in range(origin_map["Q"], origin_map["Q"] + factors["Q"])
        )

        output_multiplicity = factors["N"] * factors["M"] * factors["G"]
        valid = valid_one_channel * output_multiplicity
        logical = prod(shape) * prod(dims[a] for a in REDUCTION_AXES)
        padding = logical - valid

        if padding < 0:
            raise ValueError(f"Negative padding count in group {index}")

        groups.append(CompactExecutionGroup(
            group_id=f"{layer_id}:tile:{index}",
            output_origin=tuple(origin),
            output_shape=shape,
            spatial_factors=tuple(factors[a] for a in OUTPUT_AXES),
            logical_slots=logical,
            valid_contributions=valid,
            padding_slots=padding,
        ))

    total_outputs = prod(dims[a] for a in OUTPUT_AXES)
    total_logical = total_outputs * prod(dims[a] for a in REDUCTION_AXES)
    total_valid = sum(g.valid_contributions for g in groups)
    total_padding = sum(g.padding_slots for g in groups)

    result = CompactExecutionGroupSet(
        layer_id=layer_id,
        output_axes=OUTPUT_AXES,
        dimensions=tuple((a, dims[a]) for a in OUTPUT_AXES + REDUCTION_AXES),
        spatial_factors=tuple((a, factors[a]) for a in OUTPUT_AXES),
        groups=tuple(groups),
        total_output_coordinates=total_outputs,
        total_logical_slots=total_logical,
        total_valid_contributions=total_valid,
        total_padding_slots=total_padding,
    )

    result.validate()
    return result
