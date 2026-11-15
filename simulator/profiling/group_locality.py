"""Geometry-aware data-locality features for compact convolution groups.

This module measures exact projected tensor-coordinate footprints.
Geometric overlap is not interpreted as a cache hit or guaranteed DRAM saving.
"""

from dataclasses import dataclass
from itertools import product
from math import prod

from simulator.profiling.convolution_adapter import (
    ConvolutionProjectionAdapter,
)


@dataclass(frozen=True)
class GroupLocalityFeatures:
    layer_id: str
    group_id: str
    element_bytes: int

    input_access_attempts: int
    unique_input_elements: int
    input_reuse_factor: float

    weight_access_attempts: int
    unique_weight_elements: int
    weight_reuse_factor: float

    output_elements: int
    padding_slots: int
    logical_slots: int
    padding_fraction: float

    input_footprint_bytes: int
    weight_footprint_bytes: int
    output_footprint_bytes: int
    total_footprint_bytes: int

    # Retained only for on-demand pairwise comparison.
    input_coordinates: frozenset
    weight_coordinates: frozenset


@dataclass(frozen=True)
class GroupOverlapFeatures:
    left_group_id: str
    right_group_id: str

    shared_input_elements: int
    left_input_overlap_fraction: float
    right_input_overlap_fraction: float
    input_jaccard: float

    shared_weight_elements: int
    left_weight_overlap_fraction: float
    right_weight_overlap_fraction: float
    weight_jaccard: float


def _group_coordinate_ranges(group_set, group):
    """Return output-coordinate ranges and reduction-coordinate ranges."""
    dimensions = dict(group_set.dimensions)
    output_axes = tuple(group_set.output_axes)

    if set(dimensions) != set(output_axes) | {"C", "R", "S"}:
        raise ValueError(
            "Expected compact convolution dimensions for output axes and C,R,S"
        )

    origin = dict(zip(output_axes, group.output_origin))
    shape = dict(zip(output_axes, group.output_shape))

    output_ranges = [
        range(origin[axis], origin[axis] + shape[axis])
        for axis in output_axes
    ]
    reduction_ranges = [
        range(dimensions[axis]) for axis in ("C", "R", "S")
    ]

    return output_axes, output_ranges, reduction_ranges


def analyze_group_locality(
    adapter: ConvolutionProjectionAdapter,
    group_set,
    group,
    element_bytes=2,
):
    """Compute exact unique input/weight coordinates for one compact group.

    Coordinates are obtained through the existing Timeloop projection adapter.
    Invalid padded input accesses are excluded from the input footprint.
    Output accumulation is represented by unique output elements, not by
    counting every reduction contribution as a separate output-memory access.
    """
    if not isinstance(adapter, ConvolutionProjectionAdapter):
        raise TypeError("adapter must be a ConvolutionProjectionAdapter")

    if element_bytes <= 0:
        raise ValueError("element_bytes must be positive")

    if group.logical_slots <= 0:
        raise ValueError("group.logical_slots must be positive")

    output_axes, output_ranges, reduction_ranges = (
        _group_coordinate_ranges(group_set, group)
    )

    input_coordinates = set()
    weight_coordinates = set()
    input_attempts = 0
    weight_attempts = 0

    for output_coordinate in product(*output_ranges):
        output_values = dict(zip(output_axes, output_coordinate))

        for reduction_coordinate in product(*reduction_ranges):
            values = dict(output_values)
            values.update(zip(("C", "R", "S"), reduction_coordinate))

            accesses = adapter.project(values)

            input_access = accesses["Inputs"]
            weight_access = accesses["Weights"]

            if input_access.valid:
                input_attempts += 1
                input_coordinates.add(input_access.coordinates)

            if weight_access.valid:
                weight_attempts += 1
                weight_coordinates.add(weight_access.coordinates)

    unique_inputs = len(input_coordinates)
    unique_weights = len(weight_coordinates)
    output_elements = prod(group.output_shape)

    input_reuse = (
        input_attempts / unique_inputs if unique_inputs else 0.0
    )
    weight_reuse = (
        weight_attempts / unique_weights if unique_weights else 0.0
    )

    input_bytes = unique_inputs * element_bytes
    weight_bytes = unique_weights * element_bytes
    output_bytes = output_elements * element_bytes

    padding_fraction = (
        group.padding_slots / group.logical_slots
    )

    return GroupLocalityFeatures(
        layer_id=group_set.layer_id,
        group_id=group.group_id,
        element_bytes=element_bytes,
        input_access_attempts=input_attempts,
        unique_input_elements=unique_inputs,
        input_reuse_factor=input_reuse,
        weight_access_attempts=weight_attempts,
        unique_weight_elements=unique_weights,
        weight_reuse_factor=weight_reuse,
        output_elements=output_elements,
        padding_slots=group.padding_slots,
        logical_slots=group.logical_slots,
        padding_fraction=padding_fraction,
        input_footprint_bytes=input_bytes,
        weight_footprint_bytes=weight_bytes,
        output_footprint_bytes=output_bytes,
        total_footprint_bytes=input_bytes + weight_bytes + output_bytes,
        input_coordinates=frozenset(input_coordinates),
        weight_coordinates=frozenset(weight_coordinates),
    )


def compare_group_locality(left, right):
    """Measure exact geometric overlap between two analyzed groups."""
    if left.layer_id != right.layer_id:
        raise ValueError("Groups must belong to the same layer")

    if left.element_bytes != right.element_bytes:
        raise ValueError("Groups must use the same element size")

    shared_inputs = left.input_coordinates & right.input_coordinates
    shared_weights = left.weight_coordinates & right.weight_coordinates

    input_union = left.input_coordinates | right.input_coordinates
    weight_union = left.weight_coordinates | right.weight_coordinates

    return GroupOverlapFeatures(
        left_group_id=left.group_id,
        right_group_id=right.group_id,
        shared_input_elements=len(shared_inputs),
        left_input_overlap_fraction=(
            len(shared_inputs) / len(left.input_coordinates)
            if left.input_coordinates else 0.0
        ),
        right_input_overlap_fraction=(
            len(shared_inputs) / len(right.input_coordinates)
            if right.input_coordinates else 0.0
        ),
        input_jaccard=(
            len(shared_inputs) / len(input_union) if input_union else 0.0
        ),
        shared_weight_elements=len(shared_weights),
        left_weight_overlap_fraction=(
            len(shared_weights) / len(left.weight_coordinates)
            if left.weight_coordinates else 0.0
        ),
        right_weight_overlap_fraction=(
            len(shared_weights) / len(right.weight_coordinates)
            if right.weight_coordinates else 0.0
        ),
        weight_jaccard=(
            len(shared_weights) / len(weight_union) if weight_union else 0.0
        ),
    )
