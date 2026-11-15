"""Generic evaluator for Timeloop data-space projection expressions.

This module is workload-agnostic. It evaluates the nested projection
representation found in parsed-processed-input.yaml.

Each projected axis is a sum of products:
    [['R', 'Wdilation'], ['P', 'Wstride']]

which evaluates to:
    R * Wdilation + P * Wstride

Operator-specific offsets and boundary handling are supplied by adapters.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class DataSpaceProjection:
    name: str
    axes: tuple[tuple[tuple[str, ...], ...], ...]
    read_write: bool = False


def parse_projection_definitions(data_spaces: Sequence[dict]):
    """Parse all data-space projections without operator assumptions."""
    projections = {}

    for entry in data_spaces:
        name = entry["name"]

        if name in projections:
            raise ValueError(f"Duplicate data-space definition: {name}")

        parsed_axes = []

        for axis_index, axis in enumerate(entry["projection"]):
            if not axis:
                raise ValueError(
                    f"Empty projection axis: {name}[{axis_index}]"
                )

            parsed_terms = []

            for term in axis:
                if not term:
                    raise ValueError(
                        f"Empty projection term: {name}[{axis_index}]"
                    )

                if not all(
                    isinstance(symbol, str) and symbol
                    for symbol in term
                ):
                    raise ValueError(
                        f"Invalid projection symbol: {name}[{axis_index}]"
                    )

                parsed_terms.append(tuple(term))

            parsed_axes.append(tuple(parsed_terms))

        projections[name] = DataSpaceProjection(
            name=name,
            axes=tuple(parsed_axes),
            read_write=bool(entry.get("read_write", False)),
        )

    return projections


def evaluate_projection_axis(
    expression: Sequence[Sequence[str]],
    values: Mapping[str, int],
) -> int:
    """Evaluate one axis as a sum of multiplicative terms."""
    result = 0

    for term in expression:
        product = 1

        for symbol in term:
            if symbol not in values:
                raise KeyError(
                    f"Missing projection variable: {symbol}"
                )

            product *= values[symbol]

        result += product

    return result


def project_data_space(
    projection: DataSpaceProjection,
    values: Mapping[str, int],
    offsets: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Evaluate all tensor axes, applying optional adapter offsets."""
    if offsets is None:
        offsets = (0,) * len(projection.axes)

    if len(offsets) != len(projection.axes):
        raise ValueError(
            f"Expected {len(projection.axes)} offsets, "
            f"received {len(offsets)}"
        )

    coordinates = []

    for axis, offset in zip(projection.axes, offsets):
        coordinates.append(
            evaluate_projection_axis(axis, values) + offset
        )

    return tuple(coordinates)
