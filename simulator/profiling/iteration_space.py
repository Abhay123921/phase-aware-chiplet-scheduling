"""Symbolic reconstruction of hierarchical Timeloop iteration spaces.

This module:
- Preserves the temporal and spatial mapping hierarchy.
- Decodes individual loop-combination indices lazily.
- Reconstructs workload coordinates using mixed-radix composition.
- Does not materialize all logical iterations.
- Does not claim to model execution time or tensor traffic.
"""

from dataclasses import dataclass
from math import prod
from typing import Dict

from simulator.profiling.mapping_parser import MappingDescription


@dataclass(frozen=True)
class IterationLevel:
    name: str
    target: str
    mapping_type: str
    factors: Dict[str, int]
    loop_order: tuple
    iteration_count: int

    @classmethod
    def from_nest(cls, name, nest):
        return cls(
            name=name,
            target=nest.target,
            mapping_type=nest.mapping_type,
            factors=dict(nest.factors),
            loop_order=tuple(nest.loop_order),
            iteration_count=prod(nest.factors.values()),
        )


@dataclass(frozen=True)
class LogicalIteration:
    level_indices: Dict[str, Dict[str, int]]
    workload_coordinates: Dict[str, int]


@dataclass(frozen=True)
class SymbolicIterationSpace:
    workload_dimensions: Dict[str, int]
    levels: tuple

    @classmethod
    def from_mapping(cls, mapping, workload_dimensions):
        """Construct an iteration space from a parsed mapping.

        The level order follows the current Timeloop hierarchy:
        DRAM -> buffer -> inter-PE temporal -> inter-PE spatial -> reg.

        Non-unit inter-PE temporal and register temporal loops are
        retained explicitly; their interpretation as execution events
        is not implied by this representation.
        """
        specifications = [
            ("dram", "DRAM", "temporal"),
            ("buffer", "buffer", "temporal"),
            ("inter_pe_temporal", "inter_PE_spatial", "temporal"),
            ("spatial", "inter_PE_spatial", "spatial"),
            ("reg", "reg", "temporal"),
        ]

        levels = []

        for name, target, mapping_type in specifications:
            nest = mapping.get_nest(target, mapping_type)
            levels.append(IterationLevel.from_nest(name, nest))

        space = cls(
            workload_dimensions=dict(workload_dimensions),
            levels=tuple(levels),
        )

        space.validate()
        return space

    def get_level(self, name):
        matches = [level for level in self.levels if level.name == name]

        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one level named {name!r}"
            )

        return matches[0]

    @property
    def temporal_iteration_count(self):
        return prod(
            level.iteration_count
            for level in self.levels
            if level.mapping_type == "temporal"
        )

    @property
    def spatial_parallelism(self):
        return self.get_level("spatial").iteration_count

    @property
    def total_logical_combinations(self):
        return prod(level.iteration_count for level in self.levels)

    def validate(self):
        """Check that all hierarchy factors reconstruct workload extents."""
        dimensions = set(self.workload_dimensions)

        for level in self.levels:
            unknown = set(level.factors) - dimensions
            if unknown:
                raise ValueError(
                    f"{level.name}: unknown dimensions {sorted(unknown)}"
                )

            if any(value <= 0 for value in level.factors.values()):
                raise ValueError(
                    f"{level.name}: factors must be positive"
                )

        for dimension, extent in self.workload_dimensions.items():
            if extent <= 0:
                raise ValueError(
                    f"Invalid workload extent for {dimension}: {extent}"
                )

            reconstructed = prod(
                level.factors.get(dimension, 1)
                for level in self.levels
            )

            if reconstructed != extent:
                raise ValueError(
                    f"{dimension}: hierarchy product={reconstructed}, "
                    f"workload extent={extent}"
                )

        return True

    @staticmethod
    def decode_level(level, linear_index):
        """Decode one linear loop index into dimension-specific digits.

        loop_order is outer-to-inner, so the final loop varies fastest.
        """
        if not 0 <= linear_index < level.iteration_count:
            raise IndexError(
                f"{level.name} index {linear_index} outside "
                f"[0, {level.iteration_count})"
            )

        remaining = linear_index
        coordinates = {
            dimension: 0 for dimension in level.factors
        }

        for dimension, factor in reversed(level.loop_order):
            coordinates[dimension] = remaining % factor
            remaining //= factor

        if remaining != 0:
            raise ValueError(
                f"Could not completely decode {level.name} index"
            )

        return coordinates

    def decode(
        self,
        dram_index,
        buffer_index,
        spatial_index=0,
        inter_pe_temporal_index=0,
        reg_index=0,
    ):
        """Decode one selected hierarchy combination.

        All indices are explicit. This method does not enumerate
        neighboring iterations or infer execution timing.
        """
        selected = {
            "dram": dram_index,
            "buffer": buffer_index,
            "inter_pe_temporal": inter_pe_temporal_index,
            "spatial": spatial_index,
            "reg": reg_index,
        }

        level_indices = {}

        for level in self.levels:
            index = selected[level.name]
            level_indices[level.name] = self.decode_level(
                level, index
            )

        # Compose mixed-radix digits from outermost to innermost level.
        workload_coordinates = {}

        for dimension in self.workload_dimensions:
            coordinate = 0

            for level in self.levels:
                factor = level.factors.get(dimension, 1)
                digit = level_indices[level.name].get(dimension, 0)

                coordinate = coordinate * factor + digit

            workload_coordinates[dimension] = coordinate

        for dimension, coordinate in workload_coordinates.items():
            if not 0 <= coordinate < self.workload_dimensions[dimension]:
                raise ValueError(
                    f"Reconstructed coordinate out of bounds: "
                    f"{dimension}={coordinate}"
                )

        return LogicalIteration(
            level_indices=level_indices,
            workload_coordinates=workload_coordinates,
        )

    def summary(self):
        return {
            "workload_dimensions": dict(self.workload_dimensions),
            "levels": {
                level.name: {
                    "target": level.target,
                    "mapping_type": level.mapping_type,
                    "iteration_count": level.iteration_count,
                    "loop_order": level.loop_order,
                    "factors": dict(level.factors),
                }
                for level in self.levels
            },
            "temporal_iteration_count": self.temporal_iteration_count,
            "spatial_parallelism": self.spatial_parallelism,
            "total_logical_combinations": self.total_logical_combinations,
        }


def encode_level(level, digits):
    """Encode dimension digits using a level's outer-to-inner loop order."""
    if not isinstance(digits, dict):
        raise TypeError("Level digits must be supplied as a dictionary")

    expected = set(level.factors)
    received = set(digits)

    if received != expected:
        raise ValueError(
            f"{level.name}: expected digits for {sorted(expected)}, "
            f"received {sorted(received)}"
        )

    for dimension, factor in level.factors.items():
        value = digits[dimension]
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value < factor
        ):
            raise ValueError(
                f"{level.name}: invalid digit for {dimension}: {value}"
            )

    encoded_dimensions = {name for name, _ in level.loop_order}
    required_dimensions = {
        name for name, factor in level.factors.items() if factor > 1
    }

    if not required_dimensions.issubset(encoded_dimensions):
        raise ValueError(
            f"{level.name}: loop order omits non-unit dimensions "
            f"{sorted(required_dimensions - encoded_dimensions)}"
        )

    index = 0
    for dimension, factor in level.loop_order:
        index = index * factor + digits[dimension]

    if not 0 <= index < level.iteration_count:
        raise ValueError(f"Encoded index outside level {level.name}")

    return index


def encode_workload_coordinates(space, coordinates):
    """Encode workload coordinates into all hierarchy-level indices.

    This is the inverse of SymbolicIterationSpace.decode().
    """
    if not isinstance(coordinates, dict):
        raise TypeError("Workload coordinates must be supplied as a dictionary")

    expected = set(space.workload_dimensions)
    received = set(coordinates)

    if received != expected:
        raise ValueError(
            f"Expected workload dimensions {sorted(expected)}, "
            f"received {sorted(received)}"
        )

    for dimension, extent in space.workload_dimensions.items():
        value = coordinates[dimension]
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value < extent
        ):
            raise ValueError(
                f"Coordinate out of bounds for {dimension}: {value}"
            )

    indices = {}

    for level_index, level in enumerate(space.levels):
        digits = {}

        for dimension, factor in level.factors.items():
            inner_factor = prod(
                later.factors.get(dimension, 1)
                for later in space.levels[level_index + 1:]
            )

            digits[dimension] = (
                coordinates[dimension] // inner_factor
            ) % factor

        indices[level.name] = encode_level(level, digits)

    return indices
