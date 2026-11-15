"""Compact, architecture-independent hierarchical reduction plans."""

from dataclasses import asdict, dataclass, field
from math import prod
import json
from typing import Any

from simulator.profiling.tlep_schema import (
    MetricProvenance,
)


@dataclass(frozen=True)
class ReductionPlan:
    """Symbolic description of a complete hierarchical reduction.

    This class describes coordinate partitioning without materializing
    individual output groups or execution events.
    """

    workload_id: str
    operator_id: str

    reduction_axes: tuple[str, ...]
    output_axes: tuple[str, ...]

    axis_extents: dict[str, int]

    # Mapping level -> dimension -> factor.
    level_factors: dict[str, dict[str, int]]

    provenance: dict[str, MetricProvenance] = field(default_factory=dict)
    schema_version: str = "1.0"

    def __post_init__(self):
        if not self.workload_id.strip() or not self.operator_id.strip():
            raise ValueError("Workload and operator IDs cannot be empty")

        if not self.reduction_axes or not self.output_axes:
            raise ValueError("Reduction and output axes cannot be empty")

        if len(set(self.reduction_axes)) != len(self.reduction_axes):
            raise ValueError("Duplicate reduction axes")

        if len(set(self.output_axes)) != len(self.output_axes):
            raise ValueError("Duplicate output axes")

        if set(self.reduction_axes) & set(self.output_axes):
            raise ValueError("Reduction and output axes must be disjoint")

        all_axes = set(self.reduction_axes) | set(self.output_axes)

        if set(self.axis_extents) != all_axes:
            raise ValueError("Axis extents must cover exactly all axes")

        if any(
            not name.strip() or extent <= 0
            for name, extent in self.axis_extents.items()
        ):
            raise ValueError("Invalid axis extent")

        if not self.level_factors:
            raise ValueError("At least one hierarchy level is required")

        for level_name, factors in self.level_factors.items():
            if not level_name.strip():
                raise ValueError("Hierarchy level name cannot be empty")

            if set(factors) - all_axes:
                raise ValueError(
                    f"{level_name}: factors contain unknown axes"
                )

            if any(value <= 0 for value in factors.values()):
                raise ValueError(
                    f"{level_name}: factors must be positive"
                )

        for axis, extent in self.axis_extents.items():
            reconstructed = prod(
                factors.get(axis, 1)
                for factors in self.level_factors.values()
            )

            if reconstructed != extent:
                raise ValueError(
                    f"{axis}: hierarchy product {reconstructed} "
                    f"does not match extent {extent}"
                )

    @property
    def total_reduction_coordinates(self) -> int:
        return prod(self.axis_extents[a] for a in self.reduction_axes)

    @property
    def total_output_coordinates(self) -> int:
        return prod(self.axis_extents[a] for a in self.output_axes)

    @property
    def total_logical_combinations(self) -> int:
        return prod(self.axis_extents.values())

    def reduction_factor_by_level(self) -> dict[str, int]:
        return {
            level: prod(
                factors.get(axis, 1)
                for axis in self.reduction_axes
            )
            for level, factors in self.level_factors.items()
        }

    def output_factor_by_level(self) -> dict[str, int]:
        return {
            level: prod(
                factors.get(axis, 1)
                for axis in self.output_axes
            )
            for level, factors in self.level_factors.items()
        }

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["provenance"] = {
            name: {
                **asdict(record),
                "kind": record.kind.value,
            }
            for name, record in self.provenance.items()
        }
        return result

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
