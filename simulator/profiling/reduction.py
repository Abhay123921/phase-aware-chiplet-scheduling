"""Generic logical reduction dependencies for tile-level execution profiles.

This module represents partial contributions to logical outputs.
It does not prescribe a physical reduction topology, communication
mechanism, synchronization protocol, or execution timing.
"""

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Optional

from simulator.profiling.tlep_schema import MetricProvenance


@dataclass(frozen=True)
class ReductionContributor:
    """One logical contribution to an output reduction."""

    contributor_id: str
    logical_coordinates: dict[str, int]
    reduction_coordinates: dict[str, int]

    partial_work_count: Optional[float] = None

    def __post_init__(self):
        if not self.contributor_id.strip():
            raise ValueError("contributor_id cannot be empty")

        if not self.reduction_coordinates:
            raise ValueError("reduction_coordinates cannot be empty")

        for name, coordinates in (
            ("logical_coordinates", self.logical_coordinates),
            ("reduction_coordinates", self.reduction_coordinates),
        ):
            for dimension, coordinate in coordinates.items():
                if not dimension.strip() or coordinate < 0:
                    raise ValueError(f"Invalid {name}: {dimension}")

        if (
            self.partial_work_count is not None
            and self.partial_work_count < 0
        ):
            raise ValueError("partial_work_count cannot be negative")


@dataclass(frozen=True)
class ReductionGroup:
    """Contributors and completion requirement for one logical output."""

    group_id: str
    workload_id: str
    operator_id: str

    output_tensor: str
    output_coordinates: dict[str, int]

    reduction_axes: tuple[str, ...]
    reduction_operator: str

    contributors: tuple[ReductionContributor, ...]
    expected_contributors: int

    # Unknown until independently reconstructed or modeled.
    communication_bytes: Optional[float] = None
    reduction_cycles: Optional[float] = None

    provenance: dict[str, MetricProvenance] = field(default_factory=dict)
    schema_version: str = "1.0"

    def __post_init__(self):
        for name in (
            "group_id",
            "workload_id",
            "operator_id",
            "output_tensor",
            "reduction_operator",
            "schema_version",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} cannot be empty")

        if not self.reduction_axes:
            raise ValueError("reduction_axes cannot be empty")

        if len(set(self.reduction_axes)) != len(self.reduction_axes):
            raise ValueError("reduction_axes must be unique")

        if any(not axis.strip() for axis in self.reduction_axes):
            raise ValueError("Reduction axis names cannot be empty")

        if self.expected_contributors <= 0:
            raise ValueError("expected_contributors must be positive")

        if len(self.contributors) > self.expected_contributors:
            raise ValueError("More contributors than expected")

        for dimension, coordinate in self.output_coordinates.items():
            if not dimension.strip() or coordinate < 0:
                raise ValueError("Invalid output coordinate")

        contributor_ids = [
            contributor.contributor_id for contributor in self.contributors
        ]
        if len(contributor_ids) != len(set(contributor_ids)):
            raise ValueError("Duplicate contributor_id")

        reduction_tuples = []

        for contributor in self.contributors:
            if set(contributor.reduction_coordinates) != set(
                self.reduction_axes
            ):
                raise ValueError(
                    "Contributor reduction dimensions do not match "
                    "the group's reduction_axes"
                )

            reduction_tuple = tuple(
                contributor.reduction_coordinates[axis]
                for axis in self.reduction_axes
            )
            reduction_tuples.append(reduction_tuple)

        if len(reduction_tuples) != len(set(reduction_tuples)):
            raise ValueError("Duplicate reduction-coordinate contribution")

        for name in ("communication_bytes", "reduction_cycles"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative")

    @property
    def received_contributors(self) -> int:
        return len(self.contributors)

    @property
    def is_complete(self) -> bool:
        return self.received_contributors == self.expected_contributors

    @property
    def missing_contributors(self) -> int:
        return self.expected_contributors - self.received_contributors

    def validate_complete(self) -> bool:
        """Require all expected logical contributions to be present."""
        if not self.is_complete:
            raise ValueError(
                f"Incomplete reduction group {self.group_id}: "
                f"{self.received_contributors}/"
                f"{self.expected_contributors} contributors"
            )
        return True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
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
