"""Generic Tile-Level Execution Profile (TLEP) schema.

This module is independent of any particular workload, accelerator,
chiplet architecture, profiling tool, or scheduling policy.

All optional quantities remain None when unavailable. In particular,
unknown communication or execution quantities must not be interpreted
as zero.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from typing import Any, Optional


SCHEMA_VERSION = "1.0"


class ProvenanceKind(str, Enum):
    MEASURED = "measured"
    RECONSTRUCTED = "reconstructed"
    ESTIMATED = "estimated"
    ASSUMPTION = "assumption"


@dataclass(frozen=True)
class MetricProvenance:
    """Origin and derivation information for a profile quantity."""

    kind: ProvenanceKind
    source: str
    method: str
    confidence: Optional[float] = None
    notes: Optional[str] = None

    def __post_init__(self):
        if not self.source.strip():
            raise ValueError("Provenance source cannot be empty")

        if not self.method.strip():
            raise ValueError("Provenance method cannot be empty")

        if self.confidence is not None:
            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError("Confidence must be between 0 and 1")


@dataclass(frozen=True)
class TensorAccess:
    """Logical accesses to a tensor by one tile.

    read_elements and write_elements represent total element accesses,
    not necessarily unique elements.

    unique_elements represents the number of distinct tensor elements
    involved, when known.

    Byte counts are optional because element sizes may be unavailable.
    """

    tensor_name: str
    role: str

    read_elements: Optional[int] = None
    write_elements: Optional[int] = None
    unique_elements: Optional[int] = None

    element_size_bytes: Optional[float] = None
    read_bytes: Optional[float] = None
    write_bytes: Optional[float] = None

    memory_level: Optional[str] = None

    def __post_init__(self):
        if not self.tensor_name.strip():
            raise ValueError("Tensor name cannot be empty")

        if not self.role.strip():
            raise ValueError("Tensor role cannot be empty")

        for name in (
            "read_elements",
            "write_elements",
            "unique_elements",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative")

        for name in (
            "element_size_bytes",
            "read_bytes",
            "write_bytes",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative")


@dataclass(frozen=True)
class MemoryFootprint:
    """Storage footprint of a tensor at a specified memory level."""

    tensor_name: str
    memory_level: str

    elements: Optional[int] = None
    bytes: Optional[float] = None

    capacity_unit: Optional[str] = None

    def __post_init__(self):
        if not self.tensor_name.strip():
            raise ValueError("Tensor name cannot be empty")

        if not self.memory_level.strip():
            raise ValueError("Memory level cannot be empty")

        if self.elements is not None and self.elements < 0:
            raise ValueError("Footprint elements cannot be negative")

        if self.bytes is not None and self.bytes < 0:
            raise ValueError("Footprint bytes cannot be negative")


@dataclass(frozen=True)
class ResourceSignature:
    """Architecture-independent resource-demand description.

    operation_counts uses explicit operation names and counts, such as
    {"MAC": 100000} or {"FLOP": 200000}.

    Memory and communication quantities use bytes when available.
    available_parallelism is a logical count, not a utilization ratio.
    """

    operation_counts: dict[str, float] = field(default_factory=dict)

    memory_read_bytes: Optional[float] = None
    memory_write_bytes: Optional[float] = None

    communication_bytes: Optional[float] = None
    storage_bytes: Optional[float] = None

    available_parallelism: Optional[float] = None

    additional_demands: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        for name, count in self.operation_counts.items():
            if not name.strip() or count < 0:
                raise ValueError("Invalid operation count")

        for name in (
            "memory_read_bytes",
            "memory_write_bytes",
            "communication_bytes",
            "storage_bytes",
            "available_parallelism",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative")

        for name, value in self.additional_demands.items():
            if not name.strip() or value < 0:
                raise ValueError("Invalid additional resource demand")


@dataclass
class TileProfile:
    """Generic execution profile for a logical workload tile."""

    workload_id: str
    operator_id: str
    tile_id: str
    operator_type: str

    parent_tile_id: Optional[str] = None

    # Operator-specific coordinates remain generic key-value pairs.
    logical_coordinates: dict[str, int] = field(default_factory=dict)

    # Half-open iteration ranges: (start, end).
    iteration_ranges: dict[str, tuple[int, int]] = field(
        default_factory=dict
    )

    work_count: Optional[float] = None
    work_unit: Optional[str] = None

    tensor_accesses: list[TensorAccess] = field(default_factory=list)
    memory_footprints: list[MemoryFootprint] = field(default_factory=list)

    dependencies: list[str] = field(default_factory=list)

    resource_signature: ResourceSignature = field(
        default_factory=ResourceSignature
    )

    estimated_cycles: Optional[float] = None
    measured_cycles: Optional[float] = None
    cycle_reference: Optional[str] = None

    # Maps metric/field names to their provenance.
    provenance: dict[str, MetricProvenance] = field(
        default_factory=dict
    )

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        for name in (
            "workload_id",
            "operator_id",
            "tile_id",
            "operator_type",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} cannot be empty")

        if self.work_count is not None and self.work_count < 0:
            raise ValueError("work_count cannot be negative")

        for name in ("estimated_cycles", "measured_cycles"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative")

        for dimension, coordinate in self.logical_coordinates.items():
            if not dimension.strip() or coordinate < 0:
                raise ValueError("Invalid logical coordinate")

        for dimension, bounds in self.iteration_ranges.items():
            if not dimension.strip():
                raise ValueError("Iteration dimension cannot be empty")

            if len(bounds) != 2:
                raise ValueError("Iteration ranges need two bounds")

            start, end = bounds
            if start < 0 or end < start:
                raise ValueError(f"Invalid iteration range: {dimension}")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""
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
        """Serialize the profile as JSON."""
        return json.dumps(self.to_dict(), indent=indent)
