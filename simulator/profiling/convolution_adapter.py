"""Convolution-specific adapter over the generic projection engine."""

from dataclasses import dataclass
from typing import Mapping

from simulator.profiling.projection import (
    DataSpaceProjection,
    parse_projection_definitions,
    project_data_space,
)


@dataclass(frozen=True)
class ProjectedTensorAccess:
    data_space: str
    coordinates: tuple[int, ...]
    access_mode: str
    valid: bool
    invalid_reason: str | None = None


@dataclass(frozen=True)
class ConvolutionProjectionAdapter:
    instance: Mapping[str, int]
    projections: Mapping[str, DataSpaceProjection]

    @classmethod
    def from_problem(cls, problem):
        instance = problem["instance"]
        projections = parse_projection_definitions(
            problem["shape"]["data_spaces"]
        )

        required = {"Weights", "Inputs", "Outputs"}
        if set(projections) != required:
            raise ValueError(
                f"Unexpected convolution data spaces: {set(projections)}"
            )

        if any(len(projections[name].axes) != 5 for name in required):
            raise ValueError("Expected five tensor axes per data space")

        return cls(instance=dict(instance), projections=projections)

    def project(self, workload_coordinates):
        values = dict(self.instance)
        values.update(workload_coordinates)

        offsets = {
            "Weights": (0, 0, 0, 0, 0),
            "Inputs": (
                0, 0, -self.instance["Wpad"],
                -self.instance["Hpad"], 0,
            ),
            "Outputs": (0, 0, 0, 0, 0),
        }

        extents = {
            "Weights": tuple(
                self.instance[k] for k in ("C", "M", "R", "S", "G")
            ),
            "Inputs": tuple(
                self.instance[k] for k in ("N", "C", "W", "H", "G")
            ),
            "Outputs": tuple(
                self.instance[k] for k in ("N", "M", "Q", "P", "G")
            ),
        }

        results = {}

        for name, projection in self.projections.items():
            coordinates = project_data_space(
                projection, values, offsets[name]
            )

            valid = all(
                0 <= coordinate < extent
                for coordinate, extent in zip(coordinates, extents[name])
            )

            reason = None
            if not valid:
                reason = "padding" if name == "Inputs" else "out_of_bounds"

            results[name] = ProjectedTensorAccess(
                data_space=name,
                coordinates=coordinates,
                access_mode=(
                    "read_write" if projection.read_write else "read"
                ),
                valid=valid,
                invalid_reason=reason,
            )

        return results
