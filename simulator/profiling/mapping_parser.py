"""Generic parser for Timeloop mapping descriptions.

Supports arbitrary workload dimension names. It does not assume
convolution-specific dimensions or generate execution tiles.
"""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Optional

import yaml


@dataclass(frozen=True)
class MappingNest:
    target: str
    mapping_type: str
    factors: dict[str, int]
    permutation: str
    loop_order: tuple[tuple[str, int], ...]
    split: Optional[int] = None


@dataclass(frozen=True)
class DataSpacePolicy:
    target: str
    keep: tuple[str, ...] = ()
    bypass: tuple[str, ...] = ()


@dataclass
class MappingDescription:
    nests: list[MappingNest] = field(default_factory=list)
    data_space_policies: dict[str, DataSpacePolicy] = field(
        default_factory=dict
    )

    def get_nest(self, target, mapping_type):
        matches = [
            nest for nest in self.nests
            if nest.target == target
            and nest.mapping_type == mapping_type
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one {mapping_type} nest for {target}; "
                f"found {len(matches)}"
            )
        return matches[0]

    def factor_product(self, dimension):
        product = 1
        for nest in self.nests:
            product *= nest.factors.get(dimension, 1)
        return product

    def validate_workload_dimensions(self, dimensions):
        errors = []

        for dimension, extent in dimensions.items():
            reconstructed = self.factor_product(dimension)
            if reconstructed != extent:
                errors.append(
                    f"{dimension}: mapping product={reconstructed}, "
                    f"workload extent={extent}"
                )

        mapped_dimensions = {
            dimension
            for nest in self.nests
            for dimension in nest.factors
        }

        for dimension in mapped_dimensions - set(dimensions):
            errors.append(
                f"{dimension}: appears in mapping but not workload"
            )

        if errors:
            raise ValueError(
                "Mapping/workload factor validation failed:\n"
                + "\n".join(errors)
            )

        return True


def parse_factors(factor_string):
    """Parse Timeloop factors such as 'C3 M1 R7 S7 N1 P16 Q8 G1'.

    Dimension symbols contain letters/underscores only.
    Numeric suffixes represent the complete positive factor.
    """
    if not isinstance(factor_string, str):
        raise TypeError("Mapping factors must be a string")

    factors = {}

    for token in factor_string.split():
        match = re.fullmatch(r"([A-Za-z_]+)(\d+)", token)

        if match is None:
            raise ValueError(
                f"Invalid mapping factor token: {token!r}"
            )

        dimension, value = match.groups()

        if dimension in factors:
            raise ValueError(
                f"Duplicate mapping factor: {dimension}"
            )

        value = int(value)

        if value <= 0:
            raise ValueError(
                f"Mapping factors must be positive: {token}"
            )

        factors[dimension] = value

    if not factors:
        raise ValueError(
            f"Could not parse mapping factors: {factor_string}"
        )

    return factors


def parse_nest(entry):
    target = entry["target"]
    mapping_type = entry["type"]

    if mapping_type not in ("temporal", "spatial"):
        raise ValueError(
            f"Unsupported mapping nest type: {mapping_type}"
        )

    factors = parse_factors(entry["factors"])
    permutation = entry["permutation"]

    if len(permutation) != len(set(permutation)):
        raise ValueError(
            f"Repeated dimensions in permutation: {permutation}"
        )

    if set(permutation) != set(factors):
        raise ValueError(
            f"Permutation/factor dimensions differ for {target}: "
            f"{permutation} vs {list(factors)}"
        )

    # Timeloop map.txt presents the loops outer-to-inner.
    # YAML permutation is inner-to-outer.
    loop_order = tuple(
        (dimension, factors[dimension])
        for dimension in reversed(permutation)
        if factors[dimension] > 1
    )

    return MappingNest(
        target=target,
        mapping_type=mapping_type,
        factors=factors,
        permutation=permutation,
        loop_order=loop_order,
        split=entry.get("split"),
    )


def parse_mapping_file(mapping_path):
    mapping_path = Path(mapping_path)
    data = yaml.safe_load(mapping_path.read_text())

    if not isinstance(data, dict) or "mapping" not in data:
        raise ValueError(
            f"Invalid Timeloop mapping file: {mapping_path}"
        )

    description = MappingDescription()

    for entry in data["mapping"]:
        target = entry["target"]
        mapping_type = entry["type"]

        if mapping_type == "datatype":
            policy = DataSpacePolicy(
                target=target,
                keep=tuple(entry.get("keep", [])),
                bypass=tuple(entry.get("bypass", [])),
            )

            if target in description.data_space_policies:
                raise ValueError(
                    f"Duplicate datatype policy for {target}"
                )

            description.data_space_policies[target] = policy

        elif mapping_type in ("temporal", "spatial"):
            description.nests.append(parse_nest(entry))

        else:
            raise ValueError(
                f"Unsupported mapping type: {mapping_type}"
            )

    return description


def load_mapping_and_workload(mapping_path, workload_path):
    """Load mapping and workload dimensions, then validate factors."""
    mapping = parse_mapping_file(mapping_path)

    workload_data = yaml.safe_load(
        Path(workload_path).read_text()
    )

    problem = workload_data["problem"]
    instance = problem["instance"]
    dimensions = problem["shape"]["dimensions"]

    workload_dimensions = {
        dimension: int(instance[dimension])
        for dimension in dimensions
    }

    mapping.validate_workload_dimensions(workload_dimensions)

    return mapping, workload_dimensions
