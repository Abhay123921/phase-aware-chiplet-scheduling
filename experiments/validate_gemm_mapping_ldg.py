#!/usr/bin/env python3
"""Validate a real Timeloop GEMM mapping against the generic LDG model."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulator.profiling.gemm_ldg_adapter import build_gemm_ldg_from_mapping
from simulator.profiling.iteration_space import (
    SymbolicIterationSpace,
    encode_workload_coordinates,
)
from simulator.profiling.mapping_parser import load_mapping_and_workload


def validate_mapping_artifacts(directory: Path) -> dict:
    mapping_path = directory / "timeloop-mapper.map.yaml"
    workload_path = directory / "parsed-processed-input.yaml"

    for path in (mapping_path, workload_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required mapper artifact not found: {path}")

    mapping, dimensions = load_mapping_and_workload(
        mapping_path,
        workload_path,
    )

    required_dimensions = {"M", "N", "K"}
    if set(dimensions) != required_dimensions:
        raise ValueError(
            f"Expected exactly {sorted(required_dimensions)}, "
            f"got {sorted(dimensions)}"
        )

    M, N, K = (dimensions[name] for name in ("M", "N", "K"))
    expected_operations = M * N * K
    expected_outputs = M * N

    space = SymbolicIterationSpace.from_mapping(mapping, dimensions)
    space.validate()

    graph_count = 0
    work_item_count = 0
    contribution_count = 0
    round_trip_count = 0
    hierarchy_tuples = set()

    for m in range(M):
        for n in range(N):
            graph = build_gemm_ldg_from_mapping(
                mapping=mapping,
                workload_dimensions=dimensions,
                workload_id=directory.name,
                operator_id="gemm_0",
                group_id=f"C_{m}_{n}",
                output_coordinates={"M": m, "N": n},
            )

            if not graph.validate():
                raise AssertionError(f"Invalid LDG for output C[{m},{n}]")

            if graph.logical_slot_count != K:
                raise AssertionError(
                    f"C[{m},{n}]: expected {K} logical slots, "
                    f"got {graph.logical_slot_count}"
                )

            if graph.valid_work_item_count != K:
                raise AssertionError(
                    f"C[{m},{n}]: expected {K} valid work items, "
                    f"got {graph.valid_work_item_count}"
                )

            if graph.invalid_work_item_count != 0:
                raise AssertionError(
                    f"C[{m},{n}]: unexpected invalid work items"
                )

            contributions = [
                edge for edge in graph.edges
                if edge.relation == "contributes_to"
            ]
            if len(contributions) != K:
                raise AssertionError(
                    f"C[{m},{n}]: expected {K} contribution edges, "
                    f"got {len(contributions)}"
                )

            for k in range(K):
                item_id = f"gemm_0:m{m}:n{n}:k{k}"
                item = graph.work_items[item_id]

                decoded = space.decode(
                    dram_index=item.hierarchy_indices["dram"],
                    buffer_index=item.hierarchy_indices["buffer"],
                    inter_pe_temporal_index=(
                        item.hierarchy_indices["inter_pe_temporal"]
                    ),
                    spatial_index=item.hierarchy_indices["spatial"],
                    reg_index=item.hierarchy_indices["reg"],
                )

                expected_coordinates = {"M": m, "N": n, "K": k}
                if decoded.workload_coordinates != expected_coordinates:
                    raise AssertionError(
                        f"Round-trip mismatch for {item_id}: "
                        f"{decoded.workload_coordinates}"
                    )

                encoded = encode_workload_coordinates(
                    space,
                    expected_coordinates,
                )
                if encoded != item.hierarchy_indices:
                    raise AssertionError(
                        f"Encoder/adapter mismatch for {item_id}: "
                        f"encoded={encoded}, adapter={item.hierarchy_indices}"
                    )

                hierarchy_key = tuple(
                    item.hierarchy_indices[level]
                    for level in (
                        "dram",
                        "buffer",
                        "inter_pe_temporal",
                        "spatial",
                        "reg",
                    )
                )

                if hierarchy_key in hierarchy_tuples:
                    raise AssertionError(
                        f"Hierarchy-index collision at {item_id}: "
                        f"{hierarchy_key}"
                    )

                hierarchy_tuples.add(hierarchy_key)
                round_trip_count += 1

            graph_count += 1
            work_item_count += graph.logical_slot_count
            contribution_count += len(contributions)

    checks = {
        "output_graphs": graph_count,
        "logical_work_items": work_item_count,
        "valid_contribution_edges": contribution_count,
        "coordinate_round_trips": round_trip_count,
        "unique_hierarchy_tuples": len(hierarchy_tuples),
        "expected_outputs": expected_outputs,
        "expected_operations": expected_operations,
        "dimensions": dimensions,
    }

    expected_counts = {
        "output_graphs": expected_outputs,
        "logical_work_items": expected_operations,
        "valid_contribution_edges": expected_operations,
        "coordinate_round_trips": expected_operations,
        "unique_hierarchy_tuples": expected_operations,
    }

    for name, expected in expected_counts.items():
        if checks[name] != expected:
            raise AssertionError(
                f"{name}: expected {expected}, got {checks[name]}"
            )

    return checks



def validate_padding_case(mapping, dimensions, logical_K: int) -> dict:
    """Validate logical reduction padding within the mapped K extent."""
    M, N, mapped_K = (
        dimensions[name] for name in ("M", "N", "K")
    )

    if isinstance(logical_K, bool) or not isinstance(logical_K, int):
        raise ValueError("logical_K must be an integer")
    if not 0 < logical_K < mapped_K:
        raise ValueError(
            f"Padding test requires 0 < logical_K < mapped_K; "
            f"got {logical_K} and {mapped_K}"
        )

    graph_count = 0
    slot_count = 0
    valid_count = 0
    invalid_count = 0
    contribution_count = 0
    round_trip_count = 0
    hierarchy_tuples = set()

    space = SymbolicIterationSpace.from_mapping(mapping, dimensions)

    for m in range(M):
        for n in range(N):
            graph = build_gemm_ldg_from_mapping(
                mapping=mapping,
                workload_dimensions=dimensions,
                workload_id="gemm_padding_validation",
                operator_id="gemm_padding_test",
                group_id=f"C_{m}_{n}",
                output_coordinates={"M": m, "N": n},
                logical_K=logical_K,
            )

            assert graph.validate()
            assert graph.logical_slot_count == mapped_K
            assert graph.valid_work_item_count == logical_K
            assert graph.invalid_work_item_count == mapped_K - logical_K

            contributions = [
                edge for edge in graph.edges
                if edge.relation == "contributes_to"
            ]
            assert len(contributions) == logical_K

            invalid_items = [
                item for item in graph.work_items.values()
                if not item.is_valid
            ]
            assert len(invalid_items) == mapped_K - logical_K

            for k in range(mapped_K):
                item_id = f"gemm_padding_test:m{m}:n{n}:k{k}"
                item = graph.work_items[item_id]

                decoded = space.decode(
                    dram_index=item.hierarchy_indices["dram"],
                    buffer_index=item.hierarchy_indices["buffer"],
                    inter_pe_temporal_index=(
                        item.hierarchy_indices["inter_pe_temporal"]
                    ),
                    spatial_index=item.hierarchy_indices["spatial"],
                    reg_index=item.hierarchy_indices["reg"],
                )

                expected = {"M": m, "N": n, "K": k}
                assert decoded.workload_coordinates == expected

                encoded = encode_workload_coordinates(space, expected)
                assert encoded == item.hierarchy_indices

                key = tuple(
                    item.hierarchy_indices[level]
                    for level in (
                        "dram", "buffer", "inter_pe_temporal",
                        "spatial", "reg",
                    )
                )
                assert key not in hierarchy_tuples, (
                    f"Hierarchy collision at {item_id}: {key}"
                )
                hierarchy_tuples.add(key)
                round_trip_count += 1

                if k < logical_K:
                    assert item.is_valid
                    assert item.work_count == 1
                else:
                    assert not item.is_valid
                    assert item.invalid_reason == "reduction_padding"
                    assert not any(
                        edge.source_id == item.work_item_id
                        and edge.relation == "contributes_to"
                        for edge in graph.edges
                    )

            graph_count += 1
            slot_count += graph.logical_slot_count
            valid_count += graph.valid_work_item_count
            invalid_count += graph.invalid_work_item_count
            contribution_count += len(contributions)

    expected_outputs = M * N
    expected_slots = expected_outputs * mapped_K
    expected_valid = expected_outputs * logical_K
    expected_invalid = expected_outputs * (mapped_K - logical_K)

    expected = {
        "output_graphs": expected_outputs,
        "logical_slots": expected_slots,
        "valid_work_items": expected_valid,
        "invalid_padding_items": expected_invalid,
        "contribution_edges": expected_valid,
        "coordinate_round_trips": expected_slots,
        "unique_hierarchy_tuples": expected_slots,
    }
    actual = {
        "output_graphs": graph_count,
        "logical_slots": slot_count,
        "valid_work_items": valid_count,
        "invalid_padding_items": invalid_count,
        "contribution_edges": contribution_count,
        "coordinate_round_trips": round_trip_count,
        "unique_hierarchy_tuples": len(hierarchy_tuples),
    }

    assert actual == expected, (
        f"Padding validation mismatch: expected={expected}, actual={actual}"
    )

    return {
        "logical_K": logical_K,
        "mapped_K": mapped_K,
        **actual,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate a mapper-derived GEMM logical dependency graph."
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("timeloop/outputs/gemm/gemm_16x16x16"),
        help="Directory containing Timeloop mapper output artifacts.",
    )
    parser.add_argument(
        "--padding-logical-k",
        type=int,
        default=None,
        help=(
            "Additionally validate logical K padding within the mapped K "
            "extent. Must be positive and smaller than mapped K."
        ),
    )

    args = parser.parse_args()

    result = validate_mapping_artifacts(args.artifact_dir)

    print("=== MAPPER-DERIVED GEMM LDG INTEGRATION VALIDATION ===")
    print("Artifact directory:", args.artifact_dir)
    print("Workload dimensions:", result["dimensions"])
    print("Output graphs:", result["output_graphs"])
    print("Logical work items:", result["logical_work_items"])
    print("Valid contribution edges:", result["valid_contribution_edges"])
    print("Coordinate round-trip checks:", result["coordinate_round_trips"])
    print("Unique hierarchy tuples:", result["unique_hierarchy_tuples"])
    print("Expected logical operations:", result["expected_operations"])
    print("Hierarchy collisions: 0")
    print("Missing hierarchy tuples: 0")
    if args.padding_logical_k is not None:
        padding = validate_padding_case(
            mapping=load_mapping_and_workload(
                args.artifact_dir / "timeloop-mapper.map.yaml",
                args.artifact_dir / "parsed-processed-input.yaml",
            )[0],
            dimensions=result["dimensions"],
            logical_K=args.padding_logical_k,
        )

        print("\\n=== EXPLICIT REDUCTION-PADDING VALIDATION ===")
        print("Logical K:", padding["logical_K"])
        print("Mapped K:", padding["mapped_K"])
        print("Padding output graphs:", padding["output_graphs"])
        print("Mapped logical slots:", padding["logical_slots"])
        print("Valid work items:", padding["valid_work_items"])
        print("Invalid padding items:", padding["invalid_padding_items"])
        print("Contribution edges:", padding["contribution_edges"])
        print("Padding contribution edges: 0")
        print("Padding hierarchy round-trips:", padding["coordinate_round_trips"])
        print("Unique mapped hierarchy tuples:", padding["unique_hierarchy_tuples"])

    print("Final result: PASS")


if __name__ == "__main__":
    main()
