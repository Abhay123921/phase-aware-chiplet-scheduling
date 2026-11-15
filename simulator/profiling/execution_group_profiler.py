"""Reconstruct logical execution-group profiles from LDG ownership.

These profiles describe logical work, not physical execution events,
measured bandwidth, communication, or chiplet timing.
"""

from collections import Counter

from simulator.profiling.tlep_schema import (
    MetricProvenance,
    ProvenanceKind,
    ResourceSignature,
    TileProfile,
)


def build_execution_group_profiles(
    *,
    execution_group_graph,
    operator_graph,
    ldg_graphs,
):
    """Build one logical TileProfile for each execution group.

    Operation counts are populated only when every referenced valid
    work item has a known count and a supported MAC interpretation.
    """

    ldg_graphs = tuple(ldg_graphs)

    execution_group_graph.validate(
        operator_graph=operator_graph,
        ldg_graphs=ldg_graphs,
    )

    ldg_map = {
        (ldg.workload_id, ldg.operator_id, ldg.group_id): ldg
        for ldg in ldg_graphs
    }

    profiles = {}

    for group_id, group in execution_group_graph.groups.items():
        valid_count = 0
        known_work = 0.0
        complete_work = True
        supported_mac = True
        hierarchy_counts = Counter()

        for ref in group.ldg_references:
            ldg = ldg_map[
                (ref.workload_id, ref.operator_id, ref.ldg_group_id)
            ]

            for work_item_id in ref.work_item_ids:
                item = ldg.work_items[work_item_id]

                # Graph validation already prohibits scheduling padding.
                if not item.is_valid:
                    raise ValueError(
                        f"Invalid work item in execution group: {work_item_id}"
                    )

                valid_count += 1

                adapter = item.metadata.get("adapter")
                if adapter not in ("convolution", "gemm"):
                    supported_mac = False

                if item.work_count is None:
                    complete_work = False
                else:
                    known_work += item.work_count

                for level, index in item.hierarchy_indices.items():
                    hierarchy_counts[(level, index)] += 1

        mac_count = (
            known_work if complete_work and supported_mac else None
        )

        operation_counts = (
            {"MAC": mac_count} if mac_count is not None else {}
        )

        provenance = {
            "valid_work_item_count": MetricProvenance(
                kind=ProvenanceKind.RECONSTRUCTED,
                source="ExecutionGroup LDGWorkReference ownership",
                method="Count referenced valid logical work items",
                notes="Logical scheduling candidates, not physical events.",
            ),
        }

        if mac_count is not None:
            provenance["resource_signature.operation_counts.MAC"] = (
                MetricProvenance(
                    kind=ProvenanceKind.RECONSTRUCTED,
                    source="Convolution/GEMM LDG work_item.work_count",
                    method="Sum valid work counts with supported MAC semantics",
                    notes="Logical MAC contributions, not measured instructions.",
                )
            )

        hierarchy_partition_counts = Counter(
            level for level, index in hierarchy_counts
        )
        additional_demands = {
            "valid_logical_work_items": float(valid_count),
            "distinct_hierarchy_partitions": float(len(hierarchy_counts)),
        }

        for level in (
            "dram",
            "buffer",
            "inter_pe_temporal",
            "spatial",
            "reg",
        ):
            key = f"hierarchy_partitions.{level}"
            additional_demands[key] = float(
                hierarchy_partition_counts.get(level, 0)
            )
            provenance[f"resource_signature.additional_demands.{key}"] = (
                MetricProvenance(
                    kind=ProvenanceKind.RECONSTRUCTED,
                    source="LDG work_item.hierarchy_indices",
                    method=f"Count distinct partition indices at {level} level",
                    notes="Logical partition count, not physical transactions.",
                )
            )

        operator = operator_graph.operators[group.operator_id]

        profile = TileProfile(
            workload_id=execution_group_graph.workload_id,
            operator_id=group.operator_id,
            tile_id=group_id,
            operator_type=operator.operator_type,
            work_count=mac_count,
            work_unit="logical_MAC" if mac_count is not None else None,
            resource_signature=ResourceSignature(
                operation_counts=operation_counts,
                additional_demands=additional_demands,
            ),
            estimated_cycles=None,
            measured_cycles=None,
            provenance=provenance,
        )

        profiles[group_id] = profile

    return profiles
