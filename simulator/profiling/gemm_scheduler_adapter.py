"""Adapt GEMM execution groups to the existing compact scheduler interface.

The original execution-group dependency graph remains authoritative.
This module provides only the ordered group view required by the scheduler.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SchedulerGroup:
    group_id: str


@dataclass(frozen=True)
class SchedulerGroupSet:
    layer_id: str
    groups: tuple


def adapt_gemm_execution_groups_for_scheduler(
    *,
    execution_group_graph,
    gemm_profiles,
):
    """Return a scheduler-compatible ordered group set and profile tuple.

    Group order follows execution_group_graph.groups insertion order.
    Profiles must have exactly the same IDs and order. No profile values
    are modified or re-estimated by this adapter.
    """
    if not execution_group_graph.groups:
        raise ValueError("GEMM execution-group graph is empty")

    if not gemm_profiles:
        raise ValueError("GEMM resource profiles are empty")

    graph_groups = tuple(execution_group_graph.groups.values())
    profiles = tuple(gemm_profiles)

    if len(graph_groups) != len(profiles):
        raise ValueError("GEMM group/profile count mismatch")

    layer_id = str(execution_group_graph.workload_id)
    expected_ids = tuple(group.execution_group_id for group in graph_groups)
    profile_ids = tuple(profile.group_id for profile in profiles)

    if expected_ids != profile_ids:
        raise ValueError(
            "GEMM group/profile ordering or identifier mismatch"
        )

    for profile in profiles:
        if str(profile.layer_id) != layer_id:
            raise ValueError(
                "GEMM profile layer_id does not match execution graph workload"
            )

    scheduler_group_set = SchedulerGroupSet(
        layer_id=layer_id,
        groups=tuple(
            SchedulerGroup(group_id=group_id)
            for group_id in expected_ids
        ),
    )

    return scheduler_group_set, profiles
