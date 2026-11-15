"""Generic operator-level tensor dependency graph.

This graph represents operator declarations and tensor-level
producer-consumer dependencies. It does not prescribe execution order,
physical placement, synchronization, communication cost, or timing.
"""

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional
import json


OPERATOR_GRAPH_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class OperatorNode:
    """One operator in a workload computation graph."""

    operator_id: str
    operator_type: str
    input_tensors: tuple[str, ...]
    output_tensors: tuple[str, ...]
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.operator_id.strip():
            raise ValueError("operator_id cannot be empty")
        if not self.operator_type.strip():
            raise ValueError("operator_type cannot be empty")
        if len(set(self.input_tensors)) != len(self.input_tensors):
            raise ValueError("Operator input tensors must be unique")
        if len(set(self.output_tensors)) != len(self.output_tensors):
            raise ValueError("Operator output tensors must be unique")
        if any(not x.strip() for x in self.input_tensors):
            raise ValueError("Input tensor IDs cannot be empty")
        if any(not x.strip() for x in self.output_tensors):
            raise ValueError("Output tensor IDs cannot be empty")


@dataclass(frozen=True)
class TensorNode:
    """One logical tensor and its producer, if internally generated."""

    tensor_id: str
    shape: Optional[tuple[int, ...]] = None
    dtype: Optional[str] = None
    producer_operator_id: Optional[str] = None
    is_external: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.tensor_id.strip():
            raise ValueError("tensor_id cannot be empty")
        if self.dtype is not None and not self.dtype.strip():
            raise ValueError("dtype cannot be empty")
        if self.shape is not None:
            if any(
                not isinstance(dim, int) or isinstance(dim, bool) or dim < 0
                for dim in self.shape
            ):
                raise ValueError("Tensor dimensions must be nonnegative integers")
        if self.producer_operator_id is not None:
            if not self.producer_operator_id.strip():
                raise ValueError("producer_operator_id cannot be empty")
        if self.is_external and self.producer_operator_id is not None:
            raise ValueError(
                "An external tensor cannot have an internal producer"
            )
        if not self.is_external and self.producer_operator_id is None:
            raise ValueError(
                "A non-external tensor must declare its producer"
            )


@dataclass(frozen=True)
class TensorDependency:
    """A producer-consumer relationship carried by a specific tensor."""

    tensor_id: str
    producer_operator_id: str
    consumer_operator_id: str
    relation: str = "data"
    tensor_region: Optional[dict[str, tuple[int, int]]] = None

    def __post_init__(self):
        for name in (
            "tensor_id",
            "producer_operator_id",
            "consumer_operator_id",
            "relation",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} cannot be empty")
        if self.producer_operator_id == self.consumer_operator_id:
            raise ValueError("An operator cannot depend on itself")
        if self.tensor_region is not None:
            for axis, bounds in self.tensor_region.items():
                if not axis.strip() or len(bounds) != 2:
                    raise ValueError("Invalid tensor region")
                start, end = bounds
                if (
                    not isinstance(start, int)
                    or isinstance(start, bool)
                    or not isinstance(end, int)
                    or isinstance(end, bool)
                    or start < 0
                    or end < start
                ):
                    raise ValueError("Invalid tensor region bounds")


@dataclass
class OperatorDependencyGraph:
    """Operator-level DAG with explicit tensor-specific dependencies."""

    workload_id: str
    operators: dict[str, OperatorNode]
    tensors: dict[str, TensorNode]
    dependencies: tuple[TensorDependency, ...]
    schema_version: str = OPERATOR_GRAPH_SCHEMA_VERSION

    def __post_init__(self):
        if not self.workload_id.strip():
            raise ValueError("workload_id cannot be empty")

    @property
    def operator_count(self) -> int:
        return len(self.operators)

    @property
    def tensor_count(self) -> int:
        return len(self.tensors)

    @property
    def external_tensor_count(self) -> int:
        return sum(tensor.is_external for tensor in self.tensors.values())

    def topological_order(self) -> tuple[str, ...]:
        """Return a valid operator topological order or reject cycles."""

        indegree = {operator_id: 0 for operator_id in self.operators}
        adjacency = {operator_id: [] for operator_id in self.operators}

        for dependency in self.dependencies:
            adjacency[dependency.producer_operator_id].append(
                dependency.consumer_operator_id
            )
            indegree[dependency.consumer_operator_id] += 1

        ready = [
            operator_id
            for operator_id, degree in indegree.items()
            if degree == 0
        ]
        order = []

        while ready:
            current = ready.pop(0)
            order.append(current)

            for consumer in adjacency[current]:
                indegree[consumer] -= 1
                if indegree[consumer] == 0:
                    ready.append(consumer)

        if len(order) != len(self.operators):
            raise ValueError("Operator dependency graph contains a cycle")

        return tuple(order)

    def validate(self) -> bool:
        """Validate tensor declarations, dependencies and DAG structure."""

        if not self.operators:
            raise ValueError("Operator graph cannot be empty")

        if set(self.operators) != {
            operator.operator_id for operator in self.operators.values()
        }:
            raise ValueError("Operator dictionary keys do not match IDs")

        if set(self.tensors) != {
            tensor.tensor_id for tensor in self.tensors.values()
        }:
            raise ValueError("Tensor dictionary keys do not match IDs")

        if not self.tensors:
            raise ValueError("Operator graph must contain tensors")

        expected_dependencies = set()

        for operator in self.operators.values():
            for tensor_id in operator.input_tensors:
                if tensor_id not in self.tensors:
                    raise ValueError(
                        f"Unknown input tensor {tensor_id} "
                        f"for operator {operator.operator_id}"
                    )

                tensor = self.tensors[tensor_id]

                if not tensor.is_external:
                    expected_dependencies.add(
                        (
                            tensor_id,
                            tensor.producer_operator_id,
                            operator.operator_id,
                            "data",
                        )
                    )

            for tensor_id in operator.output_tensors:
                if tensor_id not in self.tensors:
                    raise ValueError(
                        f"Unknown output tensor {tensor_id} "
                        f"for operator {operator.operator_id}"
                    )

                tensor = self.tensors[tensor_id]

                if tensor.is_external:
                    raise ValueError(
                        f"Operator {operator.operator_id} cannot produce "
                        f"external tensor {tensor_id}"
                    )

                if tensor.producer_operator_id != operator.operator_id:
                    raise ValueError(
                        f"Producer mismatch for tensor {tensor_id}"
                    )

        for tensor in self.tensors.values():
            if tensor.is_external:
                continue

            producer_id = tensor.producer_operator_id

            if producer_id not in self.operators:
                raise ValueError(
                    f"Unknown producer {producer_id} "
                    f"for tensor {tensor.tensor_id}"
                )

            producer = self.operators[producer_id]

            if tensor.tensor_id not in producer.output_tensors:
                raise ValueError(
                    f"Producer {producer_id} does not declare output "
                    f"{tensor.tensor_id}"
                )

        actual_dependencies = set()

        for dependency in self.dependencies:
            if dependency.relation != "data":
                raise ValueError(
                    f"Unsupported dependency relation: "
                    f"{dependency.relation}"
                )

            if dependency.tensor_id not in self.tensors:
                raise ValueError(
                    f"Unknown dependency tensor {dependency.tensor_id}"
                )

            if dependency.producer_operator_id not in self.operators:
                raise ValueError("Unknown dependency producer")

            if dependency.consumer_operator_id not in self.operators:
                raise ValueError("Unknown dependency consumer")

            tensor = self.tensors[dependency.tensor_id]

            if tensor.is_external:
                raise ValueError(
                    "External tensors cannot create internal "
                    "producer-consumer dependencies"
                )

            if tensor.producer_operator_id != dependency.producer_operator_id:
                raise ValueError("Dependency producer does not match tensor")

            consumer = self.operators[dependency.consumer_operator_id]

            if dependency.tensor_id not in consumer.input_tensors:
                raise ValueError(
                    "Dependency tensor is not declared as consumer input"
                )

            actual_dependencies.add(
                (
                    dependency.tensor_id,
                    dependency.producer_operator_id,
                    dependency.consumer_operator_id,
                    dependency.relation,
                )
            )

        if len(actual_dependencies) != len(self.dependencies):
            raise ValueError("Duplicate tensor dependencies")

        if actual_dependencies != expected_dependencies:
            missing = expected_dependencies - actual_dependencies
            extra = actual_dependencies - expected_dependencies
            raise ValueError(
                f"Dependency mismatch: missing={missing}, extra={extra}"
            )

        self.topological_order()
        return True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {
            "schema_version": self.schema_version,
            "workload_id": self.workload_id,
            "operators": [
                {
                    "operator_id": node.operator_id,
                    "operator_type": node.operator_type,
                    "input_tensors": list(node.input_tensors),
                    "output_tensors": list(node.output_tensors),
                    "attributes": node.attributes,
                }
                for node in self.operators.values()
            ],
            "tensors": [
                {
                    "tensor_id": node.tensor_id,
                    "shape": (
                        list(node.shape) if node.shape is not None else None
                    ),
                    "dtype": node.dtype,
                    "producer_operator_id": node.producer_operator_id,
                    "is_external": node.is_external,
                    "metadata": node.metadata,
                }
                for node in self.tensors.values()
            ],
            "dependencies": [
                {
                    "tensor_id": edge.tensor_id,
                    "producer_operator_id": edge.producer_operator_id,
                    "consumer_operator_id": edge.consumer_operator_id,
                    "relation": edge.relation,
                    "tensor_region": edge.tensor_region,
                }
                for edge in self.dependencies
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize the graph as JSON."""

        return json.dumps(self.to_dict(), indent=indent)


def build_operator_dependency_graph(
    *,
    workload_id: str,
    operators: Iterable[OperatorNode],
    tensors: Iterable[TensorNode],
) -> OperatorDependencyGraph:
    """Build tensor-specific dependencies from operator declarations."""

    operator_map = {}

    for operator in operators:
        if operator.operator_id in operator_map:
            raise ValueError(f"Duplicate operator ID: {operator.operator_id}")
        operator_map[operator.operator_id] = operator

    tensor_map = {}

    for tensor in tensors:
        if tensor.tensor_id in tensor_map:
            raise ValueError(f"Duplicate tensor ID: {tensor.tensor_id}")
        tensor_map[tensor.tensor_id] = tensor

    dependencies = []

    for operator in operator_map.values():
        for tensor_id in operator.input_tensors:
            if tensor_id not in tensor_map:
                raise ValueError(f"Unknown input tensor: {tensor_id}")

            tensor = tensor_map[tensor_id]

            if not tensor.is_external:
                dependencies.append(
                    TensorDependency(
                        tensor_id=tensor_id,
                        producer_operator_id=tensor.producer_operator_id,
                        consumer_operator_id=operator.operator_id,
                    )
                )

    graph = OperatorDependencyGraph(
        workload_id=workload_id,
        operators=operator_map,
        tensors=tensor_map,
        dependencies=tuple(dependencies),
    )
    graph.validate()
    return graph
