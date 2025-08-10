from .executor_base import Executor
from .taskspace_control_executor import TaskspaceControlExecutor
from .cartesian_planning_executor import CartesianPlanningExecutor


class ExecutorFactory:
    _registry = {
        "taskspace_control": TaskspaceControlExecutor,
        "cartesian_planning": CartesianPlanningExecutor,
    }

    @classmethod
    def create(cls, executor_type: str, *args, **kwargs) -> Executor:
        executor_cls = cls._registry.get(executor_type.lower())
        if executor_cls is None:
            raise ValueError(f"Unknown executor type: {executor_type}")
        return executor_cls(*args, **kwargs)
