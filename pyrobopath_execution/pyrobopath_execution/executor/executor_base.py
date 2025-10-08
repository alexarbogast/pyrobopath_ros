from abc import ABC, abstractmethod
from typing import Dict

from rclpy.node import Node

from pyrobopath.toolpath_scheduling import MultiAgentToolpathSchedule
from pyrobopath_execution.agent_context import AgentContext


class Executor(ABC):
    def __init__(self, node: Node, contexts: Dict[str, AgentContext]):
        self._node = node
        self._contexts = contexts

    @abstractmethod
    def execute(self, sched: MultiAgentToolpathSchedule): ...

    @abstractmethod
    def move_home(self, tf=2.0): ...

    @abstractmethod
    def shutdown(self): ...
