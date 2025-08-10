from abc import ABC, abstractmethod
from typing import Dict

from pyrobopath.toolpath_scheduling import MultiAgentToolpathSchedule
from pyrobopath_ros.agent_context import AgentContext


class Executor(ABC):
    def __init__(self, contexts: Dict[str, AgentContext]):
        self._contexts = contexts

    @abstractmethod
    def execute(self, sched: MultiAgentToolpathSchedule): ...

    @abstractmethod
    def move_home(self, tf=2.0): ...

    @abstractmethod
    def shutdown(self): ...
