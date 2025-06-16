from __future__ import annotations

import itertools
import logging
from abc import ABC, abstractmethod
from typing import List, Iterator, Dict

from .state_data import SharedStateData
from solvingengine.stategraph import transit


class State(ABC):
    """
    Abstract base class that defines the interface for all states.
    """

    _ids: Iterator[int] = itertools.count(0)

    state_name: str = ""

    def __init__(self) -> None:
        """
        Initializes a new State instance with a unique id, and empty predecessors and successors.
        """
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self.id: int = next(State._ids)
        self.predecessors: Dict[str, State] = {}
        self.successors: Dict[str, State] = {}
        self.transitors: Dict[str, transit.Transit] = {}
        self.run_predecessors: List[State] = []
        self.run_successors: List[State] = []
        self.run_transitors: List[transit.Transit] = []
        self.run_decisions: Dict[str, str] = {}
        self.decided: bool = False
        self.executed: bool = False

    def reset_state(self) -> None:
        """
        Reset the state to prepare for the next run.
        :return:
        """
        self.logger.debug("Start resetting state %s's transitors", self.state_name)
        for transitor in self.run_transitors:
            transitor.reset_transit()
        self.run_transitors.clear()
        self.logger.debug("End resetting state %s's transitors", self.state_name)

        self.run_predecessors.clear()
        self.run_successors.clear()
        self.run_decisions.clear()
        self.decided = False
        self.executed = False

    def add_predecessor(self, state: State) -> None:
        """
        Add a preceding state and update the relationships.

        :param state: The state to be set as a predecessor.
        :type state: State
        """
        if state.state_name not in self.predecessors.keys():
            self.predecessors.setdefault(state.state_name, state)
        if self.state_name not in state.successors.keys():
            state.successors.setdefault(self.state_name, self)

    def add_successor(self, state: State, transitors: List[str]) -> None:
        """
        Add a succeeding state and update the relationships.

        :param transitors:
        :param state: The state to be set as a successor.
        :type state: State
        """
        if state.state_name not in self.successors.keys():
            self.successors.setdefault(state.state_name, state)
            transit_entry = self.state_name + "To" + state.state_name
            """only one entry from one state to the other state"""
            transit_instance = getattr(transit, transit_entry)()
            transit_instance.add_transitors(transitors)
            self.transitors.setdefault(state.state_name, transit_instance)
        if self not in state.predecessors.keys():
            state.predecessors.setdefault(self.state_name, self)

    def can_be_decided(self) -> bool:
        """
        Checks if the state can be decided based on its predecessors.

        :return: True if all predecessors have been executed, False otherwise.
        :rtype: bool
        """
        return all(
            run_predecessor.executed for run_predecessor in self.run_predecessors
        )

    def can_be_executed(self) -> bool:
        """
        Checks if the state can be executed based on its decide.

        :return: True if decide have been decided, False otherwise.
        :rtype: bool
        """
        return self.decided

    def decide(self, shared_state_data: SharedStateData) -> None:
        """
        Execute the state decide, deciding which states and transits are adopted.

        :param shared_state_data: The shared state data.
        :raises AssertionError: If not all predecessors have been executed.
        """
        assert self.can_be_decided(), "Not all predecessors have been executed"
        self.logger.info("Executing state %s's decide", self.state_name)
        self._decide(shared_state_data)
        self.logger.debug("State %s executed", self.state_name)
        self.decided = True

    def add_run_transitors(self) -> None:
        """
        Add running transitors to the state and update the relationships.
        :return:
        """
        # print("self.run_decisions.items()",self.run_decisions.items())
        for nt_state_name, tst_m in self.run_decisions.items():
            # add the transit interface
            # print("nt_state_name:", nt_state_name)
            transitor = self.transitors.get(nt_state_name)
            successor_state = self.successors.get(nt_state_name)
            # print("transitor:",  transitor)
            successor_state.run_predecessors.append(self)
            # add the transiting method
            if tst_m in transitor.transitors:
                # print("tst:", tst_m)
                transitor.run_transitors.append(tst_m)
            self.run_transitors.append(transitor)
            # print("runlist:", self.run_transitors)
            self.run_successors.append(successor_state)

    def execute(self, shared_state_data: SharedStateData) -> None:
        """
        Execute the state, assuring that the state decide have been executed.

        :param shared_state_data: The shared state data.
        :raises AssertionError: If state has not been executed.
        """
        assert self.can_be_executed(), "State has not been decided"
        self.logger.info("Executing state %s", self.state_name)
        self._execute(shared_state_data)
        self.logger.debug("State %s executed", self.state_name)
        self.executed = True

    def _execute(self, shared_state_data: SharedStateData) -> None:
        """
        Executes the state transitors.
        """
        # print("self.runtran:", self.run_transitors)
        for transitor in self.run_transitors:
            # print("transitor:", transitor)
            # print("trans:", transitor.run_transitors)
            transitor.transiting(transitor.run_transitors, shared_state_data)

    @abstractmethod
    def _decide(self, shared_state_data: SharedStateData) -> None:
        """
        Abstract method for the actual execution of the state.
        This should be implemented in derived classes.

        :param shared_state_data: The shared state data.
        """
        pass


class InputAP(State):
    """
    Operation to generate thoughts.
    """

    state_name: str = "InputAP"

    def __init__(self) -> None:
        """
        Initializes a new state.
        """
        super().__init__()

    def _decide(self, shared_state_data: SharedStateData) -> None:
        """
        Executes the state decide selecting the successor states and transits.
        """
        """first, it needs a classifier """
        solver_type = ""
        if solver_type == "SP-1":
            d = {"OutputSolution": "specific_solver1"}
            self.run_decisions.update(d)
        else:
            # print("share_data:", shared_state_data)
            if any(shared_state_data.text):
                # deciding the transit interface and transiting method
                d = {"OutputSolution": "deepseek_r1"}
                # d = {"APText": "preprocessing"}
                self.run_decisions.update(d)
            if any(shared_state_data.diagram_url):
                # deciding the transit interface and transiting method
                d = {"APDiagram": "base64_img"}
                self.run_decisions.update(d)

        self.add_run_transitors()


class APText(State):
    """
    Operation to generate thoughts.
    """

    state_name: str = "APText"

    def __init__(self) -> None:
        """
        Initializes a new Generate state.
        """
        super().__init__()

    def _decide(self, shared_state_data: SharedStateData) -> None:
        """
        Executes the state decide selecting the successor states and transits.
        """
        """extracting explicit relations is necessary"""
        d = {
            # "VectorText": "bertencoder",
            # "VectorText": "graph2tree_wang_encoder",
            "VectorText": "graph2tree_li_encoder",
            # "ExplicitRelationSet": "S2_extracting",
            # "ImplicitRelationSet": "keyword_acquiring",
        }
        self.run_decisions.update(d)
        self.add_run_transitors()


class APDiagram(State):
    """
        Operation to generate thoughts.
        """

    state_name: str = "APDiagram"

    def __init__(self) -> None:
        """
            Initializes a new Generate state.
            """
        super().__init__()

    def _decide(self, shared_state_data: SharedStateData) -> None:
        """
        Executes the Generate state by generating thoughts from the predecessors.
        """
        pass


class VectorText(State):
    """
        Operation to generate thoughts.
        """

    state_name: str = "VectorText"

    def __init__(self) -> None:
        """
            Initializes a new Generate state.
            """
        super().__init__()

    def _decide(self, shared_state_data: SharedStateData) -> None:
        """
        Executes the Generate state by generating thoughts from the predecessors.
        """
        d = {
            # "OutputSolution": "sympy_solving",
            #  "EquationSystem": "scene_reasoning",
            #  "EquationSystem": "gtsdecoder",
            #  "EquationSystem": "wanggtsdecoder",
             "EquationSystem": "rnndecoder",
             }
        self.run_decisions.update(d)
        self.add_run_transitors()


class VectorDiagram(State):
    """
    Operation to generate thoughts.
    """

    state_name: str = "VectorDiagram"

    def __init__(self) -> None:
        """
        Initializes a new Generate state.
        """
        super().__init__()

    def _decide(self, shared_state_data: SharedStateData) -> None:
        """
        Executes the Generate state by generating thoughts from the predecessors.
        """
        pass


class AnnotatedText(State):
    """
    Operation to generate thoughts.
    """

    state_name: str = "AnnotatedText"

    def __init__(self) -> None:
        """
        Initializes a new Generate state.
        """
        super().__init__()

    def _decide(self, shared_state_data: SharedStateData) -> None:
        """
        Executes the Generate state by generating thoughts from the predecessors.
        """
        pass


class AnnotatedVectorText(State):
    """
    Operation to generate thoughts.
    """

    state_name: str = "AnnotatedVectorText"

    def __init__(self) -> None:
        """
        Initializes a new Generate state.
        """
        super().__init__()

    def _decide(self, shared_state_data: SharedStateData) -> None:
        """
        Executes the Generate state by generating thoughts from the predecessors.
        """
        pass


class ExplicitRelationSet(State):
    """
    Operation to generate thoughts.
    """

    state_name: str = "ExplicitRelationSet"

    def __init__(self) -> None:
        """
        Initializes a new Generate state.
        """
        super().__init__()

    def _decide(self, shared_state_data: SharedStateData) -> None:
        """
        Executes the Generate state by generating thoughts from the predecessors.
        """
        d = {"RelationSet": "fusing_ExplicitRelationSet"}
        self.run_decisions.update(d)
        self.add_run_transitors()


class ImplicitRelationSet(State):
    """
    Operation to generate thoughts.
    """

    state_name: str = "ImplicitRelationSet"

    def __init__(self) -> None:
        """
        Initializes a new Generate state.
        """
        super().__init__()

    def _decide(self, shared_state_data: SharedStateData) -> None:
        """
        Executes the Generate state by generating thoughts from the predecessors.
        """
        d = {"RelationSet": "fusing_ImplicitRelationSet"}
        self.run_decisions.update(d)
        self.add_run_transitors()


class DiagramRelationSet(State):
    """
    Operation to generate thoughts.
    """

    state_name: str = "DiagramRelationSet"

    def __init__(self) -> None:
        """
        Initializes a new Generate state.
        """
        super().__init__()

    def _decide(self, shared_state_data: SharedStateData) -> None:
        """
        Executes the Generate state by generating thoughts from the predecessors.
        """
        pass


class SceneRelationSet(State):
    """
    Operation to generate thoughts.
    """

    state_name: str = "SceneRelationSet"

    def __init__(self) -> None:
        """
        Initializes a new Generate state.
        """
        super().__init__()

    def _decide(self, shared_state_data: SharedStateData) -> None:
        """
        Executes the Generate state by generating thoughts from the predecessors.
        """
        pass


class RelationSet(State):
    """
    Operation to generate thoughts.
    """

    state_name: str = "RelationSet"

    def __init__(self) -> None:
        """
        Initializes a new Generate state.
        """
        super().__init__()

    def _decide(self, shared_state_data: SharedStateData) -> None:
        """
        Executes the Generate state by generating thoughts from the predecessors.
        """
        d = {"EquationSystem": "scene_reasoning"}
        self.run_decisions.update(d)
        self.add_run_transitors()


class EquationSystem(State):
    """
    Operation to generate thoughts.
    """

    state_name: str = "EquationSystem"

    def __init__(self) -> None:
        """
        Initializes a new Generate state.
        """
        super().__init__()

    def _decide(self, shared_state_data: SharedStateData) -> None:
        """
        Executes the Generate state by generating thoughts from the predecessors.
        """
        d = {
            "OutputSolution": "compute_expression",
            #  "OutputSolution": "sympy_solving",
             }
        self.run_decisions.update(d)
        self.add_run_transitors()


class OutputSolution(State):
    """
    Operation to generate thoughts.
    """

    state_name: str = "OutputSolution"

    def __init__(self) -> None:
        """
        Initializes a new Generate state.
        """
        super().__init__()

    def _decide(self, shared_state_data: SharedStateData) -> None:
        """
        Executes the Generate state by generating thoughts from the predecessors.
        """
        pass
