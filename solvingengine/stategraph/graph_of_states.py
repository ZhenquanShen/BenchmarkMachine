#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Time: 2023/9/20 20:13
@Author: Xiaopan Lyu
@Description:
Copyright (c) 2023 . All rights reserved.
"""
from typing import List

from solvingengine.stategraph.state.states import State


class GraphOfStates(object):
    """
    Represents the Graph of States, which describes the solving skills possessed by the solving engine.
    """

    def __init__(self) -> None:
        """
        Initializes a new Graph of States instance with empty states, roots, and leaves.
        The roots are the entry points in the graph with no predecessors.
        The leaves are the exit points in the graph with no successors.
        """
        self.states: List[State] = []
        self.roots: List[State] = []
        self.leaves: List[State] = []

    def add_state(self, state: State) -> None:
        """
        Add an state to the graph considering its predecessors and successors.
        Adjust roots and leaves based on the added state's position within the graph.

        :param state: The state to add.
        :type state: State
        """
        self.states.append(state)
        if len(self.roots) == 0:
            self.roots = [state]
            self.leaves = [state]
            assert (
                len(state.predecessors) == 0
            ), "First state should have no predecessors"
        else:
            if len(state.predecessors) == 0:
                self.roots.append(state)
            for _, predecessor in state.predecessors.items():
                if predecessor in self.leaves:
                    self.leaves.remove(predecessor)
            if len(state.successors) == 0:
                self.leaves.append(state)
