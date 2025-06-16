"""
#!/usr/bin/env: Python3
# -*- encoding: utf-8-*-
Description:
Author: Xiaopan LYU
Date: 2023-07-15 17:00:35
LastEditTime: 2023-09-16 11:39:41
LastEditors: Xiaopan LYU
"""
import logging
from dataclasses import asdict
from typing import List, Dict, Any
from .stategraph import GOS_generator, GraphOfStates
from solvingengine.stategraph.state import SharedStateData, State
import time


class SolvingEngine(object):
    def __init__(self) -> None:

        """
        Initialize the SolvingEngine instance with the Graph of States and Shared State Data.
        """

        self.logger = logging.getLogger(self.__class__.__module__)
        self.graph: GraphOfStates = GOS_generator()
        self.shared_state_data: SharedStateData = SharedStateData()
        self.run_states: List[State] = []
        self.result: Dict[str, Any] = {}
        self.executed: bool = False
        self.elapsed_time = 0

    def reset_running_states(self):
        """
        Reset the running states to prepare for the next run.
        :return:
        """
        for state in self.run_states:
            state.reset_state()

        self.shared_state_data.clear()  # Clear the SharedStateData

        self.result.clear()
        self.executed = False

    def run(self, feed_input_ap: dict) -> None:
        """
        Run the SolvingEngine and execute the states from the Graph of States based on their readiness.
        Ensures the program is in a valid state before execution.
        :raises AssertionError: If the Graph of State has no roots.
        :raises AssertionError: If the successor of n state is not in the Graph of States.
        """
        # print("feed",feed_input_ap)
        if self.executed:
            self.reset_running_states()

        self.logger.debug("Checking that the program is in a valid state")
        assert self.graph.roots is not None, "The states graph has no root"
        self.logger.debug("The program is in a valid state")

        self.shared_state_data.__dict__.update(feed_input_ap)

        execution_queue = [self.graph.roots[0]]

        start_time = time.time()
        while len(execution_queue) > 0:
            current_state = execution_queue.pop(0)
            self.run_states.append(
                current_state
            )  # record the states that executed in solving the given problem
            current_state.decide(self.shared_state_data)  # decide
            current_state.execute(self.shared_state_data)
            if current_state.executed:
                self.logger.info("State %s executed", current_state.state_name)
                for state in current_state.run_successors:
                    # print('STATE:',state)
                    assert (
                        state in self.graph.states
                    ), "The successor of an state is not in the states graph"
                    if state not in execution_queue:
                        execution_queue.append(state)

        end_time = time.time()
        self.elapsed_time = end_time - start_time

        self.logger.info("All states executed")
        self.executed = True

        # self.result.update(asdict(self.shared_state_data))

    def get_solving_result(self, out_type="1"):

        if self.executed:
            if out_type == "1":
                return self.result

        else:
            self.logger.info(
                "The current solving engine has not completed the execution."
            )
    
    def get_sharedata(self):
        return self.shared_state_data
    
    def get_elapsed_time(self):
        return self.elapsed_time


    