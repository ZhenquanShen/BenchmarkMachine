#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Time: 2023/9/21 9:34
@Author: Xiaopan Lyu
@Description:
Copyright (c) 2023 . All rights reserved.
"""
from solvingengine.stategraph import state
from .graph_of_states import GraphOfStates
from solvingengine.config.config import mimic_cfg


def GOS_generator() -> GraphOfStates:
    """
    Generates the Graph of States for the solving engine.

    :return: Graph of States
    :rtype: GraphOfStates
    """
    state_graph = GraphOfStates()

    states = mimic_cfg.state_graph
    state_instances = {}

    for st_k, st_v in states.items():

        """create or acquire current state instance"""
        if st_k not in state_instances.keys():
            cur_st_instance = getattr(state, st_k)()
            state_instances.setdefault(st_k, cur_st_instance)
        else:
            cur_st_instance = state_instances.get(st_k)

        """add current state predecessors"""
        if st_v["predecessors"] is not None:
            for p_st in st_v["predecessors"]:
                """ensure each state only has one instance"""
                if p_st not in state_instances.keys():
                    p_st_instance = getattr(state, p_st)()
                    state_instances.setdefault(p_st, p_st_instance)
                else:
                    p_st_instance = state_instances.get(p_st)
                cur_st_instance.add_predecessor(p_st_instance)

        """add current state successors"""
        if st_v["successors"] is not None:
            
            for s_st, transitors in st_v["successors"].items():
                """ensure each state only has one instance"""
                if s_st not in state_instances.keys():
                    s_st_instance = getattr(state, s_st)()
                    state_instances.setdefault(s_st, s_st_instance)
                else:
                    s_st_instance = state_instances.get(s_st)
                # print("transitor:", transitors)
                cur_st_instance.add_successor(s_st_instance, transitors)

        state_graph.add_state(cur_st_instance)

    return state_graph
