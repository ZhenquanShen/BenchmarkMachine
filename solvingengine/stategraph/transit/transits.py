"""
#!/usr/bin/env: Python3
# -*- encoding: utf-8-*-
Description: Docstring
Author: Xiaopan Lyu
Date: 2022-12-18 20:07:55
LastEditTime: 2023-07-29 17:08:13
LastEditors: Xiaopan LYU
"""
from __future__ import annotations

import inspect
import itertools
import logging
import re
import torch
import copy
from abc import ABC, abstractmethod
from typing import List, Iterator

import sympy as sp
from ltp import LTP
from sympy.parsing.sympy_parser import parse_expr
from data_utils.tree import Tree
from solvingengine.stategraph import state
from solvingengine.stategraph.state import SharedStateData
from solvingengine.stategraph.transit import add_iqr as iqr_adder
from solvingengine.stategraph.transit import find_iqr as iqr_finder
from solvingengine.stategraph.transit import knowledge as know
from solvingengine.stategraph.transit import scene_transit as scene_tfm
from solvingengine.stategraph.transit import unit_transit as unit_tfm
from solvingengine.config import mimic_cfg
from data_utils.common_func import CommonFunction as cf
from data_utils.file_helper import FileHelper as fh
from solvingengine.stategraph.transit.encoder_decoder import models, evaluate
from solvingengine.stategraph.transit.model_config import Bert as bt
from solvingengine.stategraph.transit.model_config import TreeDecoder as gts
from solvingengine.stategraph.transit.model_config import Local_Model_Li as opt_model_li
from solvingengine.stategraph.transit.model_config import RnnDecoder as rd
from solvingengine.stategraph.transit.model_config import opt_file
import data_utils.expressions_transfer as exp_transf
from data_utils.read_dataset import input_lang
from data_utils.read_dataset import DataLoader
import ollama
from openai import OpenAI
import sys 
sys.path.append('D:/code/benchmarkmachine/solvingengine/stategraph/transit/GraphConstruction')  


USE_CUDA = torch.cuda.is_available()
# USE_CUDA = False


class Transit(ABC):
    """
    Abstract base class that defines the interface for all transits.
    """

    _ids: Iterator[int] = itertools.count(0)

    transit_name: str = ""

    def __init__(self) -> None:
        """
        Initializes a new Transit instance with a unique id, empty transitors and empty run_transitors.
        """
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self.id: int = next(Transit._ids)
        self.transitors: List[str] = []
        self.run_transitors: List[str] = []
        self.checked: bool = False
        self.transited: bool = False

    def reset_transit(self):
        """
        Reset the transit to prepare for the next run.
        :return:
        """
        self.logger.debug("Start resetting the %s", self.transit_name)
        self.run_transitors.clear()
        self.transited = False
        self.logger.debug("End resetting the %s", self.transit_name)

    def add_transitors(self, transitors: List[str]) -> None:
        """
        Add a succeeding state and update the relationships.

        :param transitors:
        :param state: The state to be set as a successor.
        :type state: State
        """
        self.check(transitors)
        self.transitors = transitors

    def check(self, transitors: List[str]) -> None:
        """
        Check whether all transitors have been defined.

        :param shared_state_data: The shared state data.
        :raises AssertionError: If not all predecessors have been executed.
        """
        assert len(transitors), "Do not provide transitors"
        self.logger.info(
            "Check  whether all transitors of %s have been defined", self.transit_name
        )
        self._check(transitors)
        self.logger.debug("Tansitors of %s have been checked", self.transit_name)
        self.checked = True

    def transiting(
        self, transitors: List[str], shared_state_data: SharedStateData
    ) -> None:
        """
        Execute all the transitors.

        :param transitors: The given transitors.
        :param shared_state_data: The shared state data.
        :raises AssertionError: If not all predecessors have been executed.
        """
        assert len(transitors), "Do not provide transitors"
        self.logger.info("Execute  all transitors of %s", self.transit_name)
        self._transiting(transitors, shared_state_data)
        self.logger.debug("All transitors of %s have been executed", self.transit_name)
        self.transited = True

    def _transiting(
        self, transitors: List[str], shared_state_data: SharedStateData
    ) -> None:
        """
        Execute the transitors, where the transitors are  selected by the state decide.

        :param transitors: The given transitors.
        :param shared_state_data: The shared state data.
        """
        for t_name in transitors:
            self.logger.debug("Start execution of the transit %s", t_name)
            getattr(self, t_name)(shared_state_data)  # 调用func1
            self.logger.debug("End execution of transit %s", t_name)

    @abstractmethod
    def _check(self, transitors: List[str]) -> None:
        """
        Abstract method for the actual execution of the state.
        This should be implemented in derived classes.

        :param shared_state_data: The shared state data.
        """
        pass


class InputAPToOutputSolution(Transit):
    """
    Transit from InputAP to OutputSolution, here include all specific solver
    """

    transit_name: str = "InputAPToOutputSolution"

    def __init__(self):
        super().__init__()

    def _check(self, transitors: List[str]) -> None:
        """
        Executes the transit check, checking whether all transitors have been defined.
        :param transitors:
        :raises AssertionError: If operation has no predecessors.
        """
        functions = [
            fn_name
            for fn_name, _ in inspect.getmembers(
                InputAPToOutputSolution, predicate=inspect.isfunction
            )
        ]
        for t in transitors:
            if t not in functions:
                raise ValueError("transitors %s has not defined".format(t))

    def specific_solver1(self, shared_state_data: SharedStateData):
        # shared_state_data.ap_text = shared_state_data.text
        print("specific solver 1: please define this function")

    def deepseek_r1(self, shared_state_data: SharedStateData):
        # shared_state_data.ap_text = shared_state_data.text
        # response = ollama.generate( 
        # model="deepseek-r1:70b",
        # prompt=shared_state_data.text,
        # options={"temperature": 0.7, "num_gpu_layers": 50})
        print("id:",shared_state_data.id)
        client = OpenAI(api_key="sk-f3e8f9861e1f471ea0afc109527156ce", base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "user", "content": shared_state_data.text + '给出计算求解的表达式和最终答案，表达式在equation：后， 答案在answer：后'},
        ],
        stream=False
        )
        full_response = response.choices[0].message.content 
        
        with open("deepseek_response.txt",  "a", encoding="utf-8") as file:
            file.write(f"{shared_state_data.id} {full_response}\n")  # `\n`表示换行 

        shared_state_data.equation_solution = float(re.search(r'answer:\s*(\d+)',  full_response).group(1))
        shared_state_data.equation_system = re.search(r'equation:\s*(.*?)\s*\n',  full_response).group(1).strip()


class InputAPToAPText(Transit):
    """
    Transit from InputAP to APText
    """

    transit_name: str = "InputAPToAPText"

    def __init__(self):
        super().__init__()

    def _check(self, transitors: List[str]) -> None:
        """
        Executes the transit check, checking whether all transitors have been defined.
        :param transitors:
        :raises AssertionError: If operation has no predecessors.
        """
        functions = [
            fn_name
            for fn_name, _ in inspect.getmembers(
                InputAPToAPText, predicate=inspect.isfunction
            )
        ]
        for t in transitors:
            if t not in functions:
                raise ValueError("transitors %s has not defined".format(t))

    def preprocessing(self, shared_state_data: SharedStateData):
        shared_state_data.ap_text = shared_state_data.text
        # print("shared_state_data.text:",shared_state_data.text)
        # print("id:",shared_state_data.id)


class InputAPToAPDiagram(Transit):
    """
    Transit from InputAP to APDiagram
    """

    transit_name: str = "InputAPToAPDiagram"

    def __init__(self):
        super().__init__()

    def _check(self, transitors: List[str]) -> None:
        """
        Executes the transit check, checking whether all transitors have been defined.
        :param transitors:
        :raises AssertionError: If operation has no predecessors.
        """
        functions = [
            fn_name
            for fn_name, _ in inspect.getmembers(
                InputAPToAPDiagram, predicate=inspect.isfunction
            )
        ]
        for t in transitors:
            if t not in functions:
                raise ValueError("transitors %s has not defined".format(t))

    def base64_img(self, shared_state_data: SharedStateData):

        # print(shared_state_data.diagram_url)
        pass


class APTextToExplicitRelationSet(Transit):
    """
    Transit from APText to ExplicitRelationSet
    """

    transit_name: str = "APTextToExplicitRelationSet"

    def __init__(self):
        super().__init__()
        self.nltp = LTP(mimic_cfg.transits.nltp.model_path)

    def _check(self, transitors: List[str]) -> None:
        """
        Executes the transit check, checking whether all transitors have been defined.
        :param transitors:
        :raises AssertionError: If operation has no predecessors.
        """
        functions = [
            fn_name
            for fn_name, _ in inspect.getmembers(
                APTextToExplicitRelationSet, predicate=inspect.isfunction
            )
        ]
        for t in transitors:
            if t not in functions:
                raise ValueError("transitors %s has not defined".format(t))

    def S2_extracting(self, shared_state_data: SharedStateData) -> None:
        self.nltp_annotation(shared_state_data)
        s2_model_pool = know.s2_model_pool
        # get problem token sequence
        vcb = fh.read_json_file(mimic_cfg.transits.s2model.vocab)
        keywords = vcb.get("keywords")
        pos = vcb.get("pos")
        prob_token_seq = shared_state_data.segmented_tokens
        word_tag = shared_state_data.pos_tagged_tokens
        # S2 model matching
        sentences_ranges = shared_state_data.sentence_seg_ranges
        matched_results = []
        for sent_range in sentences_ranges:
            s2_matched_state = {}
            sent_tagged_tokens = word_tag[sent_range[0] : sent_range[1]]
            sent_keywords = [w for w, t in sent_tagged_tokens if w in keywords]
            token_seq = []
            token_position_seq = []
            for idx, token in zip(
                range(0, len(sent_tagged_tokens)), sent_tagged_tokens
            ):
                if token[0] in sent_keywords:
                    token_seq.append(token[0])
                    token_position_seq.append(sent_range[0] + idx)
                elif token[1] in pos:
                    token_seq.append(token[1])
                    token_position_seq.append(sent_range[0] + idx)
            """filter s2 model by keywords"""
            sent_keywords_seq = "".join(sent_keywords)
            s2_model_keywords_seqs = [
                v
                for v in list(s2_model_pool.keyword_index.keys())
                if v in sent_keywords_seq
            ]
            s2_model_keywords_seqs = cf.list_sort_by_len(s2_model_keywords_seqs)
            for cur_keywords_seq in s2_model_keywords_seqs:
                is_matched = False
                s2_models = s2_model_pool.keyword_index.get(cur_keywords_seq)
                if s2_models is not None:
                    for m in s2_models:
                        pattern_list = m.pattern.split(",")
                        """filter s2 model by length"""
                        if len(pattern_list) <= len(token_seq):
                            head = pattern_list[0]
                            rear = pattern_list[-1]
                            head_index = 0
                            rear_index = 0
                            """filter sub token sequence by model's first and last token"""
                            for head_id in range(0, len(token_seq)):
                                head_index = head_id
                                if head == token_seq[head_id]:
                                    break
                            for rear_id in range(len(token_seq) - 1, -1, -1):
                                rear_index = rear_id
                                if rear == token_seq[rear_id]:
                                    break
                            if rear_index - head_index + 1 >= len(pattern_list):
                                token_seq_sub = token_seq[head_index : rear_index + 1]
                                token_position_seq_sub = token_position_seq[
                                    head_index : rear_index + 1
                                ]
                                token_with_position = [
                                    (token, position)
                                    for token, position in zip(
                                        token_seq_sub, token_position_seq_sub
                                    )
                                ]
                                # get combinations of sub token sequence
                                from itertools import combinations

                                combs = [
                                    list(x)
                                    for x in combinations(
                                        token_with_position, len(pattern_list)
                                    )
                                ]
                                for comb in combs:
                                    comb_ptn_seq = ",".join([w for w, _ in comb])
                                    model_ptn_seq = ",".join(pattern_list)
                                    if model_ptn_seq == comb_ptn_seq:  # matched
                                        s2_matched_state.setdefault(
                                            model_ptn_seq, [m, comb]
                                        )
                                        is_matched = True
                if is_matched is True:
                    break
            # check the matched s2_model, remove the conflict results
            prior_list = fh.read_json_file(mimic_cfg.transits.s2model.priority)
            matched_pth_list = list(s2_matched_state.keys())
            if len(matched_pth_list) > 0:
                rm_fact = []
                for prior in prior_list:
                    for i in range(len(prior)):
                        if prior[i] in matched_pth_list:
                            rm_fact.extend(prior[i + 1 : len(prior)])
                            break
                for rm in rm_fact:
                    if rm in matched_pth_list:
                        del s2_matched_state[rm]
            # parsing the matched relations
            sent_text = "".join(prob_token_seq[sent_range[0] : sent_range[1]])
            for _, match in s2_matched_state.items():
                import copy

                relation = copy.deepcopy(match[0].relation_template)
                var_slot = match[0].var_slot_index
                tokens = [(prob_token_seq[idx], idx) for _, idx in match[1]]
                var_entity = copy.deepcopy(var_slot)
                for k, v in var_slot.items():
                    if v.isnumeric():
                        var_entity.update({k: tokens[int(v)][0]})
                        relation = relation.replace(k, tokens[int(v)][0])
                    else:
                        relation = relation.replace(k, v)
                d = {
                    "relation": relation,
                    "var_entity": var_entity,
                    "sent_text": sent_text,
                    "matched_token": tokens,
                    "matched_model": match[0].get_state(),
                }
                expl_relation_instance = state.ExplRelation()
                expl_relation_instance.__dict__.update(d)
                matched_results.append(expl_relation_instance)
        shared_state_data.expl_relations = matched_results

    def nltp_annotation(self, shared_state_data: SharedStateData):
        # from ltp import LTP
        # ltp = LTP(mimic_cfg.transits.nltp.model_path)
        pb_text = shared_state_data.text
        segs, hiddens = self.nltp.seg([pb_text])
        pos = self.nltp.pos(hiddens)
        text_tagged = [(word, tag) for word, tag in zip(segs[0], pos[0])]
        new_text_tagged = []
        # correct the pos tag
        for word, tag in text_tagged:
            new_tag = tag
            if tag in ["nd", "nh", "ni", "nl", "ns", "nt", "nz"]:
                new_tag = "n"
            elif str(tag) == "wp":
                if word in [",", ".", "?", "，", "。", "？", "(", ")", "（", "）"]:
                    new_tag = "w"
                else:
                    new_tag = "x"  # 除分句符号以外的其它符号
            new_text_tagged.append((word, new_tag))
        # get sentence range
        sentence_seg_ranges = []
        i = 0
        for j in range(len(new_text_tagged)):
            if new_text_tagged[j][1] == "w":
                sentence_seg_ranges.append([i, j])
                i = j + 1
        shared_state_data.segmented_tokens = segs[0]
        shared_state_data.sentence_seg_ranges = sentence_seg_ranges
        shared_state_data.pos_tagged_tokens = new_text_tagged


class APTextToImplicitRelationSet(Transit):
    """
    Transit from APText to ImplicitRelationSet
    """

    transit_name: str = "APTextToImplicitRelationSet"

    def __init__(self):
        super().__init__()

    def _check(self, transitors: List[str]) -> None:
        """
        Executes the transit check, checking whether all transitors have been defined.
        :param transitors:
        :raises AssertionError: If operation has no predecessors.
        """
        functions = [
            fn_name
            for fn_name, _ in inspect.getmembers(
                APTextToImplicitRelationSet, predicate=inspect.isfunction
            )
        ]
        for t in transitors:
            if t not in functions:
                raise ValueError("transitors %s has not defined".format(t))

    def keyword_acquiring(self, shared_state_data: SharedStateData) -> None:
        text = shared_state_data.ap_text
        iqr_cls_label = iqr_finder.ProbClassify().find_iqr_class(text)
        p_iqr = []
        if len(iqr_cls_label) > 0:
            word_tokens = shared_state_data.segmented_tokens
            p_iqr = iqr_adder.IQRAcquire().get_iqr_result(iqr_cls_label, word_tokens)
        shared_state_data.impl_relations = p_iqr


class APDiagramToDiagramRelationSet(Transit):
    """
    Transit from APDiagram to DiagramRelationSet
    """

    transit_name: str = "APDiagramToDiagramRelationSet"

    def __init__(self):
        super().__init__()

    def _check(self, transitors: List[str]) -> None:
        """
        Executes the transit check, checking whether all transitors have been defined.
        :param transitors:
        :raises AssertionError: If operation has no predecessors.
        """
        functions = [
            fn_name
            for fn_name, _ in inspect.getmembers(
                APDiagramToDiagramRelationSet, predicate=inspect.isfunction
            )
        ]
        for t in transitors:
            if t not in functions:
                raise ValueError("transitors %s has not defined".format(t))
            
    def vector_diagramet_extracting(self, shared_state_data: SharedStateData) -> None:
        # convert explicit_relation_set into dict list
        # print(shared_state_data.expl_relations)
        pass

    def copy_diagram_relation(self, shared_state_data: SharedStateData) -> None:
        # convert explicit_relation_set into dict list
        # print(shared_state_data.expl_relations)
        pass


class ExplicitRelationSetToRelationSet(Transit):
    """
    Transit from ExplicitRelationSet to RelationSet
    """

    transit_name: str = "ExplicitRelationSetToRelationSet"

    def __init__(self):
        super().__init__()

    def _check(self, transitors: List[str]) -> None:
        """
        Executes the transit check, checking whether all transitors have been defined.
        :param transitors:
        :raises AssertionError: If operation has no predecessors.
        """
        functions = [
            fn_name
            for fn_name, _ in inspect.getmembers(
                ExplicitRelationSetToRelationSet, predicate=inspect.isfunction
            )
        ]
        for t in transitors:
            if t not in functions:
                raise ValueError("transitors %s has not defined".format(t))

    def fusing_ExplicitRelationSet(self, shared_state_data: SharedStateData) -> None:
        # convert explicit_relation_set into dict list
        # print(shared_state_data.expl_relations)
        pass


class ImplicitRelationSetToRelationSet(Transit):
    """
    Transit from ImplicitRelationSet to RelationSet
    """

    transit_name: str = "ImplicitRelationSetToRelationSet"

    def __init__(self):
        super().__init__()

    def _check(self, transitors: List[str]) -> None:
        """
        Executes the transit check, checking whether all transitors have been defined.
        :param transitors:
        :raises AssertionError: If operation has no predecessors.
        """
        functions = [
            fn_name
            for fn_name, _ in inspect.getmembers(
                ImplicitRelationSetToRelationSet, predicate=inspect.isfunction
            )
        ]
        for t in transitors:
            if t not in functions:
                raise ValueError("transitors %s has not defined".format(t))

    def fusing_ImplicitRelationSet(self, shared_state_data: SharedStateData) -> None:
        # print(shared_state_data.impl_relations)
        pass


class DiagramRelationSetToRelationSet(Transit):
    """
    Transit from DiagramRelationSet to RelationSet
    """

    transit_name: str = "DiagramRelationSetToRelationSet"

    def __init__(self):
        super().__init__()

    def _check(self, transitors: List[str]) -> None:
        """
        Executes the transit check, checking whether all transitors have been defined.
        :param transitors:
        :raises AssertionError: If operation has no predecessors.
        """
        functions = [
            fn_name
            for fn_name, _ in inspect.getmembers(
                DiagramRelationSetToRelationSet, predicate=inspect.isfunction
            )
        ]
        for t in transitors:
            if t not in functions:
                raise ValueError("transitors %s has not defined".format(t))

    def fusing_DiagramRelationSet(self, shared_state_data: SharedStateData) -> None:
        r = shared_state_data.diag_relations


class RelationSetToEquationSystem(Transit):
    """
    Transit from RelationSet to EquationSystem
    """

    transit_name: str = "RelationSetToEquationSystem"

    def __init__(self):
        super().__init__()

    def _check(self, transitors: List[str]) -> None:
        """
        Executes the transit check, checking whether all transitors have been defined.
        :param transitors:
        :raises AssertionError: If operation has no predecessors.
        """
        functions = [
            fn_name
            for fn_name, _ in inspect.getmembers(
                RelationSetToEquationSystem, predicate=inspect.isfunction
            )
        ]
        for t in transitors:
            if t not in functions:
                raise ValueError("transitors %s has not defined".format(t))

    def scene_reasoning(self, shared_state_data: SharedStateData) -> None:
        arith_scene = scene_tfm.ArithmeticScene()
        scenes = arith_scene.identify_scene_type(shared_state_data)
        # considering that only one arithmetic scene in the problem
        r = []
        if len(scenes) == 1:
            r = arith_scene.scene_reasoning(scenes[0], shared_state_data)
        shared_state_data.equation_system = r

    def diagram_reasoning(self, shared_state_data: SharedStateData) -> None:
        r3 = shared_state_data.diag_relations
        shared_state_data.equation_system = r3


class EquationSystemToOutputSolution(Transit):
    """
    Transit from EquationSystem to OutputSolution
    """

    transit_name: str = "EquationSystemToOutputSolution"

    def __init__(self):
        super().__init__()

    def _check(self, transitors: List[str]) -> None:
        """
        Executes the transit check, checking whether all transitors have been defined.
        :param transitors:
        :raises AssertionError: If operation has no predecessors.
        """
        functions = [
            fn_name
            for fn_name, _ in inspect.getmembers(
                EquationSystemToOutputSolution, predicate=inspect.isfunction
            )
        ]
        for t in transitors:
            if t not in functions:
                raise ValueError("transitors %s has not defined".format(t))

    def diagram_sympy_solving(self, shared_state_data: SharedStateData) -> None:
        # convert explicit_relation_set into dict list
        # print(shared_state_data.expl_relations)
        pass

    def sympy_solving(self, shared_state_data: SharedStateData) -> None:
        
        equation_system = shared_state_data.equation_system
        # print("eq:",equation_system)
        symbols_list = ["X", "Y", "Z", "H", "A", "B", "C", "D", "E", "F", "G"]
        expressions_list = []
        variables = []
        rp_unit = []
        # find all units
        qd = unit_tfm.UNIT_TYPES
        for q in qd:
            rp_unit.extend(q.keys())
        for eq in equation_system:
            # eq = eq.replace(':', '*')
            # convert equation format,for example:a+b=c+d => a+b-(c+d)=0
            # print("eq:",eq)
            relation = eq.split("=")
            expression = relation[0] + "-" + "(" + relation[1] + ")"
            parameters = re.split(
                "[\+\-\*\/=\(\)]", expression
            )  # find all parameters, split by +,-,*,/,=
            parameters = [x for x in parameters if x != ""]
            for x in parameters:
                if x in rp_unit:  # replace unit with 1
                    expression = expression.replace(x, "1")
                elif ":" in x:
                    n = str(x).split(":")
                    n = list(map(float, n))  # 把list内的元素变成float型
                    if len(n) == 2:  # process scale case
                        r = n[1] / n[0]
                        r = cf.sci2str(r)
                        expression = expression.replace(x, r)
                elif not cf.is_number(x):
                    variables.append(x)
            expressions_list.append(expression)
        variables = cf.list_remove_duplicates(variables)
        # variables sorted by its length, must be sorted in descending order.
        variables = cf.list_sort_by_len(variables, reverse=True)
        variable_symbol = []
        symbol_variable_dict = {}
        variable_symbol_dict = {}
        for var, symbol in zip(variables, symbols_list):
            symbol_variable_dict.setdefault(symbol, var)
            variable_symbol_dict.setdefault(var, symbol)
            variable_symbol.append(symbol)
            expressions_list = [
                expressions.replace(var, symbol) for expressions in expressions_list
            ]
        variable_symbol = sp.symbols(variable_symbol)  # variables symbolic
        result = {}
        try:
            expression_symbols = [
                sp.Eq(parse_expr(expression, evaluate=False), 0)
                for expression in expressions_list
            ]
            result = sp.solve(expression_symbols, variable_symbol, dict=True)
        except Exception as e:
            print(e)
        # ret_dict = {}
        # if len(result) > 0:
        #     for k, v in zip(variables_symbols, result):
        #         ret_dict.setdefault(k, v)
        # parse solution to org variables
        parse_result = {}
        if len(result) > 0:
            if isinstance(result, dict):
                # if the equation can be solved, it will generate a dict result
                for k, v in result.items():
                    parse_result.setdefault(symbol_variable_dict[str(k)], str(v))
            elif isinstance(result, list):
                for k, v in result[0].items():
                    parse_result.setdefault(symbol_variable_dict[str(k)], str(v))
        shared_state_data.equation_solution = parse_result

    
    def compute_expression(self, shared_state_data: SharedStateData) -> None:
        
        try:
            st = list()
            operators = ["+", "-", "^", "*", "/", ".", ")"]
            shared_state_data.equation_system = copy.deepcopy(shared_state_data.equation_system)
            shared_state_data.equation_system.reverse()
            # print("equation_system:", shared_state_data.equation_system)
            for p in shared_state_data.equation_system:
                if p not in operators:
                    pos = re.search("\d+\(", p)
                    if pos:
                        st.append(eval(p[pos.start(): pos.end() - 1] + "+" + p[pos.end() - 1:]))
                    elif p[-1] == "%":
                        st.append(float(p[:-1]) / 100)
                    else:
                        
                        st.append(eval(p))
                      
                elif p == "+" and len(st) > 1:
                    a = st.pop()
                    b = st.pop()
                    st.append(a + b)
                elif p == "*" and len(st) > 1:
                    a = st.pop()
                    b = st.pop()
                    st.append(a * b)
                elif p == "/" and len(st) > 1:
                    a = st.pop()
                    b = st.pop()
                    if b == 0:
                        return None
                    st.append(a / b)
                elif p == "-" and len(st) > 1:
                    a = st.pop()
                    b = st.pop()
                    st.append(a - b)
                elif p == "^" and len(st) > 1:
                    a = st.pop()
                    b = st.pop()
                    if float(eval(b)) != 2.0 or float(eval(b)) != 3.0:
                        return None
                    st.append(a ** b)
                else:
                    return None
            if len(st) == 1:
                shared_state_data.equation_solution = st.pop()
                # print("equa:", shared_state_data.equation_solution)
            return None
        except:
            return None



class APTextToVectorText(Transit):
    """
    Transit from APText to VectorText
    """

    transit_name: str = "APTextToVectorText"

    # def __init__(self, embedding_size, hidden_size, n_layers):
    def __init__(self):
        super().__init__()
        # self.config = BertConfig.from_pretrained('bert-base-uncased') 
        # self.encoder = BertModel(self.config)  

    def _check(self, transitors: List[str]) -> None:
        """
        Executes the transit check, checking whether all transitors have been defined.
        :param transitors:
        :raises AssertionError: If operation has no predecessors.
        """
        functions = [
            fn_name
            for fn_name, _ in inspect.getmembers(
                APTextToVectorText, predicate=inspect.isfunction
            )
        ]
        for t in transitors:
            if t not in functions:
                raise ValueError("transitors %s has not defined".format(t))
            
    def bertencoder(self, shared_state_data: SharedStateData) -> None:
        
        self.encoder = models.EncoderSeq(bt, opt_file, shared_state_data.output_lang.vocab_size)
        # self.encoder = models.EncoderSeq(embedding_size=embedding_size, hidden_size=hidden_size, n_layers=n_layers)

        pretrained_state_dict = torch.load('D:/code/benchmarkmachine/solvingengine/stategraph/transit/pretrained_model/encoder')
        filtered_state_dict = {k: v for k, v in pretrained_state_dict.items() if k.startswith('bert.')}
        new_state_dict = {}
        for k, v in filtered_state_dict.items():
            new_key = k.replace('bert.', '')
            new_state_dict[new_key] = v
        self.encoder.load_state_dict(new_state_dict, strict=False)
        self.encoder.eval()
        if USE_CUDA:
            self.encoder.cuda()
      
        encoder_outputs, problem_output = self.encoder(shared_state_data.pair[7])
        shared_state_data.vector_text = encoder_outputs
        shared_state_data.vector_sentense_text = problem_output
        # print("shared_state_data.vector_sentense_text:",shared_state_data.vector_sentense_text)


    def graph2tree_wang_encoder(self, shared_state_data: SharedStateData) -> None:
        
        batch_graph = DataLoader.get_single_example_graph(shared_state_data.pair[0], shared_state_data.pair[1], shared_state_data.pair[7], shared_state_data.pair[4], shared_state_data.pair[5])
        
        batch_graph = torch.LongTensor(batch_graph)
        
        # self.encoder = models.EncoderSeq(opt_model_wang, opt_file, input_lang.vocab_size)
        self.encoder = models.wangEncoderSeq(input_size=1771, embedding_size=128, hidden_size=512)
        
        pretrained_state_dict = torch.load('D:/code/benchmarkmachine/solvingengine/stategraph/transit/pretrained_model/encoder') 
        self.encoder.load_state_dict(pretrained_state_dict, strict=False)
        self.encoder.eval()
        if USE_CUDA:
            self.encoder.cuda()
      
        # input_var = shared_state_data.seq_batch["input_cell"]
        input_var = torch.LongTensor(shared_state_data.pair[0]).unsqueeze(1)
        # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
        # input_length = len(input_var)
        input_length = shared_state_data.pair[1]
        
        # input_var = torch.LongTensor(input_var).unsqueeze(1)
        # input_graph = torch.LongTensor([shared_state_data.seq_batch["graph_wang"]])

        if USE_CUDA:
            input_var = input_var.cuda()
            # input_graph = input_graph.cuda()
            batch_graph = batch_graph.cuda()

       
        # graph_embedding, problem_output = self.encoder(input_var, [input_length], input_graph)
        encoder_outputs, problem_output = self.encoder(input_var, [input_length], batch_graph)
        # encoder_outputs, problem_output = self.encoder.graph2tree_wang(shared_state_data.pair[7], input_length, graph_batch)
        # shared_state_data.vector_text = graph_embedding
        shared_state_data.vector_text = encoder_outputs
        
        shared_state_data.vector_sentense_text = problem_output

    def graph2tree_li_encoder(self, shared_state_data: SharedStateData) -> None:
        print("id:",shared_state_data.id)
        self.encoder = models.GraphEncoder(opt_model_li, opt_file, input_lang.vocab_size, using_gpu=True)
        pretrained_state_dict = torch.load('D:/code/benchmarkmachine/solvingengine/stategraph/transit/GraphConstruction/model_biGraphSAGE_BreadthTree') 
        encoder_state_dict = {}
        for key, value in pretrained_state_dict.items(): 
            if key.startswith("encoder_0"):   # 筛选encoder_0前缀的键 
                encoder_state_dict["embedding.weight"] = value.embedding.weight
                encoder_state_dict['fw_aggregator_0.fc_x.weight'] = value.fw_aggregator_0.fc_x.weight
                encoder_state_dict['fw_aggregator_0.fc_x.bias'] = value.fw_aggregator_0.fc_x.bias
                encoder_state_dict['fw_aggregator_1.fc_x.weight'] = value.fw_aggregator_1.fc_x.weight
                encoder_state_dict['fw_aggregator_1.fc_x.bias'] = value.fw_aggregator_1.fc_x.bias
                encoder_state_dict['fw_aggregator_2.fc_x.weight'] = value.fw_aggregator_2.fc_x.weight
                encoder_state_dict['fw_aggregator_2.fc_x.bias'] = value.fw_aggregator_2.fc_x.bias
                encoder_state_dict['fw_aggregator_3.fc_x.weight'] = value.fw_aggregator_3.fc_x.weight
                encoder_state_dict['fw_aggregator_3.fc_x.bias'] = value.fw_aggregator_3.fc_x.bias
                encoder_state_dict['fw_aggregator_4.fc_x.weight'] = value.fw_aggregator_4.fc_x.weight
                encoder_state_dict['fw_aggregator_4.fc_x.bias'] = value.fw_aggregator_4.fc_x.bias
                encoder_state_dict['fw_aggregator_5.fc_x.weight'] = value.fw_aggregator_5.fc_x.weight
                encoder_state_dict['fw_aggregator_5.fc_x.bias'] = value.fw_aggregator_5.fc_x.bias
                encoder_state_dict['fw_aggregator_6.fc_x.weight'] = value.fw_aggregator_6.fc_x.weight
                encoder_state_dict['fw_aggregator_6.fc_x.bias'] = value.fw_aggregator_6.fc_x.bias          
                encoder_state_dict['bw_aggregator_0.fc_x.weight'] = value.bw_aggregator_0.fc_x.weight
                encoder_state_dict['bw_aggregator_0.fc_x.bias'] = value.bw_aggregator_0.fc_x.bias
                encoder_state_dict['bw_aggregator_1.fc_x.weight'] = value.bw_aggregator_1.fc_x.weight
                encoder_state_dict['bw_aggregator_1.fc_x.bias'] = value.bw_aggregator_1.fc_x.bias
                encoder_state_dict['bw_aggregator_2.fc_x.weight'] = value.bw_aggregator_2.fc_x.weight
                encoder_state_dict['bw_aggregator_2.fc_x.bias'] = value.bw_aggregator_2.fc_x.bias
                encoder_state_dict['bw_aggregator_3.fc_x.weight'] = value.bw_aggregator_3.fc_x.weight
                encoder_state_dict['bw_aggregator_3.fc_x.bias'] = value.bw_aggregator_3.fc_x.bias
                encoder_state_dict['bw_aggregator_4.fc_x.weight'] = value.bw_aggregator_4.fc_x.weight
                encoder_state_dict['bw_aggregator_4.fc_x.bias'] = value.bw_aggregator_4.fc_x.bias
                encoder_state_dict['bw_aggregator_5.fc_x.weight'] = value.bw_aggregator_5.fc_x.weight
                encoder_state_dict['bw_aggregator_5.fc_x.bias'] = value.bw_aggregator_5.fc_x.bias
                encoder_state_dict['bw_aggregator_6.fc_x.weight'] = value.bw_aggregator_6.fc_x.weight
                encoder_state_dict['bw_aggregator_6.fc_x.bias'] = value.bw_aggregator_6.fc_x.bias
                encoder_state_dict['Linear_hidden.weight'] = value.Linear_hidden.weight
                encoder_state_dict['Linear_hidden.bias'] = value.Linear_hidden.bias
                encoder_state_dict['embedding_bilstm.weight_ih_l0'] = value.embedding_bilstm.weight_ih_l0
                encoder_state_dict['embedding_bilstm.weight_hh_l0'] = value.embedding_bilstm.weight_hh_l0
                encoder_state_dict['embedding_bilstm.bias_ih_l0'] = value.embedding_bilstm.bias_ih_l0
                encoder_state_dict['embedding_bilstm.bias_hh_l0'] = value.embedding_bilstm.bias_hh_l0
                encoder_state_dict['embedding_bilstm.weight_ih_l0_reverse'] = value.embedding_bilstm.weight_ih_l0_reverse
                encoder_state_dict['embedding_bilstm.weight_hh_l0_reverse'] = value.embedding_bilstm.weight_hh_l0_reverse
                encoder_state_dict['embedding_bilstm.bias_ih_l0_reverse'] = value.embedding_bilstm.bias_ih_l0_reverse
                encoder_state_dict['embedding_bilstm.bias_hh_l0_reverse'] = value.embedding_bilstm.bias_hh_l0_reverse
                encoder_state_dict['embedding_bilstm.weight_ih_l1'] = value.embedding_bilstm.weight_ih_l1
                encoder_state_dict['embedding_bilstm.weight_hh_l1'] = value.embedding_bilstm.weight_hh_l1
                encoder_state_dict['embedding_bilstm.bias_ih_l1'] = value.embedding_bilstm.bias_ih_l1
                encoder_state_dict['embedding_bilstm.bias_hh_l1'] = value.embedding_bilstm.bias_hh_l1
                encoder_state_dict['embedding_bilstm.weight_ih_l1_reverse'] = value.embedding_bilstm.weight_ih_l1_reverse
                encoder_state_dict['embedding_bilstm.weight_hh_l1_reverse'] = value.embedding_bilstm.weight_hh_l1_reverse
                encoder_state_dict['embedding_bilstm.bias_ih_l1_reverse'] = value.embedding_bilstm.bias_ih_l1_reverse
                encoder_state_dict['embedding_bilstm.bias_hh_l1_reverse'] = value.embedding_bilstm.bias_hh_l1_reverse
        self.encoder.load_state_dict(encoder_state_dict)
        self.encoder.eval()
        if USE_CUDA:
            self.encoder.cuda()
      
        graph_batch = shared_state_data.graph_batch['enc_graph_batch']
        
        # if graph_batch['g_fw_adj'] == []:
        #     return "None"
        fw_adj_info = torch.tensor(graph_batch['g_fw_adj'])
        bw_adj_info = torch.tensor(graph_batch['g_bw_adj'])
        feature_info = torch.tensor(graph_batch['g_ids_features'])
        batch_nodes = torch.tensor(graph_batch['g_nodes'])

        graph_embedding, graph_hidden, seq_embedding, seq_hidden = self.encoder((fw_adj_info,bw_adj_info,feature_info,batch_nodes))
        # encoder_outputs, problem_output = self.encoder.graph2tree_wang(shared_state_data.pair[7], input_length, graph_batch)
        shared_state_data.graph_embedding = graph_embedding
        shared_state_data.graph_hidden = graph_hidden
        shared_state_data.seq_embedding = seq_embedding
        shared_state_data.seq_hidden = seq_hidden
            
             
class VectorTextToEquationSystem(Transit):
    """
    Transit from VectorText to EquationSystemt
    """

    transit_name: str = "VectorTextToEquationSystem"

    # def __init__(self, hidden_size,output_lang, copy_nums, generate_nums, embedding_size):
    def __init__(self):
        super().__init__()


    def _check(self, transitors: List[str]) -> None:
        """
        Executes the transit check, checking whether all transitors have been defined.
        :param transitors:
        :raises AssertionError: If operation has no predecessors.
        """
        functions = [
            fn_name
            for fn_name, _ in inspect.getmembers(
                VectorTextToEquationSystem, predicate=inspect.isfunction
            )
        ]
        for t in transitors:
            if t not in functions:
                raise ValueError("transitors %s has not defined".format(t))
            
    def gtsdecoder(self, shared_state_data: SharedStateData) -> None:
       
       
        self.predict = models.Prediction(gts, 
                            op_nums=shared_state_data.output_lang.n_words - shared_state_data.copy_nums - 1 - len(shared_state_data.generate_nums),
                            input_size=len(shared_state_data.generate_nums))
        
        self.generate = models.GenerateNode(gts, 
                            op_nums=shared_state_data.output_lang.n_words - shared_state_data.copy_nums - 1 - len(shared_state_data.generate_nums),
                                )
        self.merge = models.Merge(gts)

        pretrained_state_dict_predict = torch.load('D:/code/benchmarkmachine/solvingengine/stategraph/transit/pretrained_model/predict')
        self.predict.load_state_dict(pretrained_state_dict_predict)
        pretrained_state_dict_generate = torch.load('D:/code/benchmarkmachine/solvingengine/stategraph/transit/pretrained_model/generate')
        self.generate.load_state_dict(pretrained_state_dict_generate)
        pretrained_state_dict_merge = torch.load('D:/code/benchmarkmachine/solvingengine/stategraph/transit/pretrained_model/merge')
        self.merge.load_state_dict(pretrained_state_dict_merge)

        seq_mask = torch.ByteTensor(1, shared_state_data.pair[1]).fill_(0)   
        num_mask = torch.ByteTensor(1, len(shared_state_data.pair[5]) + len(shared_state_data.generate_num_ids)).fill_(0)
        
       

        self.predict.eval()
        self.generate.eval()
        self.merge.eval()
        if USE_CUDA:
            self.predict.cuda()
            self.generate.cuda()
            self.merge.cuda()

        padding_hidden = torch.FloatTensor([0.0 for _ in range(self.predict.hidden_size)]).unsqueeze(0)

        batch_size = 1

        if USE_CUDA:
            seq_mask = seq_mask.cuda()
            padding_hidden = padding_hidden.cuda()
            num_mask = num_mask.cuda()

        # Prepare input and output variables
        node_stacks = [[models.TreeNode(_)] for _ in shared_state_data.vector_sentense_text.split(1, dim=0)]

        num_size = len(shared_state_data.pair[5])
        all_nums_encoder_outputs = evaluate.get_all_number_encoder_outputs(shared_state_data.vector_text, [shared_state_data.pair[5]], batch_size, num_size,
                                                                gts.hidden_size)
        num_start = shared_state_data.output_lang.num_start
        
        # B x P x N
        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]

        beams = [evaluate.TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

        for t in range(gts.MAX_OUTPUT_LENGTH):
            current_beams = []
            while len(beams) > 0:
                b = beams.pop()
                if len(b.node_stack[0]) == 0:
                    current_beams.append(b)
                    continue
                # left_childs = torch.stack(b.left_childs)
                left_childs = b.left_childs

                num_score, op, current_embeddings, current_context, current_nums_embeddings = self.predict(
                    b.node_stack, left_childs, shared_state_data.vector_text, all_nums_encoder_outputs, padding_hidden,
                    seq_mask, num_mask)

                out_score = models.nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

               
                topv, topi = out_score.topk(gts.beam_size)
               

                for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                    current_node_stack = evaluate.copy_list(b.node_stack)
                    current_left_childs = []
                    current_embeddings_stacks = evaluate.copy_list(b.embedding_stack)
                    current_out = copy.deepcopy(b.out)

                    out_token = int(ti)
                    current_out.append(out_token)

                    node = current_node_stack[0].pop()

                    if out_token < num_start:
                        generate_input = torch.LongTensor([out_token])
                        if USE_CUDA:
                            generate_input = generate_input.cuda()
                        left_child, right_child, node_label = self.generate(current_embeddings, generate_input, current_context)

                        current_node_stack[0].append(models.TreeNode(right_child))
                        current_node_stack[0].append(models.TreeNode(left_child, left_flag=True))

                        current_embeddings_stacks[0].append(evaluate.TreeEmbedding(node_label[0].unsqueeze(0), False))
                    else:
                       
                        current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)

                        while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                            sub_stree = current_embeddings_stacks[0].pop()
                            op = current_embeddings_stacks[0].pop()
                            current_num = self.merge(op.embedding, sub_stree.embedding, current_num)
                        current_embeddings_stacks[0].append(evaluate.TreeEmbedding(current_num, True))
                    if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                    else:
                        current_left_childs.append(None)
                    current_beams.append(evaluate.TreeBeam(b.score+float(tv), current_node_stack, current_embeddings_stacks,
                                                current_left_childs, current_out))
            beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
            beams = beams[:gts.beam_size]
            flag = True
            for b in beams:
                if len(b.node_stack[0]) != 0:
                    flag = False
            if flag:
                break
        
        exp = evaluate.out_expression_list(beams[0].out, shared_state_data.output_lang, shared_state_data.pair[4])
        shared_state_data.equation_system = exp


    def wanggtsdecoder(self, shared_state_data: SharedStateData) -> None:
        
        self.predict = models.Prediction(gts, 
                            op_nums=shared_state_data.output_lang.n_words-shared_state_data.copy_nums-1-len(shared_state_data.generate_nums),
                            input_size=len(shared_state_data.generate_nums))
        
        self.generate = models.GenerateNode(gts, 
                            op_nums=shared_state_data.output_lang.n_words-shared_state_data.copy_nums-1-len(shared_state_data.generate_nums)
                             )
        self.merge = models.Merge(gts)

        pretrained_state_dict_predict = torch.load('D:/code/benchmarkmachine/solvingengine/stategraph/transit/pretrained_model/predict')
        self.predict.load_state_dict(pretrained_state_dict_predict)
        pretrained_state_dict_generate = torch.load('D:/code/benchmarkmachine/solvingengine/stategraph/transit/pretrained_model/generate')
        self.generate.load_state_dict(pretrained_state_dict_generate)
        pretrained_state_dict_merge = torch.load('D:/code/benchmarkmachine/solvingengine/stategraph/transit/pretrained_model/merge')
        self.merge.load_state_dict(pretrained_state_dict_merge)


        # seq_mask = torch.ByteTensor(1, len(shared_state_data.seq_batch["input_cell"])).fill_(0)
        seq_mask = torch.ByteTensor(1, shared_state_data.pair[1]).fill_(0)
        # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)

        num_mask = torch.ByteTensor(1, len(shared_state_data.pair[5]) + len(shared_state_data.generate_num_ids)).fill_(0)
       

        self.predict.eval()
        self.generate.eval()
        self.merge.eval()
        if USE_CUDA:
            self.predict.cuda()
            self.generate.cuda()
            self.merge.cuda()

        padding_hidden = torch.FloatTensor([0.0 for _ in range(self.predict.hidden_size)]).unsqueeze(0)

        batch_size = 1

        if USE_CUDA:
            seq_mask = seq_mask.cuda()
            padding_hidden = padding_hidden.cuda()
            num_mask = num_mask.cuda()
           

        # Prepare input and output variables
        node_stacks = [[models.TreeNode(_)] for _ in shared_state_data.vector_sentense_text.split(1, dim=0)]

        num_size = len(shared_state_data.pair[5])
        all_nums_encoder_outputs = evaluate.get_all_number_encoder_outputs(shared_state_data.vector_text, [shared_state_data.pair[5]], batch_size, num_size,
                                                                gts.hidden_size)
        num_start = shared_state_data.output_lang.num_start
        
        # B x P x N
        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]

        beams = [evaluate.TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

        for t in range(gts.MAX_OUTPUT_LENGTH):
            current_beams = []
            while len(beams) > 0:
                b = beams.pop()
                if len(b.node_stack[0]) == 0:
                    current_beams.append(b)
                    continue
                # left_childs = torch.stack(b.left_childs)
                left_childs = b.left_childs

                num_score, op, current_embeddings, current_context, current_nums_embeddings = self.predict(
                    b.node_stack, left_childs, shared_state_data.vector_text, all_nums_encoder_outputs, padding_hidden,
                    seq_mask, num_mask)

                out_score = models.nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

               
                topv, topi = out_score.topk(gts.beam_size)
               

                for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                    current_node_stack = evaluate.copy_list(b.node_stack)
                    current_left_childs = []
                    current_embeddings_stacks = evaluate.copy_list(b.embedding_stack)
                    current_out = copy.deepcopy(b.out)

                    out_token = int(ti)
                    current_out.append(out_token)

                    node = current_node_stack[0].pop()

                    if out_token < num_start:
                        generate_input = torch.LongTensor([out_token])
                        if USE_CUDA:
                            generate_input = generate_input.cuda()
                        left_child, right_child, node_label = self.generate(current_embeddings, generate_input, current_context)

                        current_node_stack[0].append(models.TreeNode(right_child))
                        current_node_stack[0].append(models.TreeNode(left_child, left_flag=True))

                        current_embeddings_stacks[0].append(evaluate.TreeEmbedding(node_label[0].unsqueeze(0), False))
                    else:
                       
                        current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)

                        while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                            sub_stree = current_embeddings_stacks[0].pop()
                            op = current_embeddings_stacks[0].pop()
                            current_num = self.merge(op.embedding, sub_stree.embedding, current_num)
                        current_embeddings_stacks[0].append(evaluate.TreeEmbedding(current_num, True))
                    if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                    else:
                        current_left_childs.append(None)
                    current_beams.append(evaluate.TreeBeam(b.score+float(tv), current_node_stack, current_embeddings_stacks,
                                                current_left_childs, current_out))
            beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
            beams = beams[:gts.beam_size]
            flag = True
            for b in beams:
                if len(b.node_stack[0]) != 0:
                    flag = False
            if flag:
                break
        
        exp = evaluate.out_expression_list(beams[0].out, shared_state_data.output_lang, shared_state_data.pair[4])
        shared_state_data.equation_system = exp 

    def rnndecoder(self, shared_state_data: SharedStateData) -> None:
        # self.decoder = models.DecoderRNN(rd, input_size = shared_state_data.output_lang.n_words)
        self.decoder = models.DecoderRNN(rd, input_size = 29)
        self.attention_decoder = models.AttnUnit(rd, output_size = 29)
        # self.attention_decoder = models.AttnUnit(rd, output_size = shared_state_data.output_lang.n_words)
        
        pretrained_state_dict = torch.load('D:/code/benchmarkmachine/solvingengine/stategraph/transit/GraphConstruction/model_biGraphSAGE_BreadthTree') 
        decoder_state_dict = {}
        attention_decoder_state_dict = {}
        for key, value in pretrained_state_dict.items(): 
            if key.startswith("decoder_0"):   
                decoder_state_dict["embedding.weight"] = value.embedding.weight
                decoder_state_dict['lstm.i2h.weight'] = value.lstm.i2h.weight
                decoder_state_dict['lstm.i2h.bias'] = value.lstm.i2h.bias
                decoder_state_dict['lstm.h2h.weight'] = value.lstm.h2h.weight
                decoder_state_dict['lstm.h2h.bias'] = value.lstm.h2h.bias
            if key.startswith("decoder_1"):   
                attention_decoder_state_dict["linear_att.weight"] = value.linear_att.weight
                attention_decoder_state_dict['linear_att.bias'] = value.linear_att.bias
                attention_decoder_state_dict['linear_out.weight'] = value.linear_out.weight
                attention_decoder_state_dict['linear_out.bias'] = value.linear_out.bias
    
        
        self.decoder.load_state_dict(decoder_state_dict)
        self.attention_decoder.load_state_dict(attention_decoder_state_dict)

        self.attention_decoder.eval()

        graph_embedding = shared_state_data.graph_embedding
        graph_hidden = shared_state_data.graph_hidden
        seq_hidden = shared_state_data.seq_hidden
        using_gpu = True
        enc_outputs = graph_hidden

        if using_gpu:
            self.decoder = self.decoder.cuda()
            self.attention_decoder = self.attention_decoder.cuda()

        
        self.decoder.eval()

        prev_c = torch.zeros((1, self.decoder.hidden_size), requires_grad=False)
        prev_h = torch.zeros((1, self.decoder.hidden_size), requires_grad=False)
        if using_gpu:
            prev_c = prev_c.cuda()
            prev_h = prev_h.cuda()
        prev_c = graph_embedding
        prev_h = graph_embedding

        
        nums = shared_state_data.tree_pairs["nums"]

        queue_decode = []
        queue_decode.append({"s": (prev_c, prev_h), "parent":0, "child_index":1, "t": Tree()})
        head = 1
        while head <= len(queue_decode) and head <=100:
            s = queue_decode[head-1]["s"]
            parent_h = s[1]
            t = queue_decode[head-1]["t"]

            sibling_state = torch.zeros((1, self.decoder.hidden_size), dtype=torch.float, requires_grad=False)

            if using_gpu:
                sibling_state = sibling_state.cuda()
            flag_sibling = False
            for q_index in range(len(queue_decode)):
                if (head <= len(queue_decode)) and (q_index < head - 1) \
                        and (queue_decode[q_index]["parent"] == queue_decode[head - 1]["parent"]) \
                        and (queue_decode[q_index]["child_index"] < queue_decode[head - 1]["child_index"]):
                    flag_sibling = True
                    sibling_index = q_index
            if flag_sibling:
                sibling_state = queue_decode[sibling_index]["s"][1]

            if head == 1:
                prev_word = torch.tensor([shared_state_data.output_lang.word2index['<S>']], dtype=torch.long)
            else:
                prev_word = torch.tensor([shared_state_data.output_lang.word2index['(']], dtype=torch.long)
            if using_gpu:
                prev_word = prev_word.cuda()
            i_child = 1
            while True:
                curr_c, curr_h = self.decoder(prev_word, s[0], s[1], parent_h, sibling_state)
                prediction = self.attention_decoder(enc_outputs, curr_h, seq_hidden)

                s = (curr_c, curr_h)
                _, _prev_word = prediction.max(1)
                prev_word = _prev_word

                if int(prev_word[0]) == shared_state_data.output_lang.word2index['<E>'] or t.num_children >= rd.dec_seq_length_max:
                    break
                elif int(prev_word[0]) == shared_state_data.output_lang.word2index['<N>']:
                    queue_decode.append({"s": (s[0].clone(), s[1].clone()), "parent": head, "child_index":i_child, "t": Tree()})
                    t.add_child(int(prev_word[0]))
                else:
                    t.add_child(int(prev_word[0]))
                i_child = i_child + 1
            head = head + 1
        for i in range(len(queue_decode)-1, 0, -1):
            cur = queue_decode[i]
            queue_decode[cur["parent"]-1]["t"].children[cur["child_index"]-1] = cur["t"]

        predicted_cell = queue_decode[0]["t"].to_list(shared_state_data.output_lang)
 
        predicted_cell, predicted_seq = exp_transf.expression_code_validation(predicted_cell, shared_state_data.output_lang)
        
        predicted_equ = exp_transf.convert_ni_str_to_equation(predicted_seq, nums)  
        pattern = r'(\d+|\+|\-|\*|\/|\(|\))'
        predicted_equ = re.findall(pattern, predicted_equ)
        
        
        shared_state_data.equation_system = predicted_equ


class APTextToEquationSystem(Transit):
    """
    Transit from APText to EquationSystemt
    """

    transit_name: str = "APTextToEquationSystem"

    # def __init__(self, hidden_size,output_lang, copy_nums, generate_nums, embedding_size):
    def __init__(self):
        super().__init__()


    def _check(self, transitors: List[str]) -> None:
        """
        Executes the transit check, checking whether all transitors have been defined.
        :param transitors:
        :raises AssertionError: If operation has no predecessors.
        """
        functions = [
            fn_name
            for fn_name, _ in inspect.getmembers(
                APTextToEquationSystem, predicate=inspect.isfunction
            )
        ]
        for t in transitors:
            if t not in functions:
                raise ValueError("transitors %s has not defined".format(t))
            
    

               

        