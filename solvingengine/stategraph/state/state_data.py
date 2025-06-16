#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Time: 2023/9/20 10:59
@Author: Xiaopan Lyu
@Description:A shared data class for solving a given problem
Copyright (c) 2023 . All rights reserved.
"""
from dataclasses import dataclass, field
import numpy as np
from data_utils.pre_data import Lang
from data_utils.pre_data import Lang1


@dataclass
class s2model:
    """
    Represents the matched syntax-semantics model
    """

    id: str = field(default_factory=str)
    pattern: str = field(default_factory=str)
    relation_template: str = field(default_factory=str)
    var_slot_val: str = field(default_factory=str)
    var_slot_index: dict = field(default_factory=dict)


@dataclass
class ExplRelation:
    """
    Represents the extracted explicit relation
    """

    relation: str = field(default_factory=str)
    var_entity: dict = field(default_factory=dict)
    sent_text: str = field(default_factory=str)
    matched_token: list[tuple] = field(default_factory=list)
    matched_model: s2model = field(default_factory=str)


@dataclass
class ImplRelation:
    """
    Represents the acquired implicit relation
    """

    relation: str = field(default_factory=str)
    entity: list[str] = field(default_factory=list)


@dataclass
class DiagRelation:
    """
    Represents the acquired diagram relation
    """

    relation: str = field(default_factory=str)
    entity: list[str] = field(default_factory=list)


@dataclass
class SceneRelation:
    """
    Represents the acquired scene relation
    """

    relation: list[str] = field(default_factory=list)
    scene: str = field(default_factory=str)


@dataclass
class FuncExpression:
    """
    Represents the acquired function expression
    """

    expression: str = field(default_factory=str)
    func_domain: str = field(default_factory=str)
    func_range: str = field(default_factory=str)


@dataclass
class SharedStateData:
    """
    Represents the shared problem dataclass during solving the given problem
    """

    id: str = field(default_factory=str)
    text: str = field(default_factory=str)
    diagram_url: str = field(default_factory=str)
    gold_answer: list[str] = field(default_factory=list)

    ap_text: str = field(default_factory=str)
    ap_diagram: str = field(default_factory=str)

    sentences: list[str] = field(default_factory=list)
    segmented_tokens: list[str] = field(default_factory=list)
    sentence_seg_ranges: list[list] = field(default_factory=list)
    pos_tagged_tokens: list[tuple] = field(default_factory=list)

    vector_text: np.ndarray = field(default_factory=list)
    vector_sentense_text: np.ndarray = field(default_factory=list)

    expl_relations: list[ExplRelation] = field(default_factory=list)
    impl_relations: list[ImplRelation] = field(default_factory=list)
    diag_relations: list[DiagRelation] = field(default_factory=list)
    scene_relations: list[SceneRelation] = field(default_factory=list)
    func_expressions: list[FuncExpression] = field(default_factory=list)

    relation_set: list[str] = field(default_factory=list)

    equation_system: list[str] = field(default_factory=list)
    gold_equation_system: list[str] = field(default_factory=list)
    equation_solution: dict = field(default_factory=dict)

    readable_solution: list[str] = field(default_factory=list)
    pair: tuple = field(default_factory=tuple)
    copy_nums: int = field(default_factory=int)
    generate_nums: list = field(default_factory=list)
    generate_num_ids: list = field(default_factory=list)
    
    # output_lang: Lang1 = Lang1()
    # input_lang: Lang1 = Lang1()
    input_lang: Lang = Lang()
    output_lang: Lang = Lang()
    batch: np.ndarray = field(default_factory=list)
    inputlength: list[str] = field(default_factory=list)
    num_size_batch: list[str] = field(default_factory=list)
    output_batch: np.ndarray = field(default_factory=list)
    outputlength: list[str] = field(default_factory=list)
    num_pos_batch: list[str] = field(default_factory=list)
    num_stack_batch: list[str] = field(default_factory=list)
    loss: float = field(default=0.0)
    
    seq_batch: list[str] = field(default_factory=list)
    graph_batch: list[str] = field(default_factory=list)
    langs:list[str] = field(default_factory=list) 
    graph_embedding: np.ndarray = field(default_factory=list)
    graph_hidden: np.ndarray = field(default_factory=list)
    seq_embedding: np.ndarray = field(default_factory=list)
    seq_hidden: np.ndarray = field(default_factory=list)
    tree_pairs: tuple = field(default_factory=tuple)


    def clear(self):
        self.id = ""
        self.text = ""
        self.diagram_url = ""
        self.gold_answer = []

        self.ap_text = ""
        self.ap_diagram = ""

        self.sentences = []
        self.segmented_tokens = []
        self.sentence_seg_ranges = []
        self.pos_tagged_tokens = []

        self.vector_text = []

        self.expl_relations = []
        self.impl_relations = []
        self.diag_relations = []
        self.scene_relations = []
        self.func_expressions = []

        self.relation_set = []

        self.equation_system = []
        self.equation_solution = {}

        self.readable_solution = []
        self.pair = ()
        self.vector_sentense_text = []
        self.batch = [] 
        self.inputlength = []
        self.num_size_batch = []
        self.generate_num_ids = []
        self.output_batch = []
        self.outputlength = []
        self.num_pos_batch = []
        self.num_stack_batch = []
       
        self.seq_batch = []
        self.graph_batch = []
        self.langs = []

        self.graph_embedding = []
        self.graph_hidden = []
        self.seq_embedding = []
        self.seq_hidden = []
        self.tree_pair = ()