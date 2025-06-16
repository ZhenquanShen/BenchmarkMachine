"""
#!/usr/bin/env: Python3.7.6
# -*- encoding: utf-8-*-
Description: 
Author: Allen
Date: 2022-12-13 14:28:05
LastEditTime: 2023-07-29 10:56:30
LastEditors: Xiaopan LYU
"""
from solvingengine.config import mimic_cfg
from data_utils.file_helper import FileHelper as fh


class ArithmeticScene(object):
    def __init__(self):
        # get scene models
        self.scene_models = fh.read_json_file(mimic_cfg.transits.scene_model.path)
        vcb = fh.read_json_file(mimic_cfg.transits.scene_model.vocab)
        self.scene_keywords = vcb.get("keywords")
        self.scene_pos = vcb.get("pos")

    def do_kw_ptn_matching(self, **args):
        scene_v = args.get("scene_v")
        word_tag = args.get("word_tag")
        scene_ptns = scene_v.get("scene_ptn").get("kw_ptn")
        text = "".join([w for w, _ in word_tag])
        is_match = True
        for ptn in scene_ptns:
            ptn_list = ptn.split(",")
            for kw in ptn_list:
                if kw not in text:
                    is_match = False
                    break
            if is_match is True:
                break
        return is_match

    def do_sent_s2_ptn_matching(self, **args):
        scene_v = args.get("scene_v")
        sentences_ranges = args.get("sentences_ranges")
        word_tag = args.get("word_tag")
        scene_ptns = scene_v.get("scene_ptn").get("sent_s2_ptn")
        is_match = False
        for sent_range in sentences_ranges:
            sent_tagged_tokens = word_tag[sent_range[0] : sent_range[1]]
            sent_keywords = [
                w for w, t in sent_tagged_tokens if w in self.scene_keywords
            ]
            # currently, the arithmetic scene model must have keywords
            if len(sent_keywords) > 0:
                token_seq = []
                token_position_seq = []
                for idx, token in zip(
                    range(0, len(sent_tagged_tokens)), sent_tagged_tokens
                ):
                    if token[0] in sent_keywords:
                        token_seq.append(token[0])
                        token_position_seq.append(sent_range[0] + idx)
                    elif token[1] in self.scene_pos:
                        token_seq.append(token[1])
                        token_position_seq.append(sent_range[0] + idx)
                for ptn in scene_ptns:
                    pattern_list = ptn.split(",")
                    """filter scene model by length"""
                    if len(pattern_list) <= len(token_seq):
                        head = pattern_list[0]
                        rear = pattern_list[-1]
                        head_index = 0
                        rear_index = 0
                        """filter sub token sequence by model's first and last token"""
                        for head_id in range(0, len(token_seq)):
                            if head == token_seq[head_id]:
                                head_index = head_id
                                break
                        for rear_id in range(len(token_seq) - 1, -1, -1):
                            if rear == token_seq[rear_id]:
                                rear_index = rear_id
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
                                    is_match = True
                                    break
                    if is_match is True:
                        break
            if is_match is True:
                break
        return is_match

    def identify_scene_type(self, shared_state_data):
        # get problem token sequence
        word_tag = shared_state_data.pos_tagged_tokens
        # arithmetic scene model matching
        sentences_ranges = shared_state_data.sentence_seg_ranges
        matched_scene_list = []
        """filter scene model by matching pattern"""
        for scene_k, scene_v in self.scene_models.items():
            scene_ptn_type = list(scene_v.get("scene_ptn").keys())[0]
            """matching by sent_s2_ptn"""
            param = {
                "scene_v": scene_v,
                "sentences_ranges": sentences_ranges,
                "word_tag": word_tag,
            }
            func_name = "do_" + scene_ptn_type + "_matching"
            call_func = getattr(self, func_name)
            is_match = call_func(**param)
            if is_match is True:
                matched_scene_list.append(scene_k)
        return matched_scene_list

    def scene_reasoning(self, scene_name, shared_state_data):
        equation_system = []
        explicit_relations = shared_state_data.expl_relations
        implicit_relations = shared_state_data.impl_relations
        # if len(implicit_relation) > 0:
        #     equation_system.append(implicit_relation)
        element_ptns = self.scene_models.get(scene_name).get("element_ptn")
        scene_formula = self.scene_models.get(scene_name).get("scene_formula")
        equation_system.append(scene_formula)
        # elem_relation={}
        assigned_eqr = []
        assigned_iqr = []
        for elem, elem_ptn in element_ptns.items():
            # 按照情景挑选直陈关系
            for eqr in explicit_relations:
                if eqr not in assigned_eqr:  # one relation only link to one scene elem
                    is_match = False
                    eqr_entity = eqr.var_entity.values()
                    for ep in elem_ptn:
                        ep_list = ep.split(",")  # must match whole list
                        indicator = [v for v in ep_list if v in eqr_entity]
                        if len(indicator) == len(ep_list):
                            assigned_eqr.append(eqr)
                            eqr_relation = eqr.relation
                            eqr_relation = eqr_relation.replace(indicator[0], elem)
                            equation_system.append(eqr_relation)
                            # elem_relation.setdefault(elem,eqr_relation)
                            is_match = True
                            break
                    if is_match is True:
                        break
            # 按照情景挑选隐含关系
            if len(implicit_relations) > 0:
                for iqr in implicit_relations:
                    if (
                        iqr not in assigned_iqr
                    ):  # one relation only link to one scene elem
                        # print("iqr:", iqr)
                        is_match = False
                        iqr_entity = iqr['var_entity']
                        
                        for ep in elem_ptn:
                            ep_list = ep.split(",")  # must match whole list
                            indicator = [v for v in ep_list if v in iqr_entity]
                            if len(indicator) == len(ep_list):
                                assigned_iqr.append(iqr)
                                iqr_relation = iqr['relation']
                                iqr_relation = iqr_relation.replace(indicator[0], elem)
                                equation_system.append(iqr_relation)
                                # elem_relation.setdefault(elem,iqr_relation)
                                is_match = True
                                break
                        if is_match is True:
                            break
        return equation_system
