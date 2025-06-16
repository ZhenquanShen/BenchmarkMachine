"""
#!/usr/bin/env: Python3
# -*- encoding: utf-8-*-
Description: Docstring
Author: Xiaopan Lyu
Date: 2022-12-21 19:04:13
LastEditTime: 2023-04-15 11:55:46
LastEditors: Xiaopan LYU
"""

import json
import data_utils.yaml_handler as yh


class MimicConfig(object):
    state_graph = ""
    transits = ""
    def setup(self, state_transit_profile):
        with open(state_transit_profile, "r") as f:
            self.state_graph = json.load(f)
        self.transits = yh.yaml_to_namespace('D:/code/benchmarkmachine/solvingengine/config/transformer_settings.yaml')
        
mimic_cfg = MimicConfig()
# mimic solver configuration file
# state_transit_profile = "D:/code/benchmarkmachine/description/bert-gts.json"
# state_transit_profile = "D:/code/benchmarkmachine/description/wang-gts.json"
# state_transit_profile = "D:/code/benchmarkmachine/description/li-rnn.json"
# state_transit_profile = "D:/code/benchmarkmachine/description/scene.json"
state_transit_profile = "D:/code/benchmarkmachine/description/deepseek.json"
mimic_cfg.setup(state_transit_profile)



