'''
#!/usr/bin/env: Python3
# -*- encoding: utf-8-*-
Description: Docstring
Author: Xiaopan Lyu
Date: 2022-12-21 19:04:13
LastEditTime: 2023-04-15 11:55:27
LastEditors: Xiaopan LYU
'''
import yaml
import argparse


def dict_to_namespace(config: 'dict') -> 'Namespace':
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict_to_namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def yaml_to_namespace(filepath: str):
    yaml_cont = read_yaml(filepath)
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    r = dict_to_namespace({**yaml_cont, **vars(args)})
    return r


def read_yaml(filepath: str):
    with open(filepath, 'r') as f:
        r = yaml.load(f, Loader=yaml.FullLoader)
    return r
