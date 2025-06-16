"""
#!/usr/bin/env: Python3
# -*- encoding: utf-8-*-
Description: Docstring
Author: Xiaopan Lyu
Date: 2022-12-18 20:11:50
LastEditTime: 2023-07-24 16:38:22
LastEditors: Xiaopan LYU
"""

import json
import logging
import re


def read_json_file(fpath):
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            js = json.load(f)
            return js
    except:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                s = f.read()
                v = json.dumps(s, ensure_ascii=False)
                js = json.loads(v)
                if isinstance(js, str):
                    js = eval(js)
                return js
        except:
            try:  # json.dumps(eval(s))解决json文件内单引号的问题
                with open(fpath, "r", encoding="utf-8") as f:
                    s = f.read()
                    js = json.loads(json.dumps(eval(s), ensure_ascii=False))
                    return js
            except:
                logging.error("read file failed! {}".format(fpath))
                return []


def read_lines(fpath, filter=False):
    # Read the file and split into lines, remove \ufeff  of first line
    f = (
        open("%s" % (fpath), encoding="utf-8")
        .read()
        .strip()
        .encode("utf-8")
        .decode("utf-8-sig")
    )
    # if remove_space:
    #     f = f.replace(' ', '')
    if filter:
        # reg = '[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~\s]+'  # 用户也可以在此进行自定义过滤字符
        # \s是去空格的，但是同时也把\n去掉了，暂不使用，需要保留，。? \n,
        reg = '[!"#$%&()*+,-./:;<=>?@?★、…【】《》“”‘’！[\\]^_`{|}~]+'
        f = re.sub(reg, "", f)
        f = f.replace(" ", "")

    lines = f.split("\n")
    return lines


def read_STD_JSON(file_path):

    r = read_json_file(file_path)
    data = r["body"]
    return data


def read_Line_JSON(file_path):

    r = read_lines(file_path)
    return r


def save_json_file(json_data, fpath):
    with open("%s" % (fpath), "w", encoding="utf-8") as f:
        try:
            json.dump(json_data, f, ensure_ascii=False)
        except:
            f.writelines(str(json_data))
            f.close()


