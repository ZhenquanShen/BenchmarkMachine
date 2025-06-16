"""
#!/usr/bin/env: Python3
# -*- encoding: utf-8-*-
Description: 
Author: Xiaopan LYU
Date: 2023-07-03 08:44:12
LastEditTime: 2023-07-22 19:33:43
LastEditors: Xiaopan LYU
"""

from data_utils.pre_data import Lang, Lang1
from data_utils import pre_data
import json
import re
from transformers import AutoTokenizer
import copy
import random
import pickle as pkl
from random import randint
from collections import OrderedDict
import data_utils.yaml_handler as yh
import data_utils.graph_utils_li as graph_li
import numpy as np

class UserConfig(object):
    start = ""

    def setup(self, start_path):
        self.start = yh.yaml_to_namespace(start_path)

user_cfg = UserConfig()
user_cfg.setup("D:/code/benchmarkmachine/data_utils/dataset.yaml")

langs = pkl.load( open("D:/code/benchmarkmachine/solvingengine/stategraph/transit/GraphConstruction/cn_lang_map.pkl", "rb" ) )
input_lang, output_lang = langs

class DataLoader(object):
    """docstring for FileToProblemText."""

    def __init__(self):
        self.file_path = user_cfg.start.dataset.file_path
        self.file_type = user_cfg.start.dataset.file_type
        # self.file_type = filetype.guess(self.file_path)
        self.chunk_size = user_cfg.start.dataset.chunk_size

    def loading(self):
        """call transform method"""
        switch = {
            "STD_JSON": "read_STD_JSON",
            "CSV": "read_CSV",
            "LN_JSON": "read_Line_JSON",
            "json": "read_json",
        }
        return getattr(self, switch[self.file_type])()
    
    def preprocess(problem_list):
        data = load_data(problem_list)
        # pairs, pairs_wang, generate_nums, copy_nums = transfer_num(data)# transfer num into "NUM"
        pairs, generate_nums, copy_nums = transfer_num_1(data)# transfer num into "NUM"
        temp_pairs = []
        # k = 0
        for k in range(len(pairs)):
            p = pairs[k]
            # print("k:", k)
            temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3])) # 数字表达式中缀转前缀
            # k = k+1
        pairs = temp_pairs
        group_data = read_json("D:/code/benchmarkmachine/dataset/Math_23K_processed.json")
        pairs = DataLoader.get_train_test_fold(data,pairs,group_data)

        # input_lang, output_lang, pairs= prepare_data_23k(pairs, 5, generate_nums, copy_nums, tree=True)
        input_lang, output_lang, pairs= prepare_data_23k_1(pairs, 5, generate_nums, copy_nums, tree=True)

        generate_num_ids = []
        for num in generate_nums:
            generate_num_ids.append(output_lang.word2index[num])


        # tree_pairs = pre_data.prepare_data(input_lang, output_lang, pairs_wang, 2, generate_nums, copy_nums, tree=True)
        # tree_pairs = pre_data.remove_invalid_data(tree_pairs)
        # tree_pairs = pre_data.prepare_tree(tree_pairs, output_lang)
       
        # graph_wang_test = pre_data.create_graph_single_wang(tree_pairs)
       
        # graph_li_test, input_lang = graph_li.create_graph_single_li(tree_pairs, input_lang)
        return pairs, input_lang, output_lang, generate_nums, generate_num_ids, copy_nums
        # return pairs, input_lang, output_lang, generate_nums, generate_num_ids, copy_nums, tree_pairs, graph_li_test

    def read_CSV(self):
        import pandas as pd

        reader = pd.read_csv(self.file_path, header=None, chunksize=self.chunk_size)
        chunk = reader.get_chunk(self.chunk_size)
        probs = []
        keys = ["id", "text", "gold_answer"]
        for aline in chunk.values:
            line = list(aline)
            probs.append(dict(zip(keys, line)))
        return probs

    def read_STD_JSON(self):
        from data_utils.file_helper import FileHelper as fh

        r = fh.read_json_file(self.file_path)
        data = r["body"]
        # print("data:", data)
        return data

    def read_Line_JSON(self):
        from data_utils.file_helper import FileHelper as fh

        r = fh.read_lines(self.file_path)
        return r
    
    def read_json(path):
        with open(path,'r') as f:
            file = json.load(f)
        return file

    def get_train_test_fold(data,pairs,group):
        test_fold = []
        for item,pair,g in zip(data, pairs, group):
            pair = list(pair)
            pair.append(g['group_num'])
            pair = tuple(pair)
            test_fold.append(pair)
        return  test_fold

    def get_single_example_graph(input_batch, input_length,group,num_value,num_pos):
        batch_graph = []
        max_len = input_length
        sentence_length = input_length
        quantity_cell_list = group
        num_list = num_value
        id_num_list = num_pos
        graph_newc = DataLoader.get_quantity_cell_graph(max_len, id_num_list, sentence_length, quantity_cell_list)
        graph_quanbet = DataLoader.get_quantity_between_graph(max_len, id_num_list, sentence_length, quantity_cell_list)
        graph_attbet = DataLoader.get_attribute_between_graph(input_batch, max_len, id_num_list, sentence_length, quantity_cell_list)
        graph_greater = DataLoader.get_greater_num_graph(max_len, sentence_length, num_list, id_num_list)
        graph_lower = DataLoader.get_greater_num_graph(max_len, sentence_length, num_list, id_num_list)
        #graph_newc1 = get_quantity_graph1(input_batch, max_len, id_num_list, sentence_length, quantity_cell_list)
        graph_total = [graph_newc.tolist(),graph_greater.tolist(),graph_lower.tolist(),graph_quanbet.tolist(),graph_attbet.tolist()]
        batch_graph.append(graph_total)
        batch_graph = np.array(batch_graph)
        return batch_graph

    def get_quantity_cell_graph(max_len, id_num_list, sentence_length, quantity_cell_list,contain_zh_flag=True):
        diag_ele = np.zeros(max_len)
        for i in range(sentence_length):
            diag_ele[i] = 1
        graph = np.diag(diag_ele)
        #quantity_cell_list = quantity_cell_list.extend(id_num_list)
        if not contain_zh_flag:
            return graph
        for i in id_num_list:
            for j in quantity_cell_list:
                if i < max_len and j < max_len and j not in id_num_list and abs(i-j) < 4:
                    graph[i][j] = 1
                    graph[j][i] = 1
        return graph
    
    def get_quantity_between_graph(max_len, id_num_list, sentence_length, quantity_cell_list,contain_zh_flag=True):
        diag_ele = np.zeros(max_len)
        for i in range(sentence_length):
            diag_ele[i] = 1
        graph = np.diag(diag_ele)
        #quantity_cell_list = quantity_cell_list.extend(id_num_list)
        if not contain_zh_flag:
            return graph
        for i in id_num_list:
            for j in quantity_cell_list:
                if i < max_len and j < max_len and j not in id_num_list and abs(i-j) < 4:
                    graph[i][j] = 1
                    graph[j][i] = 1
        for i in id_num_list:
            for j in id_num_list:
                graph[i][j] = 1
                graph[j][i] = 1
        return graph
    
    def get_attribute_between_graph(input_batch, max_len, id_num_list, sentence_length, quantity_cell_list,contain_zh_flag=True):
        diag_ele = np.zeros(max_len)
        for i in range(sentence_length):
            diag_ele[i] = 1
        graph = np.diag(diag_ele)
        #quantity_cell_list = quantity_cell_list.extend(id_num_list)
        if not contain_zh_flag:
            return graph
        for i in id_num_list:
            for j in quantity_cell_list:
                if i < max_len and j < max_len and j not in id_num_list and abs(i-j) < 4:
                    graph[i][j] = 1
                    graph[j][i] = 1
        for i in quantity_cell_list:
            for j in quantity_cell_list:
                if i < max_len and j < max_len:
                    if input_batch[i] == input_batch[j]:
                        graph[i][j] = 1
                        graph[j][i] = 1
        return graph
    
    def get_greater_num_graph(max_len, sentence_length, num_list, id_num_list,contain_zh_flag=True):
        diag_ele = np.zeros(max_len)
        num_list = DataLoader.change_num(num_list)
        for i in range(sentence_length):
            diag_ele[i] = 1
        graph = np.diag(diag_ele)
        if not contain_zh_flag:
            return graph
        for i in range(len(id_num_list)):
            for j in range(len(id_num_list)):
                if float(num_list[i]) > float(num_list[j]):
                    graph[id_num_list[i]][id_num_list[j]] = 1
                else:
                    graph[id_num_list[j]][id_num_list[i]] = 1
        return graph
    
    def get_greater_num_graph(max_len, sentence_length, num_list, id_num_list,contain_zh_flag=True):
        diag_ele = np.zeros(max_len)
        num_list = DataLoader.change_num(num_list)
        for i in range(sentence_length):
            diag_ele[i] = 1
        graph = np.diag(diag_ele)
        if not contain_zh_flag:
            return graph
        for i in range(len(id_num_list)):
            for j in range(len(id_num_list)):
                if float(num_list[i]) > float(num_list[j]):
                    graph[id_num_list[i]][id_num_list[j]] = 1
                else:
                    graph[id_num_list[j]][id_num_list[i]] = 1
        return graph
    
    def change_num(num):
        new_num = []
        for item in num:
            if '/' in item:
                new_str = item.split(')')[0]
                new_str = new_str.split('(')[1]
                a = float(new_str.split('/')[0])
                b = float(new_str.split('/')[1])
                value = a/b
                new_num.append(value)
            elif '%' in item:
                value = float(item[0:-1])/100
                new_num.append(value)
            else:
                new_num.append(float(item))
        return new_num


def transfer_num(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    pairs_wang = []
    generate_nums = [] #生成的数字，即未在题目中出现，但是答案中出现的，一般为Π这种
    generate_nums_dict = {} #记录生成的数字出现次数
    copy_nums = 0 #记载在题目中出现的最多数字数目，也可以理解为N0,N1.N2......的数目
    k = 0
    for k in range(len(data)):
        d = data[k]
        # print("d:", d)
        nums = []
        input_seq = []
        seg = d["segmented_text"].strip().split(" ")
        equation = d["equation"][2:] #'x=(11-1)*2' 去掉x=
        # id = d["id"]
        # ans = d["ans"]

        equation = ['(' if x== '[' else x for x in equation]
        equation = [')' if x== ']' else x for x in equation]
        equation = ''.join(equation)

        for s in seg:                  #将数字替代为num
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
                input_seq.append("NUM")
                # input_seq.append("N"+str(len(nums)-1))
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            # elif pos and pos.start() > 0:
            #     nums.append(s[pos.start(): pos.end()])
            #     input_seq.append(s[: pos.start()])
            #     #input_seq.append("NUM")
            #     input_seq.append("N"+str(len(nums)-1))
            #     if pos.end() < len(s):
            #         input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)
        if copy_nums < len(nums):
            copy_nums = len(nums)

        nums_fraction = []

        for num in nums:  #匹配分数
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        def seg_and_tag(st):  # seg the equation and tag the num #例如(11-1)*2更换成(N1-1)*N0  N后面的数字为在题目中出现 的次序
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1:
                        res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equation)
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1
        num_pos = []
        # num_list = ["N"+str(idx) for idx in range(50)]
        for i, j in enumerate(input_seq):#记录数字在题目中出现的位置
            if j == "NUM":
                num_pos.append(i)
        assert len(nums) == len(num_pos)
        # pairs.append((input_seq, out_seq, nums, num_pos, d["ans"]))
        pair_wang = {'id':id, 'input_seq':input_seq, 'out_equ':equation, 'out_seq':out_seq,
                'nums':nums, 'num_pos':num_pos, 'num_pos_len':len(num_pos),#'ans':ans
                }
        pairs_wang.append(pair_wang)
        pairs.append((input_seq, out_seq, nums, num_pos))
        k = k + 1

    temp_g = []
    for g in generate_nums:               #保留出现五次以上的生成数
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)
    return pairs, pairs_wang, temp_g, copy_nums
    
def transfer_num_1(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = [] #生成的数字，即未在题目中出现，但是答案中出现的，一般为Π这种
    generate_nums_dict = {} #记录生成的数字出现次数
    copy_nums = 0 #记载在题目中出现的最多数字数目，也可以理解为N0,N1.N2......的数目
    for d in data:
        nums = []
        input_seq = []
        seg = d["segmented_text"].strip().split(" ")
        equations = d["equation"][2:] #'x=(11-1)*2' 去掉x=

        for s in seg:                  #将数字替代为num
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)
        if copy_nums < len(nums):
            copy_nums = len(nums)

        nums_fraction = []

        for num in nums:  #匹配分数
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        def seg_and_tag(st):  # seg the equation and tag the num #例如(11-1)*2更换成(N1-1)*N0  N后面的数字为在题目中出现 的次序
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1:
                        res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1
        num_pos = []
        for i, j in enumerate(input_seq):#记录数字在题目中出现的位置
            if j == "NUM":
                num_pos.append(i)
        assert len(nums) == len(num_pos)
        # pairs.append((input_seq, out_seq, nums, num_pos, d["ans"]))
        pairs.append((input_seq, out_seq, nums, num_pos))

    temp_g = []
    for g in generate_nums:               #保留出现五次以上的生成数
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)
    return pairs, temp_g, copy_nums

def load_data(problem_list):  # load the json data to list(dict()) for MATH 23K
    data = []
    i = 0
    for i in range(len(problem_list)):    
        if "千米/小时" in problem_list[i]["equation"]:
            problem_list[i]["equation"] = problem_list[i]["equation"][:-5]
        data.append(problem_list[i])
        i += 1
    return data

def from_infix_to_prefix(expression):#中缀改前缀
    st = list()
    res = list()
    priority = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
    expression = copy.deepcopy(expression)
    expression.reverse()
    for e in expression:
        if e in [")", "]"]:
            st.append(e)
        elif e == "(":
            c = st.pop()
            while c != ")":
                res.append(c)
                c = st.pop()
        elif e == "[":
            c = st.pop()
            while c != "]":
                res.append(c)
                c = st.pop()
        elif e in priority:
            while len(st) > 0 and st[-1] not in [")", "]"] and priority[e] < priority[st[-1]]:
                res.append(st.pop())
            st.append(e)
        else:
            res.append(e)
    while len(st) > 0:
        res.append(st.pop())
    res.reverse()
    return res

def read_json(path):
    with open(path,'r') as f:
        file = json.load(f)
    return file


# Return a list of indexes, one for each word in the sentence, plus EOS
def indexes_from_sentence(lang, sentence, tree=False):
    res = []
    for word in sentence:
        if len(word) == 0:
            continue
        if word in lang.word2index:
            res.append(lang.word2index[word])
        else:
            res.append(lang.word2index["UNK"])
    if "EOS" in lang.index2word and not tree:
        res.append(lang.word2index["EOS"])
    return res

def prepare_data_23k(pairs, trim_min_count, generate_nums, copy_nums, tree=False):
    input_lang = Lang()
    output_lang = Lang()
    train_pairs = []
    test_pairs = []
    test_pairs_ape = []
    # tokenizer = BertTokenizer.from_pretrained("hfl/bert-base-chinese")
    tokenizer = AutoTokenizer.from_pretrained("D:/code/mimicsolver/conf/model/chinese-bert-wwm-ext")
    print("Indexing words...")
    for pair in pairs:#单词编码
        if not tree:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
        elif pair[-1]:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
    input_lang.build_input_lang(trim_min_count)#去掉点不足数的单词
    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)#output to index
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)
    
    print('Number of pairs %d' % (len(pairs)))
    m = 0
    for m in range(len(pairs)):
        pair = pairs[m]
        # print("pair:", type(pair))
        num_stack = []
        for idx in range(len(pair[0])):
            if pair[0][idx] == 'NUM':
                pair[0][idx] = 'n'
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])
        inputs = tokenizer(pair[0], is_split_into_words=True,return_tensors="pt", add_special_tokens=False)

        num_pos = []
        for idx,i in enumerate(inputs['input_ids'].squeeze()):
            if tokenizer.convert_ids_to_tokens(int(i)) == 'n':
                num_pos.append(idx)
        num_stack.reverse()
        input_cell = indexes_from_sentence(input_lang, pair[0])
        output_cell = indexes_from_sentence(output_lang, pair[1], tree)
        # if output_lang.word2index["UNK"] in output_cell:
        #     continue
        if len(input_cell) > 100 or len(output_cell) > 20:
            continue
        test_pairs.append((input_cell, inputs['input_ids'].squeeze().size(0), output_cell, len(output_cell),
                           pair[2], num_pos, num_stack, inputs))
        m = m+1
    # print('Number of testing data %d' % (len(test_pairs)))
        input_lang.add_unk_symbol()
        output_lang.add_unk_symbol()
    return input_lang, output_lang, test_pairs

def prepare_data_23k_1(pairs_tested, trim_min_count, generate_nums, copy_nums, tree=False):
    input_lang = Lang1()
    output_lang = Lang1()
    train_pairs = []
    test_pairs = []
    test_pairs_ape = []
    tokenizer = AutoTokenizer.from_pretrained("D:/code/mimicsolver/conf/model/chinese-bert-wwm-ext")
    print("Indexing words...")
    for pair in pairs_tested:#单词编码
        if not tree:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
        elif pair[-1]:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
    input_lang.build_input_lang(trim_min_count)#去掉点不足数的单词
    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)#output to index
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)
    for pair in pairs_tested:
        num_stack = []
        for idx in range(len(pair[0])):
            if pair[0][idx] == 'NUM':
                pair[0][idx] = 'n'
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])
        inputs = tokenizer(pair[0], is_split_into_words=True,return_tensors="pt", add_special_tokens=False)

        num_pos = []
        for idx,i in enumerate(inputs['input_ids'].squeeze()):
            if tokenizer.convert_ids_to_tokens(int(i)) == 'n':
                num_pos.append(idx)
        num_stack.reverse()
        input_cell = indexes_from_sentence(input_lang, pair[0])
        output_cell = indexes_from_sentence(output_lang, pair[1], tree)
        if output_lang.word2index["UNK"] in output_cell:
            continue
        if len(input_cell) > 100 or len(output_cell) > 20:
            continue
        # test_pairs.append((input_cell, inputs['input_ids'].squeeze().size(0), output_cell, len(output_cell),
        #                    pair[2], num_pos, num_stack, inputs))
        test_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell), pair[2], pair[3], num_stack, pair[4]))
    print('Number of testing data %d' % (len(test_pairs)))
    return input_lang, output_lang, test_pairs


PAD_token = 0
# Pad a with the PAD symbol
def pad_seq(seq, seq_len, max_length):
    seq += [PAD_token for _ in range(max_length - seq_len)]
    return seq

# prepare the batches
def prepare_train_batch(pairs_to_batch, batch_size):
    pairs = copy.deepcopy(pairs_to_batch)
    random.shuffle(pairs)  # shuffle the pairs
    pos = 0
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    num_size_batches = []
    bert_input_batches = []
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
        input_length = []
        output_length = []
        for _, i, _, j, _, _, _, _ in batch:
            input_length.append(i)
            output_length.append(j)
        input_lengths.append(input_length)
        output_lengths.append(output_length)
        input_len_max = input_length[0]
        output_len_max = max(output_length)
        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        num_size_batch = []
        bert_input_batch = []
        for i, li, j, lj, num, num_pos, num_stack, bert_input in batch:
            num_batch.append(len(num))
            input_batch.append(pad_seq(i, li, input_len_max))
            output_batch.append(pad_seq(j, lj, output_len_max))
            num_stack_batch.append(num_stack)
            num_pos_batch.append(num_pos)
            num_size_batch.append(len(num_pos))
            bert_input_batch.append(bert_input)
        input_batches.append(input_batch)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
        num_size_batches.append(num_size_batch)
        bert_input_batches.append(bert_input_batch)
    return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches, bert_input_batches


def get_gts_batch(key, pair_batch):
    value_batch = []
    value_batch.append([pair_batch[idx][key] for idx in range(len(pair_batch))])
    return value_batch[0]

class MinibatchLoader():
    def __init__(self):
        self.batch_size = 0
        self.train_batch_nums = 0
        self.train_batches = {}
        self.test_datas = {}
        self.test_datas_size = 0
        self.valid_datas = {}
        self.valid_datas_size = 0
        self.input_lang = ''
        self.train_data_alias = []
        self.valid_data_alias = []
        self.test_data_alias = []

        '''
        self.seq_ids = gts_data_batches[0]
        self.seq_batches = gts_data_batches[1]
        self.seq_lengths = gts_data_batches[2]
        self.equation_batches = gts_data_batches[3]
        self.equation_lengths = gts_data_batches[4]
        self.num_value_batches = gts_data_batches[5]
        self.num_copy_batches = gts_data_batches[6]
        self.num_stack_batches = gts_data_batches[7]
        self.num_pos_batches = gts_data_batches[8]
        self.num_size_batches = gts_data_batches[9]
        self.num_group_batches = gts_data_batches[10]
        self.gts_graph_batches = gts_data_batches[11]

        self.enc_seq_batches = ibm_data_batches[0]
        self.enc_seq_length_batches = ibm_data_batches[1]
        self.enc_graph_batches = ibm_data_batches[2]
        self.enc_graph_length_batches = ibm_data_batches[3]
        self.dec_tree_batches = ibm_data_batches[4]
        '''
    def load_dataset(self, dataset_dir, dataset_name, batch_size, mode, input_lang):
        self.input_lang = input_lang
        if mode == "train":
            data_alias = 'seq_data'
            seq_data = self.load_pkl_data(dataset_dir, dataset_name, data_alias, mode, batch_size)
            self.train_batches[data_alias] = seq_data['data']
            self.batch_size = seq_data['batch_size']
            self.train_batch_nums = seq_data['batch_num']
            self.train_data_alias.append(data_alias)
            # load graph wang
            data_alias = 'seq_graph_wang'
            graph_wang = self.load_pkl_data(dataset_dir, dataset_name, data_alias, mode, batch_size)
            self.train_batches[data_alias] = graph_wang['data']
            self.train_data_alias.append(data_alias)
            # load graph Li
            data_alias = 'seq_graph_li'
            graph_li = self.load_pkl_data(dataset_dir, dataset_name, data_alias, mode, batch_size)
            self.train_batches[data_alias] = graph_li['data']
            self.train_data_alias.append(data_alias)

            return self.train_batch_nums
        elif mode == 'test':
            data_alias = 'seq_data'
            seq_data = self.load_pkl_data(dataset_dir, dataset_name, data_alias, mode)
            self.test_datas[data_alias] = seq_data['data']
            self.test_datas_size = len(seq_data['data'])
            self.test_data_alias.append(data_alias)
            # load graph wang
            data_alias = 'seq_graph_wang'
            graph_wang = self.load_pkl_data(dataset_dir, dataset_name, data_alias, mode)
            self.test_datas[data_alias] = graph_wang['data']
            self.test_data_alias.append(data_alias)
            # load graph Li
            data_alias = 'seq_graph_li'
            graph_li = self.load_pkl_data(dataset_dir, dataset_name, data_alias, mode)
            self.test_datas[data_alias] = graph_li['data']
            self.test_data_alias.append(data_alias)
            return self.test_datas_size
        else:
            data_alias = 'seq_data'
            seq_data = self.load_pkl_data(dataset_dir, dataset_name, data_alias, mode)
            self.valid_datas[data_alias] = seq_data['data']
            self.valid_datas_size = len(seq_data['data'])
            self.valid_data_alias.append(data_alias)
            # load graph wang
            data_alias = 'seq_graph_wang'
            graph_wang = self.load_pkl_data(dataset_dir, dataset_name, data_alias, mode)
            self.valid_datas[data_alias] = graph_wang['data']
            self.valid_data_alias.append(data_alias)
            # load graph Li
            data_alias = 'seq_graph_li'
            graph_li = self.load_pkl_data(dataset_dir, dataset_name, data_alias, mode)
            self.valid_datas[data_alias] = graph_li['data']
            self.valid_data_alias.append(data_alias)
            return self.valid_datas_size


    # dataset: pre_mode_seq.pkl, pre_mode_graph.pkl
    def load_pkl_data(self, data_dir, dataset_name,data_alias, mode, batch_size=0):
        data_file = ''
        if mode =='train':
            data_file = "{}/{}_{}_{}_{}.pkl".format(data_dir, dataset_name, data_alias, mode, batch_size)
        else:
            data_file = "{}/{}_{}_{}.pkl".format(data_dir, dataset_name, data_alias, mode)

        data = pkl.load(open(data_file, "rb" ))
        if len(data[0]) < 4:
            return {}
        else:
            return data[0]


    def read_graph_data(self, input_file):
        # transform the keys to string
        graphs_new = []
        with open(input_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                jo = json.loads(line, object_pairs_hook=OrderedDict)
                graph_i = {}
                graph_i['id'] = jo['id']
                graph_i['g_ids'] = jo['g_ids']
                graph_i['g_ids_features'] = jo['g_ids_features']
                graph_i['g_adj'] = jo['g_adj']
                graphs_new.append(graph_i)
        return graphs_new



    def random_batch(self, data_alias):
        p = randint(0,self.train_batch_nums-1)
        return self.get_batch(p, data_alias)


    def get_batch(self, batch_idx, data_alias):
        p = batch_idx
        if p < self.train_batch_nums:
            if data_alias and data_alias in self.train_data_alias:
                data_batch = self.train_batches[data_alias][p]
                return p, data_batch
        else:
            return -1, []

    def all_batch(self):
        return self.batch_size, self.train_batch_nums, self.train_batches

    def get_batch_nums(self):
        return self.train_batch_nums

    def get_batch_size(self):
        return self.batch_size

    def get_test_valid_pairs(self, tv_mode, data_alias, data_idx=-1):
        if tv_mode == 'test':
            return self.get_test_pairs(data_alias, data_idx)
        elif tv_mode == 'valid':
            return self.get_valid_pairs(data_alias, data_idx)
        else:
            return 0, []

    def get_test_valid_size(self, tv_mode):
        if tv_mode == 'test':
            return self.get_test_size()
        elif tv_mode == 'valid':
            return self.get_valid_size()
        else:
            return 0, []

    def get_test_pairs(self, data_alias, data_idx):
        if data_alias in self.test_data_alias:
            if data_idx == -1:
                return self.test_datas_size, self.test_datas[data_alias]
            elif data_idx < self.test_datas_size:
                return 1,self.test_datas[data_alias][data_idx]
        else:
            return -1, []

    def get_test_size(self):
        return self.test_datas_size

    def get_valid_pairs(self, data_alias, data_idx):
        if data_alias in self.valid_data_alias:
            if data_idx == -1:
                return self.valid_datas_size, self.valid_datas[data_alias]
            elif data_idx < self.valid_datas_size:
                return 1, self.valid_datas[data_alias][data_idx]
        else:
            return -1, []

    def get_valid_size(self):
        return self.valid_datas_size

    def get_batch_from_dic(self, key, dic):
        return get_gts_batch(key, dic)