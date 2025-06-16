from data_utils.pycorenlp import StanfordCoreNLP
import pickle as pkl
import os
import tqdm
import copy
import networkx as nx
from data_utils.pythonds.basic.stack import Stack
import numpy as np


word_size_max = 1
sample_size_per_layer = 10

_cut_root_node = True
_cut_line_node = True
_cut_pos_node = False
_link_word_nodes = False
_split_sentence = False

source_data_dir = "../TextData/"



class InputPreprocessor(object):
    def __init__(self, url = r'http://localhost:9000/', lanuage='zh'):
        # self.nlp = StanfordCoreNLP(url, memory='8g', lang=lanuage)
        self.nlp = StanfordCoreNLP('http://localhost:9000/')

    def featureExtract(self,src_text,whiteSpace=True):
        #if src_text.strip() in preparsed_file.keys():
        #    return preparsed_file[src_text.strip()]
        #print("miss!")
        #print(src_text)
        data = {}
        output = self.nlp.annotate(src_text.strip(), properties={
            'annotators': "tokenize,ssplit,pos,parse",
            "tokenize.options":"splitHyphenated=true,normalizeParentheses=false",
            "tokenize.whitespace": whiteSpace,
            'ssplit.isOneSentence': True,
            'outputFormat': 'json'})

        # nlp.annotate return a string,not a dictionay. use eval() to convert a string into a dictionary
        output_dic = eval(output)
        output = output_dic
        #print(output_dic['sentences'])
        snt = output['sentences'][0]["tokens"]
        depency = output['sentences'][0]["basicDependencies"]
        #parse = self.nlp.parse(src_text)   #适用于输入文本没有分词的情况，这个速度很慢，替换方案是先调用get_tokenize（）分词，再进行依赖分析
        data["tok"] = []
        data["pos"] = []
        data["dep"] = []
        data["governor"] = []
        data["dependent"] = []
        data['parse'] = output['sentences'][0]['parse'] #parse
        for snt_tok in snt:
            data["tok"].append(snt_tok['word'])
            data["pos"].append(snt_tok['pos'])
        for deps in depency:
            data["dep"].append(deps['dep'])
            data["governor"].append(deps['governor'])
            data["dependent"].append(deps['dependent'])
        return data
    # 分词
    def get_tokenize(self, original_text):
        word_token = self.nlp.word_tokenize(original_text)
        segmented_text = ' '.join(word_token)
        return word_token, segmented_text

    def close_server(self):
        self.nlp.close()

def get_preparsed_file():
    if not os.path.exists("./file_for_parsing.pkl"):
        file_for_parsing = {}
        processor_tmp = InputPreprocessor()

        with open(source_data_dir+"all.txt", "r") as f:
            lines = f.readlines()
            for l in tqdm.tqdm(lines):
                str_ = l.strip().split('\t')[0]
                file_for_parsing[str_] = processor_tmp.featureExtract(str_)

        pkl.dump(file_for_parsing, open("./file_for_parsing.pkl", "wb"))
    else:
        preparsed_file = pkl.load(open("./file_for_parsing.pkl", "rb"))
    return preparsed_file

class Node():
    def __init__(self, word, type_, id_):
        # word: this node's text
        self.word = word

        # type: 0 for word nodes, 1 for constituency nodes, 2 for dependency nodes(if they exists)
        self.type = type_

        # id: unique identifier for every node
        self.id = id_

        self.head = False

        self.tail = False

    def __str__(self):
        return self.word

def split_str(string):
    if " . " not in string:
        return [string]
    else:
        s_arr = string.split(" . ")
        res = []
        for s in s_arr:
            if s[-1] != "." and s != s_arr[-1]:
                s = s+" ."
            res.append(s)
        return res

def cut_root_node(con_string):
    tmp = con_string
    if con_string[0] == '(' and con_string[-1] == ')':
        tmp = con_string[1:-1].replace("ROOT", "")
        if tmp[0] == '\n':
            tmp = tmp[1:]
    return tmp

def cut_pos_node(g):
    node_arr = list(g.nodes())
    del_arr = []
    for n in node_arr:
        edge_arr = list(g.edges())
        cnt_in = 0
        cnt_out = 0
        for e in edge_arr:
            if n.id == e[0].id:
                cnt_out += 1
                out_ = e[1]
            if n.id == e[1].id:
                cnt_in += 1
                in_ = e[0]
        if cnt_in == 1 and cnt_out == 1 and out_.type == 0:
            del_arr.append((n, in_, out_))
    for d in del_arr:
        g.remove_node(d[0])
        g.add_edge(d[1], d[2])
    return g

def cut_line_node(g):
    node_arr = list(g.nodes())

    for n in node_arr:
        edge_arr = list(g.edges())
        cnt_in = 0
        cnt_out = 0
        for e in edge_arr:
            if n.id == e[0].id:
                cnt_out += 1
                out_ = e[1]
            if n.id == e[1].id:
                cnt_in += 1
                in_ = e[0]
        if cnt_in == 1 and cnt_out == 1:
            g.remove_node(n)
            #             print "remove", n
            g.add_edge(in_, out_)
    #             print "add_edge", in_, out_
    return g


def get_seq_nodes(g):
    res = []
    node_arr = list(g.nodes())
    for n in node_arr:
        if n.type == 0:
            res.append(copy.deepcopy(n))
    return sorted(res, key=lambda x:x.id)

def get_non_seq_nodes(g):
    res = []
    node_arr = list(g.nodes())
    for n in node_arr:
        if n.type != 0:
            res.append(copy.deepcopy(n))
    return sorted(res, key=lambda x:x.id)

def get_all_text(g):
    seq_arr = get_seq_nodes(g)
    nonseq_arr = get_non_seq_nodes(g)
    seq = [x.word for x in seq_arr]
    nonseq = [x.word for x in nonseq_arr]
    return seq + nonseq

def get_all_id(g):
    seq_arr = get_seq_nodes(g)
    nonseq_arr = get_non_seq_nodes(g)
    seq = [x.id for x in seq_arr]
    nonseq = [x.id for x in nonseq_arr]
    return seq + nonseq

def get_id2word(g):
    res = {}
    seq_arr = get_seq_nodes(g)
    nonseq_arr = get_non_seq_nodes(g)
    for x in seq_arr:
        res[x.id] = x.word
    for x in nonseq_arr:
        res[x.id] = x.word
    return res

def nodes_to_string(l):
    return " ".join([x.word for x in l])

def print_edges(g):
    edge_arr = list(g.edges())
    for e in edge_arr:
        print (e[0].word, e[1].word),(e[0].id, e[1].id)

def print_nodes(g, he_ta = False):
    nodes_arr = list(g.nodes())
    if he_ta:
        print ([(n.word, n.id, n.head, n.tail) for n in nodes_arr])
    else:
        print ([(n.word, n.id) for n in nodes_arr])

def graph_connect(a_, b_):
    a = copy.deepcopy(a_)
    b = copy.deepcopy(b_)
    max_id = 0
    for n in a.nodes():
        if n.id > max_id:
            max_id = n.id
    tmp = copy.deepcopy(b)
    for n in tmp.nodes():
        n.id += max_id

    res = nx.union(a, tmp)
    seq_nodes_arr = []
    for n in res.nodes():
        if n.type == 0:
            seq_nodes_arr.append(n)
    seq_nodes_arr.sort(key=lambda x:x.id)
    for idx in range(len(seq_nodes_arr)):
        if idx != len(seq_nodes_arr) - 1 and seq_nodes_arr[idx].tail == True:
            if seq_nodes_arr[idx + 1].head == True:
                res.add_edge(seq_nodes_arr[idx], seq_nodes_arr[idx + 1])
                res.add_edge(seq_nodes_arr[idx + 1], seq_nodes_arr[idx])
    return res

def get_vocab(g):
    a = set()
    for n in list(g.nodes()):
        a.add(n.word)
    return a

def get_adj(g):
    #reverse the direction
    adj_dict = {}
    for node, n_dict in g.adjacency():
        adj_dict[node.id] = []

    for node, n_dict in g.adjacency():
        for i in n_dict.items():
            adj_dict[i[0].id].append(node.id)
    return adj_dict

def get_constituency_graph(input_tmp):
    tmp_result = input_tmp

    if _cut_root_node:
        parse_str = cut_root_node(str(tmp_result['parse']))
    else:
        parse_str = str(tmp_result['parse'])
    for punc in ['(',')']:
        parse_str = parse_str.replace(punc,' ' + punc + ' ')
    parse_list = str(parse_str).split()

    res_graph = nx.DiGraph()
    pstack = Stack()
    idx = 0
    while idx < len(parse_list):
        if parse_list[idx] == '(':
            new_node = Node(word=parse_list[idx+1], id_=idx+1, type_=1)
            res_graph.add_node(new_node)
            pstack.push(new_node)

            if pstack.size() > 1:
                node_2 = pstack.pop()
                node_1 = pstack.pop()
                res_graph.add_edge(node_1, node_2)
                pstack.push(node_1)
                pstack.push(node_2)
        elif parse_list[idx] == ')':
            pstack.pop()
        elif parse_list[idx] in tmp_result['tok']:
            new_node = Node(word=parse_list[idx], id_=idx, type_=0)
            node_1 = pstack.pop()
            if node_1.id != new_node.id:
                res_graph.add_edge(node_1, new_node)
            pstack.push(node_1)
        idx += 1

    max_id = 0
    for n in res_graph.nodes():
        if n.type == 0 and n.id > max_id:
            max_id = n.id

    min_id = 99999
    for n in res_graph.nodes():
        if n.type == 0 and n.id < min_id:
            min_id = n.id

    for n in res_graph.nodes():
        if n.type == 0 and n.id == max_id:
            n.tail = True
        if n.type == 0 and n.id == min_id:
            n.head = True
    return res_graph

def create_batch_graph_src(id_batch, string_batch, processor):

    # generate constituency graph
    graph_list = []
    # processor = InputPreprocessor()
    max_node_size = 0
    for s in string_batch:
        #print(s)
        # generate multiple graph
        if _split_sentence:
            s_arr = split_str(s)

            g = cut_line_node(get_constituency_graph(processor.featureExtract(s_arr[0])))
            for sub_s in s_arr:
                if sub_s != s_arr[0]:
                    tmp = cut_line_node(get_constituency_graph(processor.featureExtract(sub_s)))
                    g = graph_connect(g, tmp)

        # decide how to cut nodes
        if _cut_pos_node:
            g = cut_pos_node(get_constituency_graph(processor.featureExtract(s)))
        elif _cut_line_node:
            g = cut_line_node(get_constituency_graph(processor.featureExtract(s)))
        else:
            g = (get_constituency_graph(processor.featureExtract(s)))

        if len(list(g.nodes())) > max_node_size:
            max_node_size = len(list(g.nodes()))
        graph_list.append(g)

    info_list = []
    batch_size = len(string_batch)
    for index in range(batch_size):
        word_list = get_all_text(graph_list[index])
        word_len = len(get_seq_nodes(graph_list[index]))
        id_arr = get_all_id(graph_list[index])
        adj_dic = get_adj(graph_list[index])
        new_dic = {}

        # transform id to position in wordlist
        for k in adj_dic.keys():
            new_dic[id_arr.index(k)] = [id_arr.index(x) for x in adj_dic[k]]

        info = {}

        g_ids = {}
        g_ids_features = {}
        g_adj = {}

        for idx in range(max_node_size):
            g_ids[idx] = idx
            if idx < len(word_list):
                g_ids_features[idx] = word_list[idx]

                if _link_word_nodes:
                    if idx <= word_len - 1:
                        if idx == 0:
                            new_dic[idx].append(idx + 1)
                        elif idx == word_len - 1:
                            new_dic[idx].append(idx - 1)
                        else:
                            new_dic[idx].append(idx - 1)
                            new_dic[idx].append(idx + 1)

                g_adj[idx] = new_dic[idx]
            else:
                g_ids_features[idx] = '<P>'
                g_adj[idx] = []

        #info['id'] = id_batch[index]
        info['g_ids'] = g_ids
        info['g_ids_features'] = g_ids_features
        info['g_adj'] = g_adj


        info_list.append(info)

    batch_vocab = []
    for x in graph_list:
        non_arr = nodes_to_string(get_non_seq_nodes(x)).split()
        for w in non_arr:
            if w not in batch_vocab:
                batch_vocab.append(w)
    return info_list, batch_vocab

def create_batch_graph_full(input_lang, enc_w_list, graph_datas):
    enc_seq_length = []
    enc_graph_list = []
    enc_graph_length = []

    batch_graph = graph_datas
    combine_batch_graph = cons_batch_graph(batch_graph)
    vector_batch_graph = vectorize_batch_graph(combine_batch_graph,input_lang)
    batch_seq_len = [len(w) for w in enc_w_list]

    enc_seq_length.append(batch_seq_len)
    enc_graph_list.append(vector_batch_graph)
    enc_graph_length.append(len(batch_graph[0]['g_ids']))

    return enc_w_list, batch_seq_len, vector_batch_graph, len(batch_graph[0]['g_ids'])

def vocab_from_denpendency_parsing(string_batch, processor):
    # generate constituency graph
    graph_list = []
    # processor = InputPreprocessor()
    max_node_size = 0
    for s in string_batch:
        #print(s)
        # generate multiple graph
        if _split_sentence:
            s_arr = split_str(s)

            g = cut_line_node(get_constituency_graph(processor.featureExtract(s_arr[0])))
            for sub_s in s_arr:
                if sub_s != s_arr[0]:
                    tmp = cut_line_node(get_constituency_graph(processor.featureExtract(sub_s)))
                    g = graph_connect(g, tmp)

        # decide how to cut nodes
        if _cut_pos_node:
            g = cut_pos_node(get_constituency_graph(processor.featureExtract(s)))
        elif _cut_line_node:
            g = cut_line_node(get_constituency_graph(processor.featureExtract(s)))
        else:
            g = (get_constituency_graph(processor.featureExtract(s)))

        if len(list(g.nodes())) > max_node_size:
            max_node_size = len(list(g.nodes()))
        graph_list.append(g)

    batch_vocab = []
    for x in graph_list:
        non_arr = nodes_to_string(get_non_seq_nodes(x)).split()
        for w in non_arr:
            if w not in batch_vocab:
                batch_vocab.append(w)
    return batch_vocab



def get_new_vocab(string_batch, processor):

    # generate constituency graph
    graph_list = []
    # processor = InputPreprocessor()
    max_node_size = 0
    for s in string_batch:

        # generate multiple graph
        if _split_sentence:
            s_arr = split_str(s)

            g = cut_line_node(get_constituency_graph(processor.featureExtract(s_arr[0])))
            for sub_s in s_arr:
                if sub_s != s_arr[0]:
                    tmp = cut_line_node(get_constituency_graph(processor.featureExtract(sub_s)))
                    g = graph_connect(g, tmp)

        # decide how to cut nodes
        if _cut_pos_node:
            g = cut_pos_node(get_constituency_graph(processor.featureExtract(s)))
        elif _cut_line_node:
            g = cut_line_node(get_constituency_graph(processor.featureExtract(s)))
        else:
            g = (get_constituency_graph(processor.featureExtract(s)))

        if len(list(g.nodes())) > max_node_size:
            max_node_size = len(list(g.nodes()))
        graph_list.append(g)

    batch_vocab = []
    for x in graph_list:
        non_arr = nodes_to_string(get_non_seq_nodes(x)).split()
        for w in non_arr:
            if w not in batch_vocab:
                batch_vocab.append(w)
    return batch_vocab



def generate_vocab_by_parser(train_pairs,batch_size):
    vocab_from_parser=[]
    index = 0
    batch_dim = len(train_pairs)//batch_size
    processor = InputPreprocessor()
    while index + batch_size <= len(train_pairs):
        # generate graphs with order and dependency information
        print ("{}/{}".format(index/batch_size, batch_dim))
        input_batch = ["".join(train_pairs[index + idx]['input_seq']) for idx in range(batch_size)]
        new_vocab = vocab_from_denpendency_parsing(input_batch, processor)
        index += batch_size
        vocab_from_parser.append(new_vocab)

    if index != len(train_pairs):
        input_batch = ["".join(train_pairs[idx]['input_seq']) for idx in range(index,len(train_pairs))]
        new_vocab = vocab_from_denpendency_parsing(input_batch, processor)
        vocab_from_parser.append(new_vocab)
    processor.close_server()

    return vocab_from_parser

def generate_denpendency_graph_batches(data_batches,input_lang):
    out_graph_batches = []
    index = 0
    processor = InputPreprocessor()
    batch_nums = len(data_batches)
    print('Prepare graph li data begin...')
    for batch in data_batches:
        print('batch {}/{}...'.format(index, batch_nums))
        index += 1
        batch_size = len(batch)
        input_batch = [" ".join(batch[idx]['input_seq']) for idx in range(batch_size)]
        id_batch = [data['id'] for data in batch]
        dec_tree_batch = [data['out_tree'] for data in batch]
        graph_batch, new_vocab = create_batch_graph_src(id_batch, input_batch, processor)
        #input_lang.add_sen_to_vocab(new_vocab)

        enc_seq_batch, enc_seq_length_batch, enc_graph_batch, enc_graph_length_batch = \
            create_batch_graph_full(input_lang, input_batch, graph_batch)
        g = {}
        g['enc_id_batch'] = id_batch
        g['enc_seq_batch'] = enc_seq_batch
        g['enc_seq_length_batch'] = enc_seq_length_batch
        g['enc_graph_batch'] = enc_graph_batch
        g['enc_graph_length_batch'] = enc_graph_length_batch
        g['dec_tree_batch'] = dec_tree_batch
        out_graph_batches.append(g)
    processor.close_server()
    print('Prepare graph li data finish...')
    return out_graph_batches, input_lang

def create_graph_single_li(data_batch, input_lang):
    out_graph_li_batch = []
    processor = InputPreprocessor()
    # k=0
    for seq_data in data_batch:
        id = [seq_data['id']]
        input_seq = [" ".join(seq_data['input_seq'])]
        dec_tree_batch = [seq_data['out_tree']]
        # print("k:", k)
        graph, new_vocab = create_batch_graph_src(id, input_seq, processor)
        #input_lang.add_sen_to_vocab(new_vocab)
        enc_seq_batch, enc_seq_length_batch, enc_graph_batch, enc_graph_length_batch = \
            create_batch_graph_full(input_lang, input_seq, graph)
        g = {}
        g['enc_id_batch'] = id
        g['enc_seq_batch'] = enc_seq_batch
        g['enc_seq_length_batch'] = enc_seq_length_batch
        g['enc_graph_batch'] = enc_graph_batch
        g['enc_graph_length_batch'] = enc_graph_length_batch
        g['dec_tree_batch'] = dec_tree_batch
        out_graph_li_batch.append(g)
        # k=k+1
    processor.close_server()
    return out_graph_li_batch, input_lang

def cons_batch_graph(graphs):
    g_ids = {}
    g_ids_features = {}
    g_fw_adj = {}
    g_bw_adj = {}
    g_nodes = []
    for g in graphs:
        #g = collections.OrderedDict(g)
        ids = g['g_ids']
        id_adj = g['g_adj']
        features = g['g_ids_features']
        nodes = []

        # we first add all nodes into batch_graph and create a mapping from graph id to batch_graph id, this mapping will be
        # used in the creation of fw_adj and bw_adj

        id_gid_map = {}
        offset = len(g_ids.keys())
        for id in ids:
            id = int(id)
            g_ids[offset + id] = len(g_ids.keys())
            g_ids_features[offset + id] = features[id]
            id_gid_map[id] = offset + id
            nodes.append(offset + id)
        g_nodes.append(nodes)

        for id in id_adj:
            adj = id_adj[id]
            id = int(id)
            g_id = id_gid_map[id]
            if g_id not in g_fw_adj:
                g_fw_adj[g_id] = []
            for t in adj:
                t = int(t)
                g_t = id_gid_map[t]
                g_fw_adj[g_id].append(g_t)
                if g_t not in g_bw_adj:
                    g_bw_adj[g_t] = []
                g_bw_adj[g_t].append(g_id)

    node_size = len(g_ids.keys())
    for id in range(node_size):
        if id not in g_fw_adj:
            g_fw_adj[id] = []
        if id not in g_bw_adj:
            g_bw_adj[id] = []

    graph = {}
    graph['g_ids'] = g_ids
    graph['g_ids_features'] = g_ids_features
    graph['g_nodes'] = g_nodes
    graph['g_fw_adj'] = g_fw_adj
    graph['g_bw_adj'] = g_bw_adj
    return graph

def vectorize_batch_graph(graph, input_lang):
    # vectorize the graph feature and normalize the adj info
    id_features = graph['g_ids_features']
    gv = {}
    nv = []
    word_max_len = 0
    for id in id_features:
        feature = id_features[id]
        word_max_len = max(word_max_len, len(feature.split()))
    word_max_len = min(word_max_len,  word_size_max)

    for id in range(len(graph['g_ids_features'])):
        feature = graph['g_ids_features'][id]
        fv = []
        for token in feature.split():
            if len(token) == 0:
                continue
            # if token in word_idx:
            #     fv.append(word_idx[token])
            # else:
            #     fv.append(word_idx['<U>'])
            fv.append(input_lang.get_symbol_idx(token))
        for _ in range(word_max_len - len(fv)):
            fv.append(0)
        fv = fv[:word_max_len]
        nv.append(fv)

    nv.append([0 for temp in range(word_max_len)])
    gv['g_ids_features'] = np.array(nv)

    g_fw_adj = graph['g_fw_adj']
    g_fw_adj_v = []

    degree_max_size = 0
    for id in g_fw_adj:
        degree_max_size = max(degree_max_size, len(g_fw_adj[id]))

    g_bw_adj = graph['g_bw_adj']
    for id in g_bw_adj:
        degree_max_size = max(degree_max_size, len(g_bw_adj[id]))

    degree_max_size = min(degree_max_size, sample_size_per_layer)
    #degree_max_size = sample_size_per_layer
    for id in g_fw_adj:
        adj = g_fw_adj[id]
        for _ in range(degree_max_size - len(adj)):
            adj.append(len(g_fw_adj.keys()))
        adj = adj[:degree_max_size]
        g_fw_adj_v.append(adj)

    # PAD node directs to the PAD node
    g_fw_adj_v.append([len(g_fw_adj.keys()) for _ in range(degree_max_size)])

    g_bw_adj_v = []
    for id in g_bw_adj:
        adj = g_bw_adj[id]
        for _ in range(degree_max_size - len(adj)):
            adj.append(len(g_bw_adj.keys()))
        adj = adj[:degree_max_size]
        g_bw_adj_v.append(adj)

    # PAD node directs to the PAD node
    g_bw_adj_v.append([len(g_bw_adj.keys()) for _ in range(degree_max_size)])

    gv['g_ids'] = graph['g_ids']
    gv['g_nodes'] =np.array(graph['g_nodes'])
    gv['g_bw_adj'] = np.array(g_bw_adj_v)
    gv['g_fw_adj'] = np.array(g_fw_adj_v)
    return gv
