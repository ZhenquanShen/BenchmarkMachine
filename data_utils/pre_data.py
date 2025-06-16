import random
import copy
import re
from data_utils import graph_utils_li as graph_li
from data_utils import graph_utils_wang as graph_wang
from data_utils import tree
PAD_token = 0

class Lang:
    """
    class to save the vocab and two dict: the word->index and index->word
    """
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  # Count word tokens
        self.num_start = 0
        self.vocab_size = 0    
        # PAD: padding token = 0
        self.add_symbol('<P>')
        # GO: start token = 1
        self.add_symbol('<S>')
        # EOS: end token = 2
        self.add_symbol('<E>')
        # NON: non-terminal token = 3
        self.add_symbol('<N>')
        # operator + = 4
        self.add_symbol('+')
        # operator + = 5
        self.add_symbol('-')
        # operator + = 6
        self.add_symbol('*')
        # operator + = 7
        self.add_symbol('/')
        # operator ^ = 8
        self.add_symbol('^')
        # ( = 9
        self.add_symbol('(')
        # ) = 10
        self.add_symbol(')')
        # UNK: unknown token = 11
        #self.add_symbol('<U>')
        self.whether_add_special_tags = True

    def add_symbol(self,s):
        if s not in self.index2word:
            self.word2index[s] = self.vocab_size
            self.word2count[s] = 1
            self.index2word.append(s)
            self.vocab_size += 1
        else:
            self.word2count[s] += 1
    def add_unk_symbol(self,):
        self.add_symbol("<U>")

    def add_sen_to_vocab(self, sentence):  # add words of sentence to vocab
        for word in sentence:
            if re.search("N\d+|NUM|\d+", word):
                continue
            if word not in self.index2word:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word.append(word)
                self.n_words += 1
            else:
                self.word2count[word] += 1

    def trim(self, min_count):  # trim words below a certain count threshold
        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.index2word), len(keep_words) / len(self.index2word)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  # Count default tokens

        for word in keep_words:
            self.word2index[word] = self.n_words
            self.index2word.append(word)
            self.n_words += 1

    def build_input_lang(self, trim_min_count):  # build the input lang vocab and dict
        if trim_min_count > 0:
            self.trim(trim_min_count)
            self.index2word = ["PAD", "NUM", "UNK"] + self.index2word
        else:
            self.index2word = ["PAD", "NUM"] + self.index2word
        self.word2index = {}
        self.n_words = len(self.index2word)
        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

    def build_output_lang(self, generate_num, copy_nums):  # build the output lang vocab and dict
        self.index2word = ["PAD", "EOS"] + self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)] +\
                          ["SOS", "UNK"]
        self.n_words = len(self.index2word)
        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

    def build_output_lang_for_tree(self, generate_num, copy_nums):  # build the output lang vocab and dict
        self.num_start = len(self.index2word)

        self.index2word = self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)] + ["UNK"]
        self.n_words = len(self.index2word)

        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

    def get_symbol_idx(self,s):
        if s not in self.index2word:
            if self.whether_add_special_tags:
                return self.word2index['<U>']
            else:
                print("not reached!")
                return 0
        return self.word2index[s]

    def get_idx_symbol(self, idx):
        if idx not in self.index2word:
            return '<U>'
        return self.index2word[idx]
# class Lang:
#     """
#     class to save the vocab and two dict: the word->index and index->word
#     """
#     def __init__(self , whether_add_special_tags = True):
#         self.symbol2idx = {}
#         self.symbol2count = {}
#         self.idx2symbol = []
#         self.vocab_size = 0
#         self.whether_add_special_tags = whether_add_special_tags
#         self.generate_nums = []
#         self.copy_nums = 0
#         if whether_add_special_tags:
#             # PAD: padding token = 0
#             self.add_symbol('<P>')
#             # GO: start token = 1
#             self.add_symbol('<S>')
#             # EOS: end token = 2
#             self.add_symbol('<E>')
#             # NON: non-terminal token = 3
#             self.add_symbol('<N>')
#             # operator + = 4
#             self.add_symbol('+')
#             # operator + = 5
#             self.add_symbol('-')
#             # operator + = 6
#             self.add_symbol('*')
#             # operator + = 7
#             self.add_symbol('/')
#             # operator ^ = 8
#             self.add_symbol('^')
#             # ( = 9
#             self.add_symbol('(')
#             # ) = 10
#             self.add_symbol(')')
#             # UNK: unknown token = 11
#             #self.add_symbol('<U>')
#         self.num_start = self.vocab_size


#     def add_symbol(self,s):
#         if s not in self.idx2symbol:
#             self.symbol2idx[s] = self.vocab_size
#             self.symbol2count[s] = 1
#             self.idx2symbol.append(s)
#             self.vocab_size += 1
#         else:
#             self.symbol2count[s] += 1
#     def add_unk_symbol(self,):
#         self.add_symbol("<U>")

#     def add_sen_to_vocab(self, sentence):  # add words of sentence to vocab
#         for word in sentence:
#             #if re.search("N\d+|NUM|\d+", word):
#             #    continue
#            self.add_symbol(word)

#     def trim_symbol(self, min_count):  # trim words below a certain count threshold
#         keep_words = []

#         for k, v in self.symbol2count.items():
#             if v >= min_count:
#                 keep_words.append(k)

#         print('keep_words %s / %s = %.4f' % (
#             len(keep_words), len(self.idx2symbol), len(keep_words) / len(self.idx2symbol)
#         ))

#         # Reinitialize dictionaries
#         self.symbol2idx = {}
#         self.symbol2count = {}
#         self.idx2symbol = []
#         self.vocab_size = 0  # Count default tokens

#         for word in keep_words:
#             self.symbol2idx[word] = self.vocab_size
#             self.idx2symbol.append(word)
#             self.vocab_size += 1

#     def build_input_lang(self, trim_min_count):  # build the input lang vocab and dict
#         if trim_min_count > 0:
#             #self.trim(trim_min_count)
#             self.idx2symbol = ["<P>", "<U>"] + self.idx2symbol
#         else:
#             self.idx2symbol = ["<P>", "<U>"] + self.idx2symbol
#         self.symbol2idx = {}
#         self.vocab_size = len(self.idx2symbol)
#         for i, j in enumerate(self.idx2symbol):
#             self.symbol2idx[j] = i

#     def build_output_lang(self, generate_num, copy_nums):  # build the output lang vocab and dict
#         for gn in generate_num:
#             self.add_symbol(gn)
#         for i in range(copy_nums):
#             ni ="N" + str(i)
#             self.add_symbol(ni)

#     def init_from_file(self, fn, min_freq, max_vocab_size):
#         # the vocab file is sorted by word_freq
#         print ("loading vocabulary file: {}".format(fn))
#         line_id = 0
#         with open(fn,"r", encoding='utf-8') as f:
#             for line in f:
#                 #print("line_id: {}".format(line_id))
#                 line_id += 1
#                 l_list = line.strip().split(' ')
#                 #print(l_list)
#                 if len(l_list) == 1:
#                     l_list.insert(0,' ')
#                 c = int(l_list[1])
#                 if c >= min_freq:
#                     self.add_symbol(l_list[0])
#                 if self.vocab_size > max_vocab_size:
#                     break


#     def get_symbol_idx_for_list(self,l):
#         r = []
#         for i in range(len(l)):
#             r.append(self.get_symbol_idx(l[i]))
#         return r

#     def get_symbol_idx(self,s):
#         if s not in self.idx2symbol:
#             if self.whether_add_special_tags:
#                 return self.symbol2idx['<U>']
#             else:
#                 print("not reached!")
#                 return 0
#         return self.symbol2idx[s]

#     def get_idx_symbol(self, idx):
#         if idx not in self.idx2symbol:
#             return '<U>'
#         return self.idx2symbol[idx]
class Lang1:
    """
    class to save the vocab and two dict: the word->index and index->word
    """
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  # Count word tokens
        self.num_start = 0

    def add_sen_to_vocab(self, sentence):  # add words of sentence to vocab
        for word in sentence:
            if re.search("N\d+|NUM|\d+", word):
                continue
            if word not in self.index2word:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word.append(word)
                self.n_words += 1
            else:
                self.word2count[word] += 1

    def trim(self, min_count):  # trim words below a certain count threshold
        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.index2word), len(keep_words) / len(self.index2word)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  # Count default tokens

        for word in keep_words:
            self.word2index[word] = self.n_words
            self.index2word.append(word)
            self.n_words += 1

    def build_input_lang(self, trim_min_count):  # build the input lang vocab and dict
        if trim_min_count > 0:
            self.trim(trim_min_count)
            self.index2word = ["PAD", "NUM", "UNK"] + self.index2word
        else:
            self.index2word = ["PAD", "NUM"] + self.index2word
        self.word2index = {}
        self.n_words = len(self.index2word)
        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

    def build_output_lang(self, generate_num, copy_nums):  # build the output lang vocab and dict
        self.index2word = ["PAD", "EOS"] + self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)] +\
                          ["SOS", "UNK"]
        self.n_words = len(self.index2word)
        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

    def build_output_lang_for_tree(self, generate_num, copy_nums):  # build the output lang vocab and dict
        self.num_start = len(self.index2word)

        self.index2word = self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)] + ["UNK"]
        self.n_words = len(self.index2word)

        for i, j in enumerate(self.index2word):
            self.word2index[j] = i



def from_infix_to_prefix(expression):
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


# remove the superfluous brackets
def remove_brackets(x):
    y = x
    if x[0] == "(" and x[-1] == ")":
        x = x[1:-1]
        flag = True
        count = 0
        for s in x:
            if s == ")":
                count -= 1
                if count < 0:
                    flag = False
                    break
            elif s == "(":
                count += 1
        if flag:
            return x
    return y


def transfer_num(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    for d in data:
        nums = []
        input_seq = []
        seg = d["segmented_text"].strip().split(" ")
        equation = d["equation"][2:]
        id = d["id"]
        ans = d["ans"]

        #replace "[","]" to "(",")"
        equation = ['(' if x== '[' else x for x in equation]
        equation = [')' if x== ']' else x for x in equation]
        equation = ''.join(equation)
        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
                #input_seq.append("NUM")
                input_seq.append("N"+str(len(nums)-1))
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            elif pos and pos.start() > 0:
                nums.append(s[pos.start(): pos.end()])
                input_seq.append(s[: pos.start()])
                #input_seq.append("NUM")
                input_seq.append("N"+str(len(nums)-1))
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)
        if copy_nums < len(nums):
            copy_nums = len(nums)

        nums_fraction = []

        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) > 0:
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
                if nums.count(st_num) > 0:
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
        num_list = ["N"+str(idx) for idx in range(50)]
        for i, j in enumerate(input_seq):
            #if j == "NUM":
            if j in num_list:    #"NUM":
                num_pos.append(i)
        assert len(nums) == len(num_pos)
        pair = {'id': id,'input_seq':input_seq,'out_equ':equation,'out_seq':out_seq,
                'nums':nums, 'num_pos':num_pos, 'num_pos_len':len(num_pos),'ans':ans}
        pairs.append(pair)

    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)
    return pairs, temp_g, copy_nums

#for example, x=(1*0.01)*2 will be converted to [x,=, [1,*,0.01], *, 2]
def convert_to_tree(r_list, i_left, i_right, output_lang):
    t = tree.Tree()
    level = 0
    left = -1
    for i in range(i_left, i_right):

        if r_list[i] == output_lang.word2index['(']:
            if level == 0:
                left = i
            level = level + 1
        elif r_list[i] == output_lang.word2index[')']:
            level = level -1
            if level == 0:
                if i == left+1:
                    c = r_list[i]
                else:
                    c = convert_to_tree(r_list, left + 1, i, output_lang)
                t.add_child(c)
        elif level == 0:
            t.add_child(r_list[i])
    return t

def attach_group_data(pairs_data, group_data):
    updated_pairs_data = []
    group_data_id = [item['id'] for item in group_data]
    for pair in pairs_data:
        id = pair['id']
        idx = group_data_id.index(id) if (id in group_data_id) else -1
        if idx == -1:
            pair.update({'group_num':[]})
        else:
            group = group_data[idx]
            pair.update({'group_num':group['group_num']})
        updated_pairs_data.append(pair)
    return updated_pairs_data



# Return a list of indexes, one for each word in the sentence, plus EOS
def indexes_from_sentence(lang, sentence, tree=False):
    res = []
    for word in sentence:
        if len(word) == 0:
            continue
        if word == '[':
            word = '('
        elif word == ']':
            word = ')'

        if word in lang.index2word:
            res.append(lang.word2index[word])
        else:
            res.append(lang.word2index["<U>"])
    if "<E>" in lang.index2word and not tree:
        res.append(lang.word2index["<E>"])
    return res

def prepare_data(input_lang, output_lang, pairs_data, trim_min_count, generate_nums, copy_nums, tree=False):
    prepared_pairs_data = []
    print("Indexing words...")

    for pair in pairs_data:
        num_stack = []
        for word in pair['out_seq']:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair['nums']):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair['nums']))])

        num_stack.reverse()
        input_cell = indexes_from_sentence(input_lang, pair['input_seq'])
        output_cell = indexes_from_sentence(output_lang, pair['out_seq'], tree)
        output_cell_prefix = indexes_from_sentence(output_lang,from_infix_to_prefix(pair['out_seq']), tree)

        pair.update({'input_cell':input_cell, 'input_cell_len':len(input_cell),
                     'out_cell_prefix':output_cell_prefix,'out_cell':output_cell,
                     'num_stack':num_stack})
        #train_pairs.append([pair[0], input_cell, len(input_cell), output_cell, len(output_cell),
        #                    pair[3], pair[4], num_stack, pair[5], pair[6]])
        prepared_pairs_data.append(pair)
    print('Indexed %d words in input language, %d words in output' % (input_lang.vocab_size, output_lang.vocab_size))
    print('Number of training data %d' % (len(prepared_pairs_data)))
    return prepared_pairs_data

def prepare_tree(data_pairs, output_lang):
    prepared_pairs = []
    for data_pair in data_pairs:
        out_cell = data_pair['out_cell']
        out_tree = convert_to_tree(out_cell, 0, len(out_cell), output_lang)
        data_pair.update({'out_tree':out_tree})
        prepared_pairs.append(data_pair)
    return prepared_pairs

def get_invalid_data_ids(data_pairs):
    exp_chinese_char = '[\u4e00-\u9fa5]'
    ex_math_exp = '^[-+]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][-+]?[0-9]+)?$'
    pattern = re.compile(exp_chinese_char)
    ids = []
    for pair in data_pairs:
        id = pair['id']
        sentence = pair['input_seq']
        word_len = 0
        for word in sentence:
            if pattern.search(word):
                word_len +=1
        if word_len < 5:
            ids.append(id)
    return ids

#移除掉所有计算式求解题目
def remove_invalid_data(data_pairs):
    new_data_pairs = []
    invalid_ids = get_invalid_data_ids(data_pairs)
    for pair in data_pairs:
        id = pair['id']
        if not (id in invalid_ids):
            new_data_pairs.append(pair)
    return new_data_pairs



# Pad a with the PAD symbol
def pad_seq(seq, seq_len, max_length):
    seq += [PAD_token for _ in range(max_length - seq_len)]
    return seq

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

def seq_data_batch_packeting(batch_size, seq_pairs, sort=True):
    pos = 0
    data_batches = []
    while pos + batch_size <= len(seq_pairs):
        batch = copy.deepcopy(seq_pairs[pos:pos+batch_size])
        if sort:
            batch = sorted(batch, key=lambda tp: tp['input_cell_len'], reverse=True)
        #padding input_cell and out_cell_prefix
        input_cell_lens = [batch[idx]['input_cell_len'] for idx in range(len(batch))]
        out_cell_batch = [batch[idx]['out_cell_prefix'] for idx in range(len(batch))]
        out_cell_lens = [len(out_cell) for out_cell in out_cell_batch]
        input_cell_len_max = max(input_cell_lens)
        out_cell_len_max = max(out_cell_lens)
        for idx in range(len(batch)):
            data_item = batch[idx]
            input_cell_len = data_item['input_cell_len']
            input_cell = data_item['input_cell']
            out_cell_len = len(data_item['out_cell_prefix'])
            out_cell = data_item['out_cell_prefix']
            batch[idx]['input_cell'] = pad_seq(input_cell, input_cell_len, input_cell_len_max)
            batch[idx]['out_cell_prefix'] = pad_seq(out_cell, out_cell_len, out_cell_len_max)
        data_batches.append(batch)
        pos += batch_size
    return data_batches

def create_graph_batch_wang(train_batches):
    print('prepare graph wang begin...')
    graph_wang_batches = []
    for data_batch in train_batches:
        new_data_batch = []
        max_word_length = max(get_gts_batch('input_cell_len', data_batch))
        for data in data_batch:
            g = {}
            g['graph_wang'] = prepare_g2t_graph_single(data, max_word_length)
            data.update(g)
            new_data_batch.append(data)
        graph_wang_batches.append(new_data_batch)
    print('prepare graph wang end...')
    return graph_wang_batches
def create_graph_single_wang(input_batch):
    graph_batch = []
    for data in input_batch:
        g = {}
        g['graph_wang'] = prepare_g2t_graph_single(data, data['input_cell_len'])
        data.update(g)
        graph_batch.append(data)
    return graph_batch

def prepare_ibm_data_batch(input_lang, enc_w_list, tree_list, graph_datas):
    enc_seq_length = []
    enc_graph_list = []
    enc_graph_length = []

    batch_graph = graph_datas
    combine_batch_graph = graph_li.cons_batch_graph(batch_graph)
    vector_batch_graph = graph_li.vectorize_batch_graph(combine_batch_graph,input_lang)
    batch_seq_len = [len(w) for w in enc_w_list]

    enc_seq_length.append(batch_seq_len)
    enc_graph_list.append(vector_batch_graph)
    enc_graph_length.append(len(batch_graph[0]['g_ids']))

    return enc_w_list, batch_seq_len, vector_batch_graph, len(batch_graph[0]['g_ids']), tree_list 
def create_graph_single_li(seq_batch, graph_batch, input_lang):
    out_graph_li_batch = []
    for seq_data, graph_data in zip(seq_batch, graph_batch):
        graph_batch = graph_li.cons_batch_graph([graph_data])
        graph = graph_li.vectorize_batch_graph(graph_batch, input_lang)
        g = {}
        g['enc_seq_batch'] = seq_data['input_seq']
        g['enc_seq_length_batch'] = seq_data['input_cell_len']
        g['enc_graph_batch'] = graph
        g['enc_graph_length_batch'] = len(g['enc_graph_batch'])
        out_graph_li_batch.append(g)
    return out_graph_li_batch



# conver diction to list
def get_gts_batch(key, pair_batch):
    value_batch = []
    value_batch.append([pair_batch[idx][key] for idx in range(len(pair_batch))])
    return value_batch[0]


# prepare graph for graph2tree
def prepare_g2t_graph_batch(data_batch):
    graph_batch = graph_wang.get_single_batch_graph(get_gts_batch('input_cell', data_batch),
                                                    get_gts_batch('input_cell_len', data_batch),
                                                    get_gts_batch('group_num', data_batch),
                                                    get_gts_batch('nums', data_batch),
                                                    get_gts_batch('num_pos', data_batch))
    return graph_batch



def prepare_g2t_graph_single(data, max_input_length):
    single_graph = graph_wang.get_single_example_graph(data['input_cell'], data['input_cell_len'],
                                                       max_input_length,data['out_cell'],
                                                       data['nums'], data['num_pos'])
    return single_graph


def get_num_stack(eq, output_lang, num_pos):
    num_stack = []
    for word in eq:
        temp_num = []
        flag_not = True
        if word not in output_lang.idx2symbol:
            flag_not = False
            for i, j in enumerate(num_pos):
                if j == word:
                    temp_num.append(i)
        if not flag_not and len(temp_num) != 0:
            num_stack.append(temp_num)
        if not flag_not and len(temp_num) == 0:
            num_stack.append([_ for _ in range(len(num_pos))])
    num_stack.reverse()
    return num_stack

# Multiplication exchange rate
def exchange(ex_copy, rate):
    ex = copy.deepcopy(ex_copy)
    idx = 1
    while idx < len(ex):
        s = ex[idx]
        if (s == "*" or s == "+") and random.random() < rate:
            lidx = idx - 1
            ridx = idx + 1
            if s == "+":
                flag = 0
                while not (lidx == -1 or ((ex[lidx] == "+" or ex[lidx] == "-") and flag == 0) or flag == 1):
                    if ex[lidx] == ")" or ex[lidx] == "]":
                        flag -= 1
                    elif ex[lidx] == "(" or ex[lidx] == "[":
                        flag += 1
                    lidx -= 1
                if flag == 1:
                    lidx += 2
                else:
                    lidx += 1

                flag = 0
                while not (ridx == len(ex) or ((ex[ridx] == "+" or ex[ridx] == "-") and flag == 0) or flag == -1):
                    if ex[ridx] == ")" or ex[ridx] == "]":
                        flag -= 1
                    elif ex[ridx] == "(" or ex[ridx] == "[":
                        flag += 1
                    ridx += 1
                if flag == -1:
                    ridx -= 2
                else:
                    ridx -= 1
            else:
                flag = 0
                while not (lidx == -1
                           or ((ex[lidx] == "+" or ex[lidx] == "-" or ex[lidx] == "*" or ex[lidx] == "/") and flag == 0)
                           or flag == 1):
                    if ex[lidx] == ")" or ex[lidx] == "]":
                        flag -= 1
                    elif ex[lidx] == "(" or ex[lidx] == "[":
                        flag += 1
                    lidx -= 1
                if flag == 1:
                    lidx += 2
                else:
                    lidx += 1

                flag = 0
                while not (ridx == len(ex)
                           or ((ex[ridx] == "+" or ex[ridx] == "-" or ex[ridx] == "*" or ex[ridx] == "/") and flag == 0)
                           or flag == -1):
                    if ex[ridx] == ")" or ex[ridx] == "]":
                        flag -= 1
                    elif ex[ridx] == "(" or ex[ridx] == "[":
                        flag += 1
                    ridx += 1
                if flag == -1:
                    ridx -= 2
                else:
                    ridx -= 1
            if lidx > 0 and ((s == "+" and ex[lidx - 1] == "-") or (s == "*" and ex[lidx - 1] == "/")):
                lidx -= 1
                ex = ex[:lidx] + ex[idx:ridx + 1] + ex[lidx:idx] + ex[ridx + 1:]
            else:
                ex = ex[:lidx] + ex[idx + 1:ridx + 1] + [s] + ex[lidx:idx] + ex[ridx + 1:]
            idx = ridx
        idx += 1
    return ex


def check_bracket(x, english=False):
    if english:
        for idx, s in enumerate(x):
            if s == '[':
                x[idx] = '('
            elif s == '}':
                x[idx] = ')'
        s = x[0]
        idx = 0
        if s == "(":
            flag = 1
            temp_idx = idx + 1
            while flag > 0 and temp_idx < len(x):
                if x[temp_idx] == ")":
                    flag -= 1
                elif x[temp_idx] == "(":
                    flag += 1
                temp_idx += 1
            if temp_idx == len(x):
                x = x[idx + 1:temp_idx - 1]
            elif x[temp_idx] != "*" and x[temp_idx] != "/":
                x = x[idx + 1:temp_idx - 1] + x[temp_idx:]
        while True:
            y = len(x)
            for idx, s in enumerate(x):
                if s == "+" and idx + 1 < len(x) and x[idx + 1] == "(":
                    flag = 1
                    temp_idx = idx + 2
                    while flag > 0 and temp_idx < len(x):
                        if x[temp_idx] == ")":
                            flag -= 1
                        elif x[temp_idx] == "(":
                            flag += 1
                        temp_idx += 1
                    if temp_idx == len(x):
                        x = x[:idx + 1] + x[idx + 2:temp_idx - 1]
                        break
                    elif x[temp_idx] != "*" and x[temp_idx] != "/":
                        x = x[:idx + 1] + x[idx + 2:temp_idx - 1] + x[temp_idx:]
                        break
            if y == len(x):
                break
        return x

    lx = len(x)
    for idx, s in enumerate(x):
        if s == "[":
            flag_b = 0
            flag = False
            temp_idx = idx
            while temp_idx < lx:
                if x[temp_idx] == "]":
                    flag_b += 1
                elif x[temp_idx] == "[":
                    flag_b -= 1
                if x[temp_idx] == "(" or x[temp_idx] == "[":
                    flag = True
                if x[temp_idx] == "]" and flag_b == 0:
                    break
                temp_idx += 1
            if not flag:
                x[idx] = "("
                x[temp_idx] = ")"
                continue
        if s == "(":
            flag_b = 0
            flag = False
            temp_idx = idx
            while temp_idx < lx:
                if x[temp_idx] == ")":
                    flag_b += 1
                elif x[temp_idx] == "(":
                    flag_b -= 1
                if x[temp_idx] == "[":
                    flag = True
                if x[temp_idx] == ")" and flag_b == 0:
                    break
                temp_idx += 1
            if not flag:
                x[idx] = "["
                x[temp_idx] = "]"
    return x


# Multiplication allocation rate
def allocation(ex_copy, rate):
    ex = copy.deepcopy(ex_copy)
    idx = 1
    lex = len(ex)
    while idx < len(ex):
        if (ex[idx] == "/" or ex[idx] == "*") and (ex[idx - 1] == "]" or ex[idx - 1] == ")"):
            ridx = idx + 1
            r_allo = []
            r_last = []
            flag = 0
            flag_mmd = False
            while ridx < lex:
                if ex[ridx] == "(" or ex[ridx] == "[":
                    flag += 1
                elif ex[ridx] == ")" or ex[ridx] == "]":
                    flag -= 1
                if flag == 0:
                    if ex[ridx] == "+" or ex[ridx] == "-":
                        r_last = ex[ridx:]
                        r_allo = ex[idx + 1: ridx]
                        break
                    elif ex[ridx] == "*" or ex[ridx] == "/":
                        flag_mmd = True
                        r_last = [")"] + ex[ridx:]
                        r_allo = ex[idx + 1: ridx]
                        break
                elif flag == -1:
                    r_last = ex[ridx:]
                    r_allo = ex[idx + 1: ridx]
                    break
                ridx += 1
            if len(r_allo) == 0:
                r_allo = ex[idx + 1:]
            flag = 0
            lidx = idx - 1
            flag_al = False
            flag_md = False
            while lidx > 0:
                if ex[lidx] == "(" or ex[lidx] == "[":
                    flag -= 1
                elif ex[lidx] == ")" or ex[lidx] == "]":
                    flag += 1
                if flag == 1:
                    if ex[lidx] == "+" or ex[lidx] == "-":
                        flag_al = True
                if flag == 0:
                    break
                lidx -= 1
            if lidx != 0 and ex[lidx - 1] == "/":
                flag_al = False
            if not flag_al:
                idx += 1
                continue
            elif random.random() < rate:
                temp_idx = lidx + 1
                temp_res = ex[:lidx]
                if flag_mmd:
                    temp_res += ["("]
                if lidx - 1 > 0:
                    if ex[lidx - 1] == "-" or ex[lidx - 1] == "*" or ex[lidx - 1] == "/":
                        flag_md = True
                        temp_res += ["("]
                flag = 0
                lidx += 1
                while temp_idx < idx - 1:
                    if ex[temp_idx] == "(" or ex[temp_idx] == "[":
                        flag -= 1
                    elif ex[temp_idx] == ")" or ex[temp_idx] == "]":
                        flag += 1
                    if flag == 0:
                        if ex[temp_idx] == "+" or ex[temp_idx] == "-":
                            temp_res += ex[lidx: temp_idx] + [ex[idx]] + r_allo + [ex[temp_idx]]
                            lidx = temp_idx + 1
                    temp_idx += 1
                temp_res += ex[lidx: temp_idx] + [ex[idx]] + r_allo
                if flag_md:
                    temp_res += [")"]
                temp_res += r_last
                return temp_res
        if ex[idx] == "*" and (ex[idx + 1] == "[" or ex[idx + 1] == "("):
            lidx = idx - 1
            l_allo = []
            temp_res = []
            flag = 0
            flag_md = False  # flag for x or /
            while lidx > 0:
                if ex[lidx] == "(" or ex[lidx] == "[":
                    flag += 1
                elif ex[lidx] == ")" or ex[lidx] == "]":
                    flag -= 1
                if flag == 0:
                    if ex[lidx] == "+":
                        temp_res = ex[:lidx + 1]
                        l_allo = ex[lidx + 1: idx]
                        break
                    elif ex[lidx] == "-":
                        flag_md = True  # flag for -
                        temp_res = ex[:lidx] + ["("]
                        l_allo = ex[lidx + 1: idx]
                        break
                elif flag == 1:
                    temp_res = ex[:lidx + 1]
                    l_allo = ex[lidx + 1: idx]
                    break
                lidx -= 1
            if len(l_allo) == 0:
                l_allo = ex[:idx]
            flag = 0
            ridx = idx + 1
            flag_al = False
            all_res = []
            while ridx < lex:
                if ex[ridx] == "(" or ex[ridx] == "[":
                    flag -= 1
                elif ex[ridx] == ")" or ex[ridx] == "]":
                    flag += 1
                if flag == 1:
                    if ex[ridx] == "+" or ex[ridx] == "-":
                        flag_al = True
                if flag == 0:
                    break
                ridx += 1
            if not flag_al:
                idx += 1
                continue
            elif random.random() < rate:
                temp_idx = idx + 1
                flag = 0
                lidx = temp_idx + 1
                while temp_idx < idx - 1:
                    if ex[temp_idx] == "(" or ex[temp_idx] == "[":
                        flag -= 1
                    elif ex[temp_idx] == ")" or ex[temp_idx] == "]":
                        flag += 1
                    if flag == 1:
                        if ex[temp_idx] == "+" or ex[temp_idx] == "-":
                            all_res += l_allo + [ex[idx]] + ex[lidx: temp_idx] + [ex[temp_idx]]
                            lidx = temp_idx + 1
                    if flag == 0:
                        break
                    temp_idx += 1
                if flag_md:
                    temp_res += all_res + [")"]
                elif ex[temp_idx + 1] == "*" or ex[temp_idx + 1] == "/":
                    temp_res += ["("] + all_res + [")"]
                temp_res += ex[temp_idx + 1:]
                return temp_res
        idx += 1
    return ex


