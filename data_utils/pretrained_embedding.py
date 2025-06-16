
import torch
import pickle as pkl
import numpy as np
from tqdm import tqdm

def generate_embedding_from_glove(args, min_freq = 2, max_vocab_size = 15000):
    data_dir = args.output_data_dir
    embedding_file_name = args.embedding_file_name
    lang_file_name = args.lang_file_name

    glove_file_path = args.glove_embedding

    langs = pkl.load( open("{}/{}".format(data_dir, lang_file_name), "rb" ) )
    input_lang, output_lang = langs


    glove2vec = {}
    words_arr = []
    cnt_find = 0
    with open(glove_file_path, encoding="utf-8") as f:
        for l in tqdm(f):
            line = l.split()
            word = line[0]
            words_arr.append(word)
            vect = np.array(line[1:]).astype(np.float)
            glove2vec[word] = vect

    word2vec = {}
    word_arr = input_lang.idx2symbol
    for w in tqdm(word_arr):
        if w in glove2vec.keys():
            word2vec[w] = glove2vec[w]

    print (len(word2vec))
    out_file = "{}/{}".format(data_dir, embedding_file_name)
    with open(out_file, "wb") as out_data:
        pkl.dump(word2vec, out_data)

def make_pretrained_embedding(embedding_size, opt_file, min_freq = 2, max_vocab_size = 15000):
    # use glove pretrained embedding and vocabulary to generate a embedding matrix
    torch.manual_seed(400)
    embedding_file_name = "{}/{}_{}".format(opt_file.data_dir, opt_file.dataset_lang, opt_file.embedding_file_name)
    lang_file_name = "{}/{}_{}".format(opt_file.data_dir, opt_file.dataset_lang, opt_file.lang_file_name)
    word2vec = pkl.load( open(embedding_file_name, "rb" ) )
    langs = pkl.load( open(lang_file_name, "rb" ) )
    input_lang, output_lang = langs

    num_embeddings, embedding_dim = embedding_size
    weight_matrix = torch.zeros((num_embeddings, embedding_dim), dtype=torch.float)
    cnt_change = 0
    for i in range(num_embeddings):
        word = input_lang.idx2symbol[i]
        if word in word2vec:
            weight_matrix[i] = torch.from_numpy(word2vec[word])
            cnt_change += 1
        else:
            weight_matrix[i] = torch.randn((embedding_dim, ))
    # print (cnt_change)
    return weight_matrix


