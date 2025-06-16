import os
import time
import random
import data_utils
import nn_modules
import numpy as np
import pickle as pkl
from tree import Tree
from data_utils.pretrained_embedding import make_pretrained_embedding

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch import optim

class GraphEncoder(nn.Module):
    def __init__(self, opt_model_local, opt_file, input_size, using_gpu):
        super(GraphEncoder, self).__init__()
        self.using_gpu = using_gpu
        if opt_model_local.dropoutagg > 0:
            self.dropout = nn.Dropout(opt_model_local.dropoutagg)

        self.graph_encode_direction = opt_model_local.graph_encode_direction
        self.sample_size_per_layer = opt_model_local.sample_size_per_layer
        self.sample_layer_size = opt_model_local.sample_layer_size
        self.hidden_size = opt_model_local.rnn_size
        self.dropout_en_in = opt_model_local.dropout_en_in
        self.dropout_en_out = opt_model_local.dropout_en_out
        self.dropoutagg = opt_model_local.dropoutagg

        self.word_embedding_size = opt_model_local.embedding_size
        self.embedding = nn.Embedding(input_size, self.word_embedding_size, padding_idx=0)
        self.embedding.weight.data = make_pretrained_embedding(self.embedding.weight.size(), opt_file)
        #self.embedding.weight.requires_grad = False
        if self.dropout_en_in > 0:
            self.input_dropout = nn.Dropout(self.dropout_en_in)
        self.fw_aggregators = []
        self.bw_aggregators = []

        self.fw_aggregator_0 = nn_modules.MeanAggregator(
            2*self.hidden_size, self.hidden_size, concat=True)
        self.fw_aggregator_1 = nn_modules.MeanAggregator(
            2*self.hidden_size, self.hidden_size, concat=True)
        self.fw_aggregator_2 = nn_modules.MeanAggregator(
            2*self.hidden_size, self.hidden_size, concat=True)
        self.fw_aggregator_3 = nn_modules.MeanAggregator(
            2*self.hidden_size, self.hidden_size, concat=True)
        self.fw_aggregator_4 = nn_modules.MeanAggregator(
            2*self.hidden_size, self.hidden_size, concat=True)
        self.fw_aggregator_5 = nn_modules.MeanAggregator(
            2*self.hidden_size, self.hidden_size, concat=True)
        self.fw_aggregator_6 = nn_modules.MeanAggregator(
            2*self.hidden_size, self.hidden_size, concat=True)

        self.bw_aggregator_0 = nn_modules.MeanAggregator(
            2*self.hidden_size, self.hidden_size, concat=True)
        self.bw_aggregator_1 = nn_modules.MeanAggregator(
            2*self.hidden_size, self.hidden_size, concat=True)
        self.bw_aggregator_2 = nn_modules.MeanAggregator(
            2*self.hidden_size, self.hidden_size, concat=True)
        self.bw_aggregator_3 = nn_modules.MeanAggregator(
            2*self.hidden_size, self.hidden_size, concat=True)
        self.bw_aggregator_4 = nn_modules.MeanAggregator(
            2*self.hidden_size, self.hidden_size, concat=True)
        self.bw_aggregator_5 = nn_modules.MeanAggregator(
            2*self.hidden_size, self.hidden_size, concat=True)
        self.bw_aggregator_6 = nn_modules.MeanAggregator(
            2*self.hidden_size, self.hidden_size, concat=True)
        self.fw_aggregators = [self.fw_aggregator_0, self.fw_aggregator_1, self.fw_aggregator_2,
                               self.fw_aggregator_3, self.fw_aggregator_4, self.fw_aggregator_5, self.fw_aggregator_6]
        self.bw_aggregators = [self.bw_aggregator_0, self.bw_aggregator_1, self.bw_aggregator_2,
                               self.bw_aggregator_3, self.bw_aggregator_4, self.bw_aggregator_5, self.bw_aggregator_6]

        self.Linear_hidden = nn.Linear(
            2 * self.hidden_size, self.hidden_size)


        _hidden_size = int(self.hidden_size/2)
        self.embedding_bilstm = nn.LSTM(input_size=self.word_embedding_size, hidden_size=int(self.hidden_size/2),
                                        bidirectional=True, bias = True, batch_first = True,
                                        dropout= self.dropout_en_out, num_layers=2)
        #self.embedding_bilstm = nn.GRU(input_size=self.word_embedding_size, hidden_size=int(self.hidden_size/2), num_layers = 1, dropout=self.opt.dropout_en_out, bidirectional=True)
        self.padding_vector = torch.randn(1,self.hidden_size, dtype = torch.float, requires_grad=True)

    def forward(self, graph_batch):
        fw_adj_info, bw_adj_info, feature_info, batch_nodes = graph_batch

        # print self.hidden_size

        if self.using_gpu:
            fw_adj_info = fw_adj_info.long().cuda()
            bw_adj_info = bw_adj_info.long().cuda()
            feature_info = feature_info.cuda()
            batch_nodes = batch_nodes.cuda()

        feature_by_sentence = feature_info[:-1,:].view(batch_nodes.size()[0], -1)
        feature_sentence_vector = self.embedding(feature_by_sentence)
        if self.dropout_en_in > 0:
            feature_sentence_vector = self.input_dropout(feature_sentence_vector)
        output_vector, (ht,_) = self.embedding_bilstm(feature_sentence_vector)
        #ht = None
        #output_vector, ht = self.embedding_bilstm(feature_sentence_vector, ht)
        seq_embedding = torch.max(output_vector, 1)[0]
        feature_vector = output_vector.contiguous().view(-1, self.hidden_size)
        if self.using_gpu:
            feature_embedded = torch.cat([feature_vector, self.padding_vector.cuda()], 0)
        else:
            feature_embedded = torch.cat([feature_vector, self.padding_vector], 0)

        batch_size = feature_embedded.size()[0]
        node_repres = feature_embedded.view(batch_size, -1)

        fw_sampler = nn_modules.UniformNeighborSampler(fw_adj_info)
        bw_sampler = nn_modules.UniformNeighborSampler(bw_adj_info)
        nodes = batch_nodes.view(-1, )

        fw_hidden = F.embedding(nodes, node_repres)
        bw_hidden = F.embedding(nodes, node_repres)

        fw_sampled_neighbors = fw_sampler((nodes, self.sample_size_per_layer))
        bw_sampled_neighbors = bw_sampler((nodes, self.sample_size_per_layer))

        fw_sampled_neighbors_len = torch.tensor(0)
        bw_sampled_neighbors_len = torch.tensor(0)

        # begin sampling
        for layer in range(self.sample_layer_size):
            if layer == 0:
                dim_mul = 1
            else:
                dim_mul = 1
            if self.using_gpu and layer <= 6:
                self.fw_aggregators[layer] = self.fw_aggregators[layer].cuda()
            if layer == 0:
                neigh_vec_hidden = F.embedding(
                    fw_sampled_neighbors, node_repres)
                tmp_sum = torch.sum(F.relu(neigh_vec_hidden), 2)
                tmp_mask = torch.sign(tmp_sum)
                fw_sampled_neighbors_len = torch.sum(tmp_mask, 1)
            else:
                if self.using_gpu:
                    neigh_vec_hidden = F.embedding(fw_sampled_neighbors, torch.cat([fw_hidden, torch.zeros(
                        [1, dim_mul * self.hidden_size]).cuda()], 0))
                else:
                    neigh_vec_hidden = F.embedding(fw_sampled_neighbors, torch.cat([fw_hidden, torch.zeros(
                        [1, dim_mul * self.hidden_size])], 0))

            if layer > 6:
                    fw_hidden = self.fw_aggregators[6](
                        (fw_hidden, neigh_vec_hidden, fw_sampled_neighbors_len))
            else:
                    fw_hidden = self.fw_aggregators[layer](
                        (fw_hidden, neigh_vec_hidden, fw_sampled_neighbors_len))

            if self.graph_encode_direction == "bi":
                if self.using_gpu and layer <= 6:
                    self.bw_aggregators[layer] = self.bw_aggregators[layer].cuda(
                    )

                if layer == 0:
                    neigh_vec_hidden = F.embedding(
                        bw_sampled_neighbors, node_repres)
                    tmp_sum = torch.sum(F.relu(neigh_vec_hidden), 2)
                    tmp_mask = torch.sign(tmp_sum)
                    bw_sampled_neighbors_len = torch.sum(tmp_mask, 1)
                else:
                    if self.using_gpu:
                        neigh_vec_hidden = F.embedding(bw_sampled_neighbors, torch.cat([bw_hidden, torch.zeros(
                            [1, dim_mul * self.hidden_size]).cuda()], 0))
                    else:
                        neigh_vec_hidden = F.embedding(bw_sampled_neighbors, torch.cat([bw_hidden, torch.zeros(
                            [1, dim_mul * self.hidden_size])], 0))
                if self.dropoutagg > 0:
                    bw_hidden = self.dropout(bw_hidden)
                    neigh_vec_hidden = self.dropout(neigh_vec_hidden)

                if layer > 6:
                    bw_hidden = self.bw_aggregators[6](
                        (bw_hidden, neigh_vec_hidden, bw_sampled_neighbors_len))
                else:
                    bw_hidden = self.bw_aggregators[layer](
                        (bw_hidden, neigh_vec_hidden, bw_sampled_neighbors_len))
        fw_hidden = fw_hidden.view(-1, batch_nodes.size()
                                   [1], self.hidden_size)

        if self.graph_encode_direction == "bi":
            bw_hidden = bw_hidden.view(-1, batch_nodes.size()
                                       [1], self.hidden_size)
            hidden = torch.cat([fw_hidden, bw_hidden], 2)
        else:
            hidden = fw_hidden

        pooled = torch.max(hidden, 1)[0]
        graph_embedding = pooled.view(-1, self.hidden_size)
        # hidden:final hidden state of  biGraphSAGE
        # graph_embedding: max values of each row in hidden
        # output_vector: sentence features after bilstm
        # hidden vector: [batch_size,max_input_length,hidden_size]
        # embedding vector: [batch_size, hidden_size]
        return graph_embedding, hidden, seq_embedding, output_vector


class Dec_LSTM(nn.Module):
    def __init__(self, rnn_size,dropout_de_out):
        # opt.dropout_de_out
        # opt.rnn_size
        super(Dec_LSTM, self).__init__()
        self.dropout_de_out = dropout_de_out
        self.rnn_size = rnn_size
        self.word_embedding_size = 300
        self.i2h = nn.Linear(self.word_embedding_size+2*self.rnn_size, 4*self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 4*self.rnn_size)

        if self.dropout_de_out > 0:
            self.dropout = nn.Dropout(self.dropout_de_out)

    def forward(self, x, prev_c, prev_h, parent_h, sibling_state):
        input_cat = torch.cat((x, parent_h, sibling_state),1)
        gates = self.i2h(input_cat) + self.h2h(prev_h)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4,1)
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)
        if self.dropout_de_out > 0:
            cellgate = self.dropout(cellgate)
        cy = (forgetgate * prev_c) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)
        return cy, hy

class DecoderRNN(nn.Module):
    def __init__(self, opt_model_local, input_size):
        # opt.rnn_size
        # opt.dropout_de_in
        # opt.dropout_de_out
        # opt.dropout_for_predict
        super(DecoderRNN, self).__init__()
        self.dropout_de_in = opt_model_local.dropout_de_in
        self.dropout_de_out = opt_model_local.dropout_de_out
        self.dropout_for_predict = opt_model_local.dropout_for_predict
        self.hidden_size = opt_model_local.rnn_size
        self.word_embedding_size = opt_model_local.embedding_size
        self.embedding = nn.Embedding(input_size, self.word_embedding_size, padding_idx=0)

        self.lstm = Dec_LSTM(self.hidden_size, self.dropout_de_out)
        if self.dropout_de_in > 0:
            self.dropout = nn.Dropout(self.dropout_de_in)

    def forward(self, input_src, prev_c, prev_h, parent_h, sibling_state):

        src_emb = self.embedding(input_src)
        if self.dropout_de_in > 0:
            src_emb = self.dropout(src_emb)
        prev_cy, prev_hy = self.lstm(src_emb, prev_c, prev_h, parent_h, sibling_state)
        return prev_cy, prev_hy

class AttnUnit(nn.Module):
    def __init__(self, opt, output_size):
        super(AttnUnit, self).__init__()
        # opt.rnn_size
        # opt.dropout_for_predict
        self.dropout_for_predict = opt.dropout_for_predict
        self.hidden_size = opt.rnn_size
        self.separate_attention = True
        if self.separate_attention:
            self.linear_att = nn.Linear(3*self.hidden_size, self.hidden_size)
        else:
            self.linear_att = nn.Linear(2*self.hidden_size, self.hidden_size)

        self.linear_out = nn.Linear(self.hidden_size, output_size)
        if self.dropout_for_predict > 0:
            self.dropout = nn.Dropout(self.dropout_for_predict)

        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, enc_s_top, dec_s_top, enc_2):
        dot = torch.bmm(enc_s_top, dec_s_top.unsqueeze(2))
        attention = self.softmax(dot.squeeze(2)).unsqueeze(2)
        enc_attention = torch.bmm(enc_s_top.permute(0,2,1), attention)

        if self.separate_attention:
            dot_2 = torch.bmm(enc_2, dec_s_top.unsqueeze(2))
            attention_2 = self.softmax(dot_2.squeeze(2)).unsqueeze(2)
            enc_attention_2 = torch.bmm(enc_2.permute(0,2,1), attention_2)

        if self.separate_attention:
            hid = F.tanh(self.linear_att(torch.cat((enc_attention.squeeze(2), enc_attention_2.squeeze(2),dec_s_top), 1)))
        else:
            hid = F.tanh(self.linear_att(torch.cat((enc_attention.squeeze(2),dec_s_top), 1)))
        h2y_in = hid
        if self.dropout_for_predict > 0:
            h2y_in = self.dropout(h2y_in)
        h2y = self.linear_out(h2y_in)
        pred = self.logsoftmax(h2y)

        return pred


