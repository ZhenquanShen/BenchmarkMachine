import torch
import torch.nn as nn
from transformers import AutoModel
from data_utils.pretrained_embedding import make_pretrained_embedding
from torch.nn.parameter import Parameter
import math
from torch.nn.modules.module import Module
import torch.nn.functional as F
import copy
import sys
import json
from solvingengine.stategraph.transit.encoder_decoder import nn_modules

class TreeNode:  # the class save the tree node
    def __init__(self, embedding, left_flag=False):
        self.embedding = embedding
        self.left_flag = left_flag

class Score(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Score, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, num_embeddings, num_mask=None):
        max_len = num_embeddings.size(1)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[1] = max_len
        hidden = hidden.repeat(*repeat_dims)  # B x O x H
        # For each position of encoder outputs
        this_batch_size = num_embeddings.size(0)
        energy_in = torch.cat((hidden, num_embeddings), 2).view(-1, self.input_size + self.hidden_size)
        score = self.score(torch.tanh(self.attn(energy_in)))  # (B x O) x 1
        score = score.squeeze(1)
        score = score.view(this_batch_size, -1)  # B x O
        if num_mask is not None:
            score = score.masked_fill_(num_mask.bool(), -1e12)
        return score


class TreeAttn(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeAttn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        max_len = encoder_outputs.size(0)

        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        this_batch_size = encoder_outputs.size(1)

        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, self.input_size + self.hidden_size)

        score_feature = torch.tanh(self.attn(energy_in))
        attn_energies = self.score(score_feature)  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask.bool(), -1e12)
        attn_energies = nn.functional.softmax(attn_energies, dim=1)  # B x S

        return attn_energies.unsqueeze(1)


class EncoderSeq(nn.Module):
    def __init__(self, opt_model_local, opt_file, input_size):
        super(EncoderSeq, self).__init__()
        self.hidden_size = opt_model_local.hidden_size
        # self.bert_rnn = BertModel.from_pretrained("hfl/bert-base-chinese")
        self.bert_rnn = AutoModel.from_pretrained("D:/code/benchmarkmachine/solvingengine/stategraph/transit/pretrained_model/MWP-BERT")
        # self.bert_rnn = AutoModel.from_pretrained("D:/code/benchmarkmachine/solvingengine/stategraph/transit/pretrained_model/wang-gts")
        # self.bert_rnn = BertConfig()
        self.input_size = input_size
        self.embedding_size = opt_model_local.embedding_size
        
        self.n_layers = opt_model_local.num_layers
        self.dropout = opt_model_local.dropout
        self.use_glove_embedding_vector = True
        self.embedding = nn.Embedding(input_size, self.embedding_size, padding_idx=0)
        # if self.use_glove_embedding_vector:
        #     self.embedding.weight.data = make_pretrained_embedding(self.embedding.weight.size(), opt_file)
            #self.embedding.weight.requires_grad = False
        self.em_dropout = nn.Dropout(self.dropout)

        self.gru_pade = nn.GRU(self.embedding_size, self.hidden_size, self.n_layers, dropout=self.dropout, bidirectional=True)
        self.gcn = Graph_Module(self.hidden_size, self.hidden_size, self.hidden_size)
        
    
    def forward(self, bert_encoding):
        input_ids = bert_encoding['input_ids'].long().cuda()
        attention_mask = bert_encoding['attention_mask'].long().cuda()
      
        bert_output = self.bert_rnn(input_ids, attention_mask=attention_mask)[0].transpose(0,1) # S x B x E
        problem_output = bert_output.mean(0)
        
        return bert_output, problem_output #seq_len, batch_size, H(768)


    def graph2tree_wang(self, input_seqs, input_length, batch_graph, hidden=None):
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_length)
        pade_hidden = hidden
        pade_outputs, pade_hidden = self.gru_pade(packed, pade_hidden)
        pade_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs)

        problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        pade_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]  # S x B x H
        _, pade_outputs = self.gcn(pade_outputs, batch_graph)
        pade_outputs = pade_outputs.transpose(0, 1)
        # hidden vector: [max_input_length,batch_size,hidden_size]
        # embedding vector: [batch_size, hidden_size]
        return pade_outputs, problem_output

class Prediction(nn.Module):
    # a seq2tree decoder with Problem aware dynamic encoding

    def __init__(self, opt_model_local, op_nums, input_size, dropout=0.5):
        super(Prediction, self).__init__()

        # Keep for reference
        self.hidden_size = opt_model_local.hidden_size
        self.input_size = input_size
        self.op_nums = op_nums

        # Define layers
        self.dropout = nn.Dropout(dropout)

        self.embedding_weight = nn.Parameter(torch.randn(1, input_size, self.hidden_size))

        # for Computational symbols and Generated numbers
        self.concat_l = nn.Linear(self.hidden_size, self.hidden_size)
        self.concat_r = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.concat_lg = nn.Linear(self.hidden_size, self.hidden_size)
        self.concat_rg = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.ops = nn.Linear(self.hidden_size * 2, op_nums)

        self.attn = TreeAttn(self.hidden_size, self.hidden_size)
        self.score = Score(self.hidden_size * 2, self.hidden_size)

    def forward(self, node_stacks, left_childs, encoder_outputs, num_pades, padding_hidden, seq_mask, mask_nums):
        current_embeddings = []

        for st in node_stacks:
            if len(st) == 0:
                current_embeddings.append(padding_hidden)
            else:
                current_node = st[-1]
                current_embeddings.append(current_node.embedding)

        current_node_temp = []
        for l, c in zip(left_childs, current_embeddings):
            if l is None:
                c = self.dropout(c)
                g = torch.tanh(self.concat_l(c))
                t = torch.sigmoid(self.concat_lg(c))
                current_node_temp.append(g * t)
            else:
                ld = self.dropout(l)
                c = self.dropout(c)
                g = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))
                t = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))
                current_node_temp.append(g * t)

        current_node = torch.stack(current_node_temp)

        current_embeddings = self.dropout(current_node)

        current_attn = self.attn(current_embeddings.transpose(0, 1), encoder_outputs, seq_mask) #seq_len, batch_size, 768
        current_context = current_attn.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N

        # the information to get the current quantity
        batch_size = current_embeddings.size(0)
        # predict the output (this node corresponding to output(number or operator)) with PADE

        repeat_dims = [1] * self.embedding_weight.dim()
        repeat_dims[0] = batch_size
        embedding_weight = self.embedding_weight.repeat(*repeat_dims)  # B x input_size x N
        embedding_weight = torch.cat((embedding_weight, num_pades), dim=1)  # B x O x N

        leaf_input = torch.cat((current_node, current_context), 2)
        leaf_input = leaf_input.squeeze(1)
        leaf_input = self.dropout(leaf_input)

        # p_leaf = nn.functional.softmax(self.is_leaf(leaf_input), 1)
        # max pooling the embedding_weight
        embedding_weight_ = self.dropout(embedding_weight)
        num_score = self.score(leaf_input.unsqueeze(1), embedding_weight_, mask_nums)

        # num_score = nn.functional.softmax(num_score, 1)

        op = self.ops(leaf_input)

        # return p_leaf, num_score, op, current_embeddings, current_attn

        return num_score, op, current_node, current_context, embedding_weight


class GenerateNode(nn.Module):
    def __init__(self, opt_model_local, op_nums, dropout=0.5):
        super(GenerateNode, self).__init__()

        self.embedding_size = opt_model_local.embedding_size
        self.hidden_size = opt_model_local.hidden_size

        self.embeddings = nn.Embedding(op_nums, self.embedding_size)
        self.em_dropout = nn.Dropout(dropout)
        self.generate_l = nn.Linear(self.hidden_size * 2 + self.embedding_size, self.hidden_size)
        self.generate_r = nn.Linear(self.hidden_size * 2 + self.embedding_size, self.hidden_size)
        self.generate_lg = nn.Linear(self.hidden_size * 2 + self.embedding_size, self.hidden_size)
        self.generate_rg = nn.Linear(self.hidden_size * 2 + self.embedding_size, self.hidden_size)

    def forward(self, node_embedding, node_label, current_context):
        node_label_ = self.embeddings(node_label)
        node_label = self.em_dropout(node_label_)
        node_embedding = node_embedding.squeeze(1)
        current_context = current_context.squeeze(1)
        node_embedding = self.em_dropout(node_embedding)
        current_context = self.em_dropout(current_context)

        l_child = torch.tanh(self.generate_l(torch.cat((node_embedding, current_context, node_label), 1)))
        l_child_g = torch.sigmoid(self.generate_lg(torch.cat((node_embedding, current_context, node_label), 1)))
        r_child = torch.tanh(self.generate_r(torch.cat((node_embedding, current_context, node_label), 1)))
        r_child_g = torch.sigmoid(self.generate_rg(torch.cat((node_embedding, current_context, node_label), 1)))
        l_child = l_child * l_child_g
        r_child = r_child * r_child_g
        return l_child, r_child, node_label_


class Merge(nn.Module):
    def __init__(self, opt_model_local):
        super(Merge, self).__init__()

        self.embedding_size = opt_model_local.embedding_size
        self.hidden_size = opt_model_local.hidden_size

        self.em_dropout = nn.Dropout(opt_model_local.dropout)
        self.merge = nn.Linear(self.hidden_size * 2 + self.embedding_size, self.hidden_size)
        self.merge_g = nn.Linear(self.hidden_size * 2 + self.embedding_size, self.hidden_size)

    def forward(self, node_embedding, sub_tree_1, sub_tree_2):
        sub_tree_1 = self.em_dropout(sub_tree_1)
        sub_tree_2 = self.em_dropout(sub_tree_2)
        node_embedding = self.em_dropout(node_embedding)

        sub_tree = torch.tanh(self.merge(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree_g = torch.sigmoid(self.merge_g(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree = sub_tree * sub_tree_g
        return sub_tree

class wangEncoderSeq(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5):
        super(wangEncoderSeq, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        self.gru_pade = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        self.gcn = Graph_Module(hidden_size, hidden_size, hidden_size)

    def forward(self, input_seqs, input_lengths, batch_graph, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        pade_hidden = hidden
        pade_outputs, pade_hidden = self.gru_pade(packed, pade_hidden)
        pade_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs)

        problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        pade_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]  # S x B x H
        _, pade_outputs = self.gcn(pade_outputs, batch_graph)
        pade_outputs = pade_outputs.transpose(0, 1)
        return pade_outputs, problem_output

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff,d_out, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Graph_Module(nn.Module):
    def __init__(self, indim, hiddim, outdim, dropout=0.3):
        super(Graph_Module, self).__init__()
        '''
        ## Variables:
        - indim: dimensionality of input node features
        - hiddim: dimensionality of the joint hidden embedding
        - outdim: dimensionality of the output node features
        - combined_feature_dim: dimensionality of the joint hidden embedding for graph
        - K: number of graph nodes/objects on the image
        '''
        self.in_dim = indim
        #self.combined_dim = outdim
        
        #self.edge_layer_1 = nn.Linear(indim, outdim)
        #self.edge_layer_2 = nn.Linear(outdim, outdim)
        
        #self.dropout = nn.Dropout(p=dropout)
        #self.edge_layer_1 = nn.utils.weight_norm(self.edge_layer_1)
        #self.edge_layer_2 = nn.utils.weight_norm(self.edge_layer_2)
        self.h = 4
        self.d_k = outdim//self.h
        
        #layer = GCN(indim, hiddim, self.d_k, dropout)
        self.graph = clones(GCN(indim, hiddim, self.d_k, dropout), 4)
        
        #self.Graph_0 = GCN(indim, hiddim, outdim//4, dropout)
        #self.Graph_1 = GCN(indim, hiddim, outdim//4, dropout)
        #self.Graph_2 = GCN(indim, hiddim, outdim//4, dropout)
        #self.Graph_3 = GCN(indim, hiddim, outdim//4, dropout)
        
        self.feed_foward = PositionwiseFeedForward(indim, hiddim, outdim, dropout)
        self.norm = LayerNorm(outdim)

    def get_adj(self, graph_nodes):
        '''
        ## Inputs:
        - graph_nodes (batch_size, K, in_feat_dim): input features
        ## Returns:
        - adjacency matrix (batch_size, K, K)
        '''
        self.K = graph_nodes.size(1)
        graph_nodes = graph_nodes.contiguous().view(-1, self.in_dim)
        
        # layer 1
        h = self.edge_layer_1(graph_nodes)
        h = F.relu(h)
        
        # layer 2
        h = self.edge_layer_2(h)
        h = F.relu(h)

        # outer product
        h = h.view(-1, self.K, self.combined_dim)
        adjacency_matrix = torch.matmul(h, h.transpose(1, 2))
        
        adjacency_matrix = self.b_normal(adjacency_matrix)

        return adjacency_matrix
    
    def normalize(self, A, symmetric=True):
        '''
        ## Inputs:
        - adjacency matrix (K, K) : A
        ## Returns:
        - adjacency matrix (K, K) 
        '''
        A = A + torch.eye(A.size(0)).cuda().float()
        d = A.sum(1)
        if symmetric:
            # D = D^{-1/2}
            D = torch.diag(torch.pow(d, -0.5))
            return D.mm(A).mm(D)
        else :
            D = torch.diag(torch.pow(d,-1))
            return D.mm(A)
       
    def b_normal(self, adj):
        batch = adj.size(0)
        for i in range(batch):
            adj[i] = self.normalize(adj[i])
        return adj

    def forward(self, graph_nodes, graph):
        '''
        ## Inputs:
        - graph_nodes (batch_size, K, in_feat_dim): input features
        ## Returns:
        - graph_encode_features (batch_size, K, out_feat_dim)
        '''
        nbatches = graph_nodes.size(0)
        mbatches = graph.size(0)
        if nbatches != mbatches:
            graph_nodes = graph_nodes.transpose(0, 1)
        # adj (batch_size, K, K): adjacency matrix
        if not bool(graph.numel()):
            adj = self.get_adj(graph_nodes)
            #adj = adj.unsqueeze(1)
            #adj = torch.cat((adj,adj,adj),1)
            adj_list = [adj,adj,adj,adj]
        else:
            adj = graph.float()
            adj_list = [adj[:,1,:],adj[:,1,:],adj[:,0,:],adj[:,0,:]]
        #print(adj_list)

        
        g_feature = \
            tuple([l(graph_nodes,x) for l, x in zip(self.graph,adj_list)])
        #g_feature_0 = self.Graph_0(graph_nodes,adj[0])
        #g_feature_1 = self.Graph_1(graph_nodes,adj[1])
        #g_feature_2 = self.Graph_2(graph_nodes,adj[2])
        #g_feature_3 = self.Graph_3(graph_nodes,adj[3])
        #print('g_feature')
        #print(type(g_feature))
        
        
        g_feature = self.norm(torch.cat(g_feature,2)) + graph_nodes
        #print('g_feature')
        #print(g_feature.shape)
        
        graph_encode_features = self.feed_foward(g_feature) + g_feature
        
        return adj, graph_encode_features

class GCN(nn.Module):
    def __init__(self, in_feat_dim, nhid, out_feat_dim, dropout):
        super(GCN, self).__init__()
        '''
        ## Inputs:
        - graph_nodes (batch_size, K, in_feat_dim): input features
        - adjacency matrix (batch_size, K, K)
        ## Returns:
        - gcn_enhance_feature (batch_size, K, out_feat_dim)
        '''
        self.gc1 = GraphConvolution(in_feat_dim, nhid)
        self.gc2 = GraphConvolution(nhid, out_feat_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x



class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        #print(input.shape)
        #print(self.weight.shape)
        support = torch.matmul(input, self.weight)
        #print(adj.shape)
        #print(support.shape)
        output = torch.matmul(adj, support)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    
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
        problem_output = graph_embedding.mean(0)
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


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(self,
                 vocab_size_or_config_json_file=21128,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 pre_trained='',
                 training=''):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                                                               and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.pre_trained = pre_trained
            self.training = training
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())