import numpy as np
import data_utils.pre_data as pre_data
# num net graph
def get_lower_num_graph(max_len, sentence_length, num_list, id_num_list,contain_zh_flag=True):
    diag_ele = np.zeros(max_len)
    num_list = pre_data.change_num(num_list)
    for i in range(sentence_length):
        diag_ele[i] = 1
    graph = np.diag(diag_ele)
    if not contain_zh_flag:
        return graph
    for i in range(len(id_num_list)):
        for j in range(len(id_num_list)):
            if float(num_list[i]) <= float(num_list[j]):
                graph[id_num_list[i]][id_num_list[j]] = 1
            else:
                graph[id_num_list[j]][id_num_list[i]] = 1
    return graph

def get_greater_num_graph(max_len, sentence_length, num_list, id_num_list,contain_zh_flag=True):
    diag_ele = np.zeros(max_len)
    num_list = pre_data.change_num(num_list)
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

# attribute between graph
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

# quantity between graph
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

# quantity cell graph
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

def get_single_batch_graph(input_batch, input_length,group,num_value,num_pos):
    batch_graph = []
    max_len = max(input_length)
    for i in range(len(input_length)):
        graph = get_single_example_graph(input_batch[i], input_length[i],max_len, group[i],num_value[i],num_pos[i])
        '''
        graph_newc = get_quantity_cell_graph(max_len, id_num_list, sentence_length, quantity_cell_list)
        graph_greater = get_greater_num_graph(max_len, sentence_length, num_list, id_num_list)
        graph_lower = get_lower_num_graph(max_len, sentence_length, num_list, id_num_list)
        graph_quanbet = get_quantity_between_graph(max_len, id_num_list, sentence_length, quantity_cell_list)
        graph_attbet = get_attribute_between_graph(input_batch_t, max_len, id_num_list, sentence_length, quantity_cell_list)
        #graph_newc1 = get_quantity_graph1(input_batch_t, max_len, id_num_list, sentence_length, quantity_cell_list)
        graph_total = [graph_newc.tolist(),graph_greater.tolist(),graph_lower.tolist(),graph_quanbet.tolist(),graph_attbet.tolist()]
        batch_graph.append(graph_total)
        '''
        batch_graph.append(graph)
    return batch_graph

def get_single_example_graph(input_batch, input_length, max_input_length, group,num_value,num_pos):
    max_len = max_input_length
    sentence_length = input_length
    quantity_cell_list = group
    num_list = num_value
    id_num_list = num_pos
    graph_newc = get_quantity_cell_graph(max_len, id_num_list, sentence_length, quantity_cell_list)
    graph_quanbet = get_quantity_between_graph(max_len, id_num_list, sentence_length, quantity_cell_list)
    graph_attbet = get_attribute_between_graph(input_batch, max_len, id_num_list, sentence_length, quantity_cell_list)
    graph_greater = get_greater_num_graph(max_len, sentence_length, num_list, id_num_list)
    graph_lower = get_greater_num_graph(max_len, sentence_length, num_list, id_num_list)
    #graph_newc1 = get_quantity_graph1(input_batch, max_len, id_num_list, sentence_length, quantity_cell_list)
    graph = [graph_newc.tolist(),graph_greater.tolist(),graph_lower.tolist(),graph_quanbet.tolist(),graph_attbet.tolist()]
    graph = np.array(graph)
    return graph


