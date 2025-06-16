
class File_Config:
    data_dir ='D:/code/benchmarkmachine/solvingengine/stategraph/transit/GraphConstruction'
    dataset_name = ''
    dataset_lang = 'cn'
    embedding_file_name = 'pretrain_embedding.pkl'
    lang_file_name = 'lang_map.pkl'
    checkpoint_dir = 'checkpoint_dir'
    model_dir = 'trained_model_dir'

opt_file = File_Config()


class Local_Model_Wang:
    encoder_name = 'graph2tree_wang'
    decoder_name = 'graph2tree_wang'
    random_seed = 123
    embedding_size = 128
    hidden_size = 512
    # rnn_size = 300
    dropout=0.5
    num_layers = 2
    # dropoutagg = 0
    learning_rate = 1e-3
    weight_decay = 1e-5
    beam_size = 5
    # graph_encode_direction = 'uni'
    # dropout_en_in = 0.1
    # dropout_en_out = 0.3
    # dropout_de_in = 0.1
    # dropout_de_out = 0.3
    # dropout_for_predict = 0.1
    # concat = True
    # teacher_force_ratio = 1.0
    # init_weight = 0.08
    # sample_size_per_layer = 10
    # sample_layer_size = 3

class Bert:
    embedding_size = 128 
    hidden_size = 768
    num_layers = 2
    dropout=0.5


class Local_Model_Li:
    encoder_name = 'graph2tree_li'
    decoder_name = 'graph2tree_li'
    random_seed = 123
    embedding_size = 300
    rnn_size = 512
    num_layers = 2
    dropout_en_in = 0.1
    dropout_en_out = 0.3
    dropout_de_in = 0.1
    dropout_de_out = 0.3
    dropout_for_predict = 0.1
    dropoutagg = 0
    concat = True
    learning_rate = 1e-3
    graph_encode_direction = 'uni'
    teacher_force_ratio = 1.0
    init_weight = 0.08
    sample_size_per_layer = 10
    sample_layer_size = 3

class TreeDecoder:
    embedding_size = 128 
    hidden_size = 512
    dropout=0.5
    beam_size=5
    MAX_OUTPUT_LENGTH = 45

class RnnDecoder:
    encoder_name = 'graph2tree_li'
    decoder_name = 'graph2tree_li'
    random_seed = 123
    embedding_size = 300
    rnn_size = 512
    num_layers = 2
    dropout_en_in = 0.1
    dropout_en_out = 0.3
    dropout_de_in = 0.1
    dropout_de_out = 0.3
    dropout_for_predict = 0.1
    dropoutagg = 0
    concat = True
    learning_rate = 1e-3
    graph_encode_direction = 'uni'
    teacher_force_ratio = 1.0
    init_weight = 0.08
    sample_size_per_layer = 10
    sample_layer_size = 3
    dec_seq_length_max = 47