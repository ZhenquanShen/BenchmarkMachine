import pickle as pkl
import json
import data_utils.pre_data as pre_data

def read_json(path):
    with open(path,'r',encoding="utf-8") as f:
        file = json.load(f)
    return file

def load_raw_data(filename):  # load the json data to list(dict()) for MATH 23K
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    js = ""
    data = []
    for i, s in enumerate(f):
        js += s
        i += 1
        if i % 7 == 0:  # every 7 line is a json
            data_d = json.loads(js)
            if "千米/小时" in data_d["equation"]:
                data_d["equation"] = data_d["equation"][:-5]
            data.append(data_d)
            js = ""

    return data
'''
train_path:file name of training set, needed, train_set = data_pair if data pair id in train set
test_path:file name of test set, needed, test_set = data_pair if data pair id in test set
valid_path:file name of valid set, if None, valid_set = data_pairs - train_set - test_set, otherwise valid_set = data_pair if data pair id in valid set
data_pairs:full data set
'''
def get_train_test_fold(train_path, test_path, valid_path, data_pairs):
    train_fold = []
    valid_fold = []
    test_fold = []

    train = read_json(train_path)
    train_id = [item['id'] for item in train]
    test = read_json(test_path)
    test_id = [item['id'] for item in test]
    valid_id = []
    if valid_path:
        valid = read_json(valid_path)
        valid_id = [item['id'] for item in valid]

    for pair in data_pairs:
        #pair = tuple(pair)
        if not valid_path:
            if pair['id'] in train_id:
                train_fold.append(pair)
            elif pair['id'] in test_id:
                test_fold.append(pair)
            else:
                valid_fold.append(pair)
        else:
            if pair['id'] in train_id:
                train_fold.append(pair)
            elif pair['id'] in test_id:
                test_fold.append(pair)
            elif pair['id'] in valid_id:
                valid_fold.append(pair)
    return train_fold, test_fold, valid_fold


def load_math23k_ttv(train_path, test_path, valid_path, data_path, group_path, lang):
    # load data
    data = load_raw_data(data_path)
    group_data =  read_json(group_path)

    input_lang, output_lang = load_lang_vocab(lang)

    pairs, generate_nums, copy_nums = pre_data.transfer_num(data)
    pairs = pre_data.attach_group_data(pairs, group_data)
    train_fold, test_fold, valid_fold = get_train_test_fold(train_path,test_path,valid_path,pairs)
    # train_fold[0]: id
    # train_fold[1]: word list
    # train_fold[2]: equation repalced with Ni
    # train_fold[3]: number list
    # train_fold[4]: number position list
    # train_fold[5]: group list- neighbor nodes of numbers
    # train_fold[6]: answer string

    pairs_test = test_fold
    pairs_train = train_fold
    pairs_valid = valid_fold


    # train_pair[0]: code list of words
    # train_pair[1]: int length of valid_pair[0]
    # train_pair[2]: code of equation
    # train_pair[3]: length of valid_pair[2]
    # train_pair[4]: list of numbers
    # train_pair[5]: list of number position
    # train_pair[6]: list of num_stack
    # train_pair[7]: group list
    # train_pair[8]: answer string
    return input_lang, output_lang, pairs_train, pairs_test, pairs_valid

#load single dataset(trainset/testset/validset)as well as group numbers
def load_math23k_data(data_path, group_path,lang):
    # load data
    data = read_json(data_path)
    group_data =  read_json(group_path)

    input_lang, output_lang = load_lang_vocab(lang)

    pairs, generate_nums, copy_nums = pre_data.transfer_num(data)
    pairs = pre_data.attach_group_data(pairs, group_data)

    return input_lang, output_lang, pairs

def write_lang_vocab(input_lang, output_lang, language, output_data_dir="./GraphConstruction", lang_file_name="lang_map.pkl"):
    out_mapfile = "{}/{}_{}".format(output_data_dir, language, lang_file_name)
    print("Write lang_map.pkl to {}".format(out_mapfile))
    with open(out_mapfile, "wb") as out_map:
        pkl.dump([input_lang, output_lang], out_map)

    print(input_lang.vocab_size)
    print(output_lang.vocab_size)


def load_lang_vocab(lang, dir="./GraphConstruction", file_name="lang_map.pkl"):
    langs = pkl.load( open("{}/{}_{}".format(dir, lang, file_name), "rb" ) )
    input_lang, output_lang = langs
    return input_lang, output_lang

def write_prepared_data(output_file_dir,dataset_name,data_alias, mode, batch_num, batch_size, data):
    out_datafile = ''
    if mode =='train':
        out_datafile = "{}/{}_{}_{}_{}.pkl".format(output_file_dir, dataset_name, data_alias, mode, batch_size)
    else:
        out_datafile = "{}/{}_{}_{}.pkl".format(output_file_dir, dataset_name, data_alias, mode)
    print("Write train file to {}".format(out_datafile))
    with open(out_datafile, "wb") as out_data:
        data_list = []
        dic = {'data_alias':data_alias}
        dic.update(dic)
        dic.update({'batch_num':batch_num})
        dic.update({'batch_size':batch_size})
        dic.update({'data':data})
        data_list.append(dic)
        pkl.dump(data_list, out_data)
