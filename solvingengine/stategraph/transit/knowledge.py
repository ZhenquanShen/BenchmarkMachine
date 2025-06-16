from data_utils.file_helper import FileHelper as fh
from solvingengine.config import mimic_cfg


class S2Model(object):
    """docstring for Problem."""

    def __init__(self, **arg):
        self.__id = arg.get("id")
        self.__pattern = arg.get("pattern")
        self.__relation_template = arg.get("relation_template")
        self.__var_slot_val = arg.get("var_slot_val")
        self.__var_slot_index = arg.get("var_slot_index")

    @property
    def id(self):
        return self.__id

    @id.setter
    def id(self, value):
        if value is None:
            raise ValueError("Invalid model id")
        else:
            self.__id = value

    @property
    def pattern(self):
        return self.__pattern

    @pattern.setter
    def pattern(self, value):
        if value is None:
            raise ValueError("Invalid pattern")
        else:
            self.__pattern = value

    @property
    def relation_template(self):
        return self.__relation_template

    @relation_template.setter
    def relation_template(self, value):
        if value is None:
            raise ValueError("Invalid relation template")
        self.__relation_template = value

    @property
    def var_slot_val(self):
        return self.__var_slot_val

    @var_slot_val.setter
    def var_slot_val(self, value):
        if value is None:
            raise ValueError("Invalid var_slot_val")
        self.__var_slot_val = value

    @property
    def var_slot_index(self):
        return self.__var_slot_index

    @var_slot_index.setter
    def var_slot_index(self, value):
        if value is None:
            raise ValueError("Invalid var_slot_index")
        self.__var_slot_index = value

    def set_state(self, **arg):
        self.__id = arg.get("id")
        self.__pattern = arg.get("pattern")
        self.__relation_template = arg.get("relation_template")
        self.__var_slot_val = arg.get("var_slot_val")
        self.__var_slot_index = arg.get("var_slot_index")

    def get_state(self):
        d = {
            "id": self.__id,
            "pattern": self.__pattern,
            "relation_template": self.__relation_template,
            "var_slot_val": self.__var_slot_val,
            "var_slot_index": self.__var_slot_index,
        }
        return d


class VS2Model(object):
    """docstring for Problem."""

    def __init__(self, **arg):
        self.__id = arg.get("id")
        self.__pattern = arg.get("pattern")
        self.__relation_template = arg.get("relation_template")
        self.__var_slot_val = arg.get("var_slot_val")
        self.__var_slot_index = arg.get("var_slot_index")

    @property
    def id(self):
        return self.__id

    @id.setter
    def id(self, value):
        if value is None:
            raise ValueError("Invalid model id")
        else:
            self.__id = value

    @property
    def pattern(self):
        return self.__pattern

    @pattern.setter
    def pattern(self, value):
        if value is None:
            raise ValueError("Invalid pattern")
        else:
            self.__pattern = value

    @property
    def relation_template(self):
        return self.__relation_template

    @relation_template.setter
    def relation_template(self, value):
        if value is None:
            raise ValueError("Invalid relation template")
        self.__relation_template = value

    @property
    def var_slot_val(self):
        return self.__var_slot_val

    @var_slot_val.setter
    def var_slot_val(self, value):
        if value is None:
            raise ValueError("Invalid var_slot_val")
        self.__var_slot_val = value

    @property
    def var_slot_index(self):
        return self.__var_slot_index

    @var_slot_index.setter
    def var_slot_index(self, value):
        if value is None:
            raise ValueError("Invalid var_slot_index")
        self.__var_slot_index = value

    def set_state(self, **arg):
        self.__id = arg.get("id")
        self.__pattern = arg.get("pattern")
        self.__relation_template = arg.get("relation_template")
        self.__var_slot_val = arg.get("var_slot_val")
        self.__var_slot_index = arg.get("var_slot_index")

    def get_state(self):
        d = {
            "id": self.__id,
            "pattern": self.__pattern,
            "relation_template": self.__relation_template,
            "var_slot_val": self.__var_slot_val,
            "var_slot_index": self.__var_slot_index,
        }
        return d


class S2ModelPool(object):
    """docstring for Problem."""

    def __init__(self, **arg):
        self.__id_index = arg.get("id_index")
        self.__pattern_index = arg.get("pattern_index")
        self.__template_index = arg.get("template_index")
        self.__length_index = arg.get("length_index")
        self.__keyword_index = arg.get("keyword_index")

    @property
    def id_index(self):
        return self.__id_index

    @id_index.setter
    def id_index(self, value):
        if value is None:
            raise ValueError("Invalid id_index")
        else:
            self.__id_index = value

    @property
    def pattern_index(self):
        return self.__pattern_index

    @pattern_index.setter
    def pattern_index(self, value):
        if value is None:
            raise ValueError("Invalid pattern_index")
        else:
            self.__pattern_index = value

    @property
    def template_index(self):
        return self.__template_index

    @template_index.setter
    def template_index(self, value):
        if value is None:
            raise ValueError("Invalid template_index")
        self.__template_index = value

    @property
    def length_index(self):
        return self.__length_index

    @length_index.setter
    def length_index(self, value):
        if value is None:
            raise ValueError("Invalid length_index")
        self.__length_index = value

    @property
    def keyword_index(self):
        return self.__keyword_index

    @keyword_index.setter
    def keyword_index(self, value):
        if value is None:
            raise ValueError("Invalid keyword_index")
        self.__keyword_index = value

    def set_state(self, **arg):
        self.__id_index = arg.get("id_index")
        self.__pattern_index = arg.get("pattern_index")
        self.__template_index = arg.get("template_index")
        self.__length_index = arg.get("length_index")
        self.__keyword_index = arg.get("keyword_index")

    def get_state(self):
        d = {
            "id_index": self.__id_index,
            "pattern_index": self.__pattern_index,
            "template_index": self.__template_index,
            "length_index": self.__length_index,
            "keyword_index": self.__keyword_index,
        }
        return d


class FileToS2ModelPool(object):
    """docstring for FileToProblemText."""

    def __init__(self, s2_model_pool):
        self.s2_model_pool = s2_model_pool

    def transform(self, transit_name):
        """_summary_
        Args:
            transit_name (string): transform method, must be available in the transits
            --["to_text_model", "to_vector_model"]
        """
        """call transform method"""
        transit_list = ["to_text_model", "to_vector_model"]
        if transit_name in transit_list:
            getattr(self, str("do_" + transit_name))()
        else:
            raise ValueError("Invalid transit name")

    def do_to_text_model(self):
        """call transform method"""
        s2_models = fh.read_json_file(mimic_cfg.transits.s2model.path)
        vcb = fh.read_json_file(mimic_cfg.transits.s2model.vocab)
        keywords = vcb.get("keywords")
        id_index = {}
        pattern_index = {}
        template_index = {}
        length_index = {}
        keyword_index = {}
        for m in s2_models:
            s2_model = S2Model(**m)
            id_index.setdefault(s2_model.id, s2_model)
            pattern_list = str(s2_model.pattern).split(",")
            if pattern_index.get(s2_model.pattern) is None:
                pattern_index.setdefault(s2_model.pattern, [])
                pattern_index[s2_model.pattern].append(s2_model)
            else:
                pattern_index[s2_model.pattern].append(s2_model)
            if template_index.get(s2_model.relation_template) is None:
                template_index.setdefault(s2_model.relation_template, [])
                template_index[s2_model.relation_template].append(s2_model)
            else:
                template_index[s2_model.relation_template].append(s2_model)
            if length_index.get(len(pattern_list)) is None:
                length_index.setdefault(len(pattern_list), [])
                length_index[len(pattern_list)].append(s2_model)
            else:
                length_index[len(pattern_list)].append(s2_model)
            kw = "".join([v for v in pattern_list if v in keywords])
            if keyword_index.get(kw) is None:
                keyword_index.setdefault(kw, [])
                keyword_index[kw].append(s2_model)
            else:
                keyword_index[kw].append(s2_model)
        d = {
            "id_index": id_index,
            "pattern_index": pattern_index,
            "template_index": template_index,
            "length_index": length_index,
            "keyword_index": keyword_index,
        }
        self.s2_model_pool.set_state(**d)

    def do_to_vector_model(self):
        """call transform method"""
        s2_models = fh.read_json_file(mimic_cfg.transits.s2model.path)
        vcb = fh.read_json_file(mimic_cfg.transits.s2model.vocab)
        keywords = vcb.get("keywords")
        pos = vcb.get("pos")
        id_index = {}
        pattern_index = {}
        template_index = {}
        length_index = {}
        keyword_index = {}
        for m in s2_models:
            s2_model = S2Model(**m)
            vs2_model = VS2Model()
            S2ModelToVS2Model(s2_model, vs2_model).transform("onehot")
            id_index.setdefault(vs2_model.id, vs2_model)
            pattern_list = str(s2_model.pattern).split(",")
            if pattern_index.get(s2_model.pattern) is None:
                pattern_index.setdefault(s2_model.pattern, [])
                pattern_index[s2_model.pattern].append(vs2_model)
            else:
                pattern_index[s2_model.pattern].append(vs2_model)
            if template_index.get(s2_model.relation_template) is None:
                template_index.setdefault(s2_model.relation_template, [])
                template_index[s2_model.relation_template].append(vs2_model)
            else:
                template_index[s2_model.relation_template].append(vs2_model)
            if length_index.get(len(pattern_list)) is None:
                length_index.setdefault(len(pattern_list), [])
                length_index[len(pattern_list)].append(vs2_model)
            else:
                length_index[len(pattern_list)].append(vs2_model)
            kw = "".join([v for v in pattern_list if v in keywords])
            if keyword_index.get(kw) is None:
                keyword_index.setdefault(kw, [])
                keyword_index[kw].append(vs2_model)
            else:
                keyword_index[kw].append(vs2_model)
        d = {
            "id_index": id_index,
            "pattern_index": pattern_index,
            "template_index": template_index,
            "length_index": length_index,
            "keyword_index": keyword_index,
        }
        self.s2_model_pool.set_state(**d)


class S2ModelToVS2Model(object):
    """docstring for S2ModelToVectorS2Model."""

    def __init__(self, s2_model_state, vs2_model_state):
        self.s2_model_state = s2_model_state
        self.vs2_model_state = vs2_model_state

    def transform(self, transit_name):
        """_summary_
        Args:
            transit_name (string): transform method, must be available in the transits
        """ """call transform method"""
        transit_list = ["onehot", "bert", "nltp", "gpt3"]
        if transit_name in transit_list:
            getattr(self, str("do_" + transit_name + "_encoding"))()
        else:
            raise ValueError("Invalid transit name")

    def do_onehot_encoding(self):
        vcb = fh.read_json_file(mimic_cfg.transits.s2model.vocab)
        keywords = vcb.get("keywords")
        pos = vcb.get("pos")
        vcb_tokens = keywords + pos
        from mimic.utility import encoder

        vcb_tokens_encodings, vcb_tokens_index = encoder.one_hot_encode(vcb_tokens)
        tokens = str(self.s2_model_state.pattern).split(",")
        vector_pattern = []
        for token in tokens:
            vector_pattern.append(vcb_tokens_encodings[vcb_tokens_index.get(token) - 1])
        d = self.s2_model_state.get_state()
        d.update({"pattern": vector_pattern})
        self.vs2_model_state.set_state(**d)

    def do_nltp_encoding(self):
        pb_text = self.problem_state.text
        segs, hiddens = ltp.seg([pb_text])
        vcb = fh.read_json_file(mimic_cfg.transits.s2model.vocab)
        keywords = vcb.get("keywords")
        pos = vcb.get("pos")
        from mimic.utility import encoder

        pos_encodings, pos_index = encoder.one_hot_encode(pos)
        word_tag = self.annotation_state.pos_tagged_tokens
        token_seq = []
        token_position_seq = []
        vector_seq = []
        word_embeddings = hiddens.get("word_input").cpu().numpy()[0]
        for idx, token, embedding in zip(
            range(0, len(word_tag)), word_tag, word_embeddings
        ):
            if token[0] in keywords:
                token_seq.append(token[0])
                token_position_seq.append(idx)
                vector_seq.append(embedding)
            elif token[1] in pos:
                token_seq.append(token[1])
                token_position_seq.append(idx)
                vector_seq.append(pos_encodings[pos_index.get(token[1])])
        self.vector_seq_state.token_sequence = token_seq
        self.vector_seq_state.token_position_sequence = token_position_seq
        self.vector_seq_state.vector_sequence = vector_seq


s2_model_pool = S2ModelPool()
FileToS2ModelPool(s2_model_pool).transform("to_text_model")
