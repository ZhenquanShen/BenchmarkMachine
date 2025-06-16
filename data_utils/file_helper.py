import os
import json
import re
import logging
import pandas as pd


class FileHelper:
    @staticmethod
    def json_to_csv(fpath, out_path, tg_names=None):
        if tg_names is None:
            tg_names = []
        FileHelper.check_file_exists(fpath)
        d = FileHelper.read_json_file(fpath)
        rows = d["body"]
        accept_data = []
        if len(tg_names) == 0:
            accept_data = rows
        else:
            for x in rows:
                accept_data.append([x[name] for name in tg_names])
        FileHelper.save_csv_file(accept_data, out_path, mode="w")

    @staticmethod
    def read_vocab(vocab_dir):
        FileHelper.check_file_exists(vocab_dir)
        """读取词汇表"""
        # texts = open_file(vocab_dir).read().strip().split('\n')
        with open(vocab_dir, encoding="utf-8") as f:
            lines = f.read().strip().encode("utf-8").decode("utf-8-sig").split("\n")
            # 如果是py2 则每个值都转化为unicode
            words = [l.strip() for l in lines]
        word2id = dict(zip(words, range(len(words))))
        id2word = dict(zip(range(len(words)), words))
        return words, word2id, id2word

    @staticmethod
    def read_lines(fpath, filter=False):
        FileHelper.check_file_exists(fpath)
        # print("Reading lines...")
        # Read the file and split into lines, remove \ufeff  of first line
        f = (
            open("%s" % (fpath), encoding="utf-8")
            .read()
            .strip()
            .encode("utf-8")
            .decode("utf-8-sig")
        )
        # if remove_space:
        #     f = f.replace(' ', '')
        if filter:
            # reg = '[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~\s]+'  # 用户也可以在此进行自定义过滤字符
            # \s是去空格的，但是同时也把\n去掉了，暂不使用，需要保留，。? \n,
            reg = '[!"#$%&()*+,-./:;<=>?@?★、…【】《》“”‘’！[\\]^_`{|}~]+'
            f = re.sub(reg, "", f)
            f = f.replace(" ", "")

        lines = f.split("\n")
        return lines

    @staticmethod
    def read_dict_lines(fpath):
        FileHelper.check_file_exists(fpath)
        data = FileHelper.read_lines(fpath)
        data = [eval(l) for l in data]
        return data

    @staticmethod
    def check_file_exists(filename):
        try:
            with open(filename) as f:
                return True
        except FileNotFoundError:
            logging.error("file doesn't exist! check file {}".format(filename))
            logging.exception("Exception Logged")
            exit()  # force stop

    @staticmethod
    def check_file_exists_with_maker(filename):
        FileHelper.check_file_dir(filename)
        try:
            with open(filename) as f:
                return True
        except Exception as e:
            # print(e)
            open(filename, "w").close()

    # return False

    @staticmethod
    def read_json_file(fpath):
        FileHelper.check_file_exists(fpath)

        # s = f.read()

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                js = json.load(f)
                return js
        except:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    s = f.read()
                    v = json.dumps(s, ensure_ascii=False)
                    js = json.loads(v)
                    if isinstance(js, str):
                        js = eval(js)
                    return js
            except:
                try:  # json.dumps(eval(s))解决json文件内单引号的问题
                    with open(fpath, "r", encoding="utf-8") as f:
                        s = f.read()
                        js = json.loads(json.dumps(eval(s), ensure_ascii=False))
                        return js
                except:
                    logging.error("read file failed! {}".format(fpath))
                    return []

    @staticmethod
    def read_csv(fpath):
        """read csv file to list

        Args:
            fpath ([str]): [description]

        Returns:
            [list]: [return list]
        """
        import pandas as pd

        a = pd.read_csv(fpath, header=None).values.tolist()
        return a

    @staticmethod
    def read_dict_list_from_json_file(fpath):
        FileHelper.check_file_exists(fpath)
        return FileHelper.read_lines(fpath)

    @staticmethod
    def read_with_array(fpath):
        lines = FileHelper.read_lines(fpath)
        return [[line] for line in lines]

    # 替换字符串中多个指定位置为指定字符
    # p:位置列表，c:对应替换的字符列表

    @staticmethod
    def replace_string_character(string, p, c):
        new = []
        for s in string:
            new.append(s)
        for index, point in enumerate(p):
            new[point] = c[index]
        return "".join(new)

    @staticmethod
    def remove_duplication_list(lists):
        list2 = []
        for i in lists:
            if i not in list2:
                list2.append(i)
        return list2

    @staticmethod
    def list_to_str(a_list, sep=""):
        return sep.join(list(map(str, a_list)))

    @staticmethod
    def save_list_to_file(lines, fpath):
        FileHelper.check_file_dir(fpath)
        with open("%s" % (fpath), "w", encoding="utf-8") as f:
            for item in lines:
                f.write("{}\n".format(item))
            f.close()
            # print("%s" % (fpath))

    @staticmethod
    def save_json_file(json_data, fpath):
        FileHelper.check_file_dir(fpath)
        with open("%s" % (fpath), "w", encoding="utf-8") as f:
            try:
                json.dump(json_data, f, ensure_ascii=False)
            except:
                f.writelines(str(json_data))
                f.close()

    @staticmethod
    def save_json_list_to_file(json_data_list, fpath):  
        FileHelper.check_file_dir(fpath)  # 如果需要，可以取消注释这行代码来检查目录  
        with open(fpath, "w", encoding="utf-8") as f:  
            for item in json_data_list:  
                try:  
                    json_str = json.dumps(item, ensure_ascii=False)  
                    f.write(json_str + "\n")  
                except (TypeError, OverflowError) as e:  
                    # 处理无法序列化为JSON的情况  
                    # 这里可以记录错误，或者采取其他措施  
                    print(f"Failed to serialize item to JSON: {e}")  
                    # 如果仍然需要写入，可以以非JSON格式写入（不推荐，因为会丢失结构信息）  
                    # f.write(str(item) + "\n")  # 不推荐这样做，除非您确定这是您想要的  

    @staticmethod
    def save_csv_file(csv_data, fpath, mode="w", header=False):
        FileHelper.check_file_dir(fpath)
        try:
            f = pd.DataFrame(csv_data)
            f.to_csv(fpath, mode=mode, index=False, header=header)
        except:
            print("write csv file failed! {}".format(fpath))

    @staticmethod
    def save_dict_to_json_file(dic, fpath):
        FileHelper.check_file_dir(fpath)
        with open("%s" % (fpath), "w", encoding="utf-8") as f:
            s = str(dic).replace("'", '"')
            f.writelines(s)
            f.close()

    @staticmethod
    def check_file_dir(fpath):
        """
		check file dir, if it doesn't exist, create it 
		param fpath: file path
		return:
		"""
        parent_path = os.path.dirname(fpath)
        if not os.path.exists(parent_path):
            print("make dir:{}".format(parent_path))
            os.makedirs(parent_path)

    @staticmethod
    def save_dict_list_to_json_file(dic_list, fpath):
        """
		将列表中的dictionary按行保存到json文件中
		:param dic_list: list with dictionary problem_text
		:param fpath: json file path
		:return:
		"""

        FileHelper.check_file_dir(fpath)

        with open("%s" % (fpath), "w", encoding="utf-8") as f:
            s = str(dic_list).replace("'", '"')
            f.writelines(s)
            f.close()
            # for each_dict in dic_list:
            #     json_file.write(json.dumps(each_dict) + ',\n')

    @staticmethod
    def app2list_dict_json(ld, fpath):
        FileHelper.check_file_exists_with_maker(fpath)
        lst_dict = FileHelper.read_json_file(fpath)
        if lst_dict == "":
            lst_dict = []
        lst_dict.extend(ld)
        FileHelper.save_dict_list_to_json_file(lst_dict, fpath)

    @staticmethod
    def append_to_json(ld, fpath):
        FileHelper.check_file_exists_with_maker(fpath)
        lst_dict = FileHelper.read_json_file(fpath)
        if lst_dict == "":
            lst_dict = []
        lst_dict.extend(ld)
        FileHelper.save_dict_list_to_json_file(lst_dict, fpath)

    @staticmethod
    def append_lines_to_file(lines, fpath):
        FileHelper.check_file_dir(fpath)
        with open("%s" % (fpath), "a+", encoding="utf-8") as f:
            for item in lines:
                f.write("{}\n".format(item))
            f.close()

    @staticmethod
    def _is_chinese_char(cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated texts, so they are not treated specially and handled
        # like the all of the other languages.
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False

    @staticmethod
    def is_chinese_chars(text):
        for char in text:
            cp = ord(char)
            if not FileHelper._is_chinese_char(cp):
                print("{} is not a Chinese char in {}!".format(cp, text))
                return False
        return False


# remain = 0  # 剩余byte数
# for x in range(len(str)):
#     if remain == 0:
# 		 if (ord(str[x]) & 0x80) == 0x00:
# 		     remain = 0
# 		 elif (ord(str[x]) & 0xE0) == 0xC0:
# 		     remain = 1
# 		 elif (ord(str[x]) & 0xF0) == 0xE0:
# 		     remain = 2
# 		 elif (ord(str[x]) & 0xF8) == 0xF0:
# 		     remain = 3
# 		 else:
# 		     return False
#     else:
# 		 if not ((ord(str[x]) & 0xC0) == 0x80):
# 		     return False
# 		 remain = remain - 1
# if remain == 0:  # 最后如果remain不等于零，可能没有匹配完整
#     return True
# else:
#     return False
