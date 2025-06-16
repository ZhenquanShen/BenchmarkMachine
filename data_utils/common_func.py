
from decimal import Decimal


class CommonFunction:
    @staticmethod
    def list_remove_duplicates(one_dim_list):
        '''
        去除list重复值,且保持原来顺序
        :param one_dim_list: type(list or ndarray)should be one dimension
        :return: a new list which keep original list order
        '''
        new_list = []  # default
        # var_type = type(one_dim_list)
        # if var_type is not np.ndarray:
        # 	one_dim_list = np.array(one_dim_list)
        #
        # if len(one_dim_list.shape) > 1:
        # 	# raise ValueError("the list must be one dimension!!!")
        # 	#
        # 	print()
        # else:
        # 	if len(one_dim_list) > 0:  # 非空列表
        # 		dim_type = type(one_dim_list[0])
        # 		if dim_type is 'dict':
        # 			one_dim_list = np.array(one_dim_list)
        # 		else:
        # 			new_list = list(set(one_dim_list))
        # 			new_list.sort(key=list(one_dim_list).index)
        # 	else:
        # 		raise ValueError("the list can't be null!!!")

        for i in one_dim_list:
            if i not in new_list:
                new_list.append(i)

        return new_list

    @staticmethod
    def list_dimension_two_2_one(list_2dim):
        '''
        将二维列表转换成一维列表
        :param list_2dim:
        :return:
        '''
        return [i for item in list_2dim for i in item]

    @staticmethod
    def list_dimension_one_2_two(list_1dim):
        '''
        将一维列表转换成二维列表
        :param list_1dim:
        :return:
        '''
        return [[i] for i in list_1dim]

    @staticmethod
    def lists_combination(lists, code=''):
        '''
         输出其中每个列表所有元素可能的所有排列组合
        :param lists: 输入多个列表组成的列表
        :param code: 用于分隔每个元素
        :return:
        '''
        '''
        reduce() 函数会对参数序列中元素进行累积。
        函数将一个数据集合（链表，元组等）中的所有数据进行下列操作：
        用传给 reduce 中的函数 function（有两个参数）先对集合中的第 1、2 个元素进行操作，
        得到的结果再与第三个数据用 function 函数运算，最后得到一个结果。也就是lists的每一行出一个元素
        '''
        try:
            import reduce
        except:
            from functools import reduce

        def myfunc(list1, list2):
            return [str(i) + code + str(j) for i in list1 for j in list2]

        # 此处要考虑lists只有一行的情况，组合方式可能要发生变化
        if len(lists) == 1:
            if len(lists[0]) > 0:
                lists = [[v] for v in lists[0]]
            else:
                return []

        return reduce(myfunc, lists)

    @staticmethod
    def list_equal_allocation(vlist, size):
        """[summary]

        Args:
            vlist ([list]): [description]
            size ([int]): [num]

        Yields:
            [list]: [description]
        """
        batches = []
        for i in range(0, len(vlist), size):
            batches.append(vlist[i:i + size])
        return batches

    @staticmethod
    def sorted_dict_values_by_key(adict, reverse=False):
        '''
        将数据字典的键排序，并且返回排序后的键对应的值
        :param adict:
        :param reverse:True 降序；False 升序. default is False
        :return:返回排序后的键对应的值的列表
        '''
        keys = sorted(adict.keys(), reverse)
        return [adict[key] for key in keys]

    @staticmethod
    def charQ2B(uchar):
        """单个字符 全角转半角"""
        inside_code = ord(uchar)
        if inside_code == 0x3000:
            inside_code = 0x0020
        else:
            inside_code -= 0xfee0
        if inside_code < 0x0020 or inside_code > 0x7e:  #转完之后不是半角字符返回原来的字符
            return uchar
        return chr(inside_code)

    @staticmethod
    def stringQ2B(ustring):
        '''把字符串全角转半角'''
        return "".join([CommonFunction.charQ2B(uchar) for uchar in ustring])

    @staticmethod
    def is_number(s):
        '''
        @Description: 

        @Args: 
          param: object

        @Returns: if it is number return true,else false.
        '''

        try:
            float(s)
            return True
        except ValueError:
            pass

        # try:
        #     import unicodedata
        #     unicodedata.numeric(s)
        #     return True
        # except (TypeError, ValueError):
        #     pass

        return False

    @staticmethod
    def is_fraction(s):
        '''
        @Description: 

        @Args: 
          param: object

        @Returns: if it is Fraction return true,else false.
        '''
        # is_f = False
        # try:
        #     Fraction(s)
        #     is_f = True
        # except:
        #     is_f = False
        values = s.split('/')
        is_f = len(values) == 2 or len(values) == 3 and all(i.isdigit()
                                                            for i in values)

        return is_f

    @staticmethod
    def is_int(s):
        if isinstance(s, int):
            return True
        else:
            return False

    @staticmethod
    def is_opt_object(s):
        r = False
        if CommonFunction.is_number(s):
            r = True
        elif CommonFunction.is_fraction(s):
            r = True
        else:
            r = False
        return r

    @staticmethod
    def opt_format(s):
        if CommonFunction.is_number(s):
            r = float(s)
        elif CommonFunction.is_fraction(s):
            from fractions import Fraction
            r = Fraction(s)
        else:
            r = s
        return r

    @staticmethod
    def is_Chinese(word):
        for ch in word:
            if '\u4e00' <= ch <= '\u9fff':
                return True
        return False

    @staticmethod
    def cn_num2arabic_num(text_tagged):
        import pycnnum
        new_text_tagged = []
        cn_num_dict = {
            '一': '1',
            '二': '2',
            '三': '3',
            '四': '4',
            '五': '5',
            '六': '6',
            '七': '7',
            '八': '8',
            '九': '9'
        }
        for word, tag in text_tagged:
            if word in cn_num_dict.keys() and tag == "m":
                new_text_tagged.append((cn_num_dict[word], tag))
            else:
                new_text_tagged.append((word, tag))
        return new_text_tagged

    @staticmethod
    def remove_exponent(num):
        return num.to_integral() if num == num.to_integral(
        ) else num.normalize()

    @staticmethod
    def sci2str(x):
        # if 'e' in str(x):
        #     if x > 0 and x < 1:
        #         import re
        #         exp_pattern = re.compile('e([+\-]\d+)')
        #         exp = exp_pattern.findall(str(x))[0]
        #         if '-0' in exp:
        #             exp = str(exp).replace('-', '').replace('0', '')
        #             exp = str(int(exp) + 1)
        #         fstr = '{:.' + exp + 'f}'
        #         y = fstr.format(x)
        #     else:
        #         y = '{:.2f}'.format(x)
        # else:
        #     y = x
        y = CommonFunction.remove_exponent(Decimal(str(x)))
        return str(y)

    @staticmethod
    def list_sort_by_len(x, reverse=True):
        # A function that returns the length of the value:
        def length(e):
            return len(e)

        x.sort(reverse=reverse, key=length)
        return x
