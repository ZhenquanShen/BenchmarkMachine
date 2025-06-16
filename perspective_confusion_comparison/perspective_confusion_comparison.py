import re
from copy import deepcopy

class Perspective_Confusion_Comparison:
    def __init__(self):
        self.results = []


    def compare_answer(self, ans, gold_ans):
        try:
            if abs(ans - gold_ans) < 1e-2:
                return True
            else:
                return False
        except:
            return False
        
    def compare_equation(self, test, tar):

        if test is None:
            return False
        elif test == tar:
            return True
        else:
            return False
        
    def compare_relation(self, relationset, ground_truth): 
        if relationset is None:
            return False
        elif relationset == ground_truth:
            return True
        else:
            return False

    

    def generate_csv(self, relation_acc, equation_acc, answer_acc, 
                     relation_acc_1, equation_acc_1, answer_acc_1,
                     relation_acc_2, equation_acc_2, answer_acc_2,
                     relation_acc_3, equation_acc_3, answer_acc_3,
                     time):
        data = [
            ["answer_acc_1", "answer_acc_2", "answer_acc_3", "answer_acc"],
            [answer_acc_1, answer_acc_2, answer_acc_3, answer_acc],
            ["equation_acc_1", "equation_acc_2", "equation_acc_3", "equation_acc"],
            [equation_acc_1, equation_acc_2, equation_acc_3, equation_acc],
            ["relation_acc_1", "relation_acc_2", "relation_acc_3", "relation_acc"],
            [relation_acc_1, relation_acc_2, relation_acc_3, relation_acc],
            ["time"],
            [time]
        ]
        with open('result.csv', 'w', newline='') as file:
            for row in data:
                # 将每一行数据转换为逗号分隔的字符串，并写入文件
                file.write(",".join(map(str, row)) + "\n")
        
        print("数据已成功写入 result.csv 文件")

    def compute_prefix_expression(self, pre_fix):
        try: 
            st = list()
            operators = operators = ["+", "-", "^", "*", "/", ".", ")"]
            pre_fix = deepcopy(pre_fix)
            pre_fix.reverse()
            for p in pre_fix:
                if p not in operators:
                    pos = re.search("\d+\(", p)
                    if pos:
                        st.append(eval(p[pos.start(): pos.end() - 1] + "+" + p[pos.end() - 1:]))
                    elif p[-1] == "%":
                        st.append(float(p[:-1]) / 100)
                    else:
                        st.append(eval(p))
                elif p == "+" and len(st) > 1:
                    a = st.pop()
                    b = st.pop()
                    st.append(a + b)
                elif p == "*" and len(st) > 1:
                    a = st.pop()
                    b = st.pop()
                    st.append(a * b)
                elif p == "/" and len(st) > 1:
                    a = st.pop()
                    b = st.pop()
                    if b == 0:
                        return None
                    st.append(a / b)
                elif p == "-" and len(st) > 1:
                    a = st.pop()
                    b = st.pop()
                    st.append(a - b)
                elif p == "^" and len(st) > 1:
                    a = st.pop()
                    b = st.pop()
                    if float(eval(b)) != 2.0 or float(eval(b)) != 3.0:
                        return None
                    st.append(a ** b)
                else:
                    return None
            if len(st) == 1:
                return st.pop()
            return None
        except:
            return None

    def out_expression_list(self, test, output_lang, num_list, num_stack=None):
            max_index = output_lang.n_words
            res = []
            for i in test:
                # if i == 0:
                #     return res
                if i < max_index - 1:
                    idx = output_lang.index2word[i]
                    if idx[0] == "N":
                        if int(idx[1:]) >= len(num_list):
                            return None
                        res.append(num_list[int(idx[1:])])
                    else:
                        res.append(idx)
                else:
                    pos_list = num_stack.pop()
                    c = num_list[pos_list[0]]
                    res.append(c)
            return res