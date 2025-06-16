from copy import deepcopy
import re
from data_utils import tree

# An expression tree node
class Et:
    # Constructor to create a node
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def get_tree_depth(root):
    if not root:
        return 0
    left, right = 0, 0
    if root.left:
        left = get_tree_depth(root.left)
    if root.right:
        right = get_tree_depth(root.right)
    return max(left, right)+1

# Returns root of constructed tree for given postfix expression
def construct_exp_tree(postfix):
    stack = []

    # Traverse through every character of input expression
    for char in postfix:

        # if operand, simply push into stack
        if char not in ["+", "-", "*", "/", "^"]:
            t = Et(char)
            stack.append(t)
        # Operator
        else:
            # Pop two top nodes
            t = Et(char)
            t1 = stack.pop()
            t2 = stack.pop()

            # make them children
            t.right = t1
            t.left = t2

            # Add this subexpression to stack
            stack.append(t)
    # Only element  will be the root of expression tree
    t = stack.pop()
    return t


def from_infix_to_postfix(expression):
    st = list()
    res = list()
    priority = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
    for e in expression:
        if e in ["(", "["]:
            st.append(e)
        elif e == ")":
            c = st.pop()
            while c != "(":
                res.append(c)
                c = st.pop()
        elif e == "]":
            c = st.pop()
            while c != "[":
                res.append(c)
                c = st.pop()
        elif e in priority:
            while len(st) > 0 and st[-1] not in ["(", "["] and priority[e] <= priority[st[-1]]:
                res.append(st.pop())
            st.append(e)
        else:
            res.append(e)
    while len(st) > 0:
        res.append(st.pop())
    return res


def from_infix_to_prefix(expression):
    st = list()
    res = list()
    priority = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
    expression = deepcopy(expression)
    expression.reverse()
    for e in expression:
        if e in [")", "]"]:
            st.append(e)
        elif e == "(":
            c = st.pop()
            while c != ")":
                res.append(c)
                c = st.pop()
        elif e == "[":
            c = st.pop()
            while c != "]":
                res.append(c)
                c = st.pop()
        elif e in priority:
            while len(st) > 0 and st[-1] not in [")", "]"] and priority[e] < priority[st[-1]]:
                res.append(st.pop())
            st.append(e)
        else:
            res.append(e)
    while len(st) > 0:
        res.append(st.pop())
    res.reverse()
    return res



class Stack(object):
    def __init__(self):	
        self.list = []
    def isEmpty(self):
        return self.list == []
    def push(self, item):
        self.list.append(item)
    def pop(self):
        return self.list.pop()
    def top(self):
        return self.list[len(self.list)-1]
    def size(self):
        return len(self.list)



def from_prefix_to_infix(expression):
    s = Stack()
    list = expression
    for par in list:
        if par in "+-*/^":
            s.push(par)
        else:   
            if (not s.isEmpty()) and (s.top() in '+-*/^'):
                s.push(par)
            else:
                while (not s.isEmpty()) and (not s.top() in '+-*/'):
                    shu = s.pop()
                    fu = s.pop()
                    par = '('+shu+fu+par+')'
                s.push(str(par))
    return s.pop()

def out_expression_list(test, output_lang, num_list, num_stack=None):
    max_index = output_lang.vocab_size
    res = []
    for i in test:
        # if i == 0:
        #     return res
        if isinstance(i, str):
            res.append(i)
        elif i < max_index - 1:
            idx = output_lang.idx2symbol[i]
            if idx[0] == "N":
                if int(idx[1:]) >= len(num_list):
                    return None
                res.append(num_list[int(idx[1:])])
            else:
                res.append(idx)
        elif len(num_stack):
            pos_list = num_stack.pop()
            c = num_list[pos_list[0]]
            res.append(c)
    return res


def compute_postfix_expression(post_fix):
    st = list()
    operators = ["+", "-", "^", "*", "/"]
    for p in post_fix:
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
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a * b)
        elif p == "/" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            if a == 0:
                return None
            st.append(b / a)
        elif p == "-" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(b - a)
        elif p == "^" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a ** b)
        else:
            return None
    if len(st) == 1:
        return st.pop()
    return None


def compute_prefix_expression(pre_fix):
    st = list()
    operators = ["+", "-", "^", "*", "/"]
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
            #removed by He, 20220822
            #if float(eval(b)) != 2.0 or float(eval(b)) != 3.0:
            #    return None
            st.append(a ** b)
        else:
            return None
    if len(st) == 1:
        return st.pop()
    return None

#cover digit expression to Ni expression
def convert_to_ni(di_expression, nums):
    nums_fraction = []

    for num in nums:
        if re.search("\d*\(\d+/\d+\)\d*", num):
            nums_fraction.append(num)
    nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

    def seg_and_tag(st):  # seg the equation and tag the num
        res = []
        for n in nums_fraction:
            if n in st:
                p_start = st.find(n)
                p_end = p_start + len(n)
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                if nums.count(n) > 0:
                    res.append("N"+str(nums.index(n)))
                else:
                    res.append(n)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
        pos_st = re.search("\d+\.\d+%?|\d+%?", st)
        if pos_st:
            p_start = pos_st.start()
            p_end = pos_st.end()
            if p_start > 0:
                res += seg_and_tag(st[:p_start])
            st_num = st[p_start:p_end]
            if nums.count(st_num) > 0:
                res.append("N"+str(nums.index(st_num)))
            else:
                res.append(st_num)
            if p_end < len(st):
                res += seg_and_tag(st[p_end:])
            return res
        for ss in st:
            res.append(ss)
        return res

    ni_expression = seg_and_tag(di_expression)
    return ni_expression


#for example, x=(1*0.01)*2 will be converted to [x,=, [1,*,0.01], *, 2]
def convert_to_tree(r_list, i_left, i_right, output_lang):
    t = tree.Tree()
    level = 0
    left = -1

    for i in range(i_left, i_right):

        if r_list[i] == output_lang.word2index['(']:
            if level == 0:
                left = i
            level = level + 1
        elif r_list[i] == output_lang.word2index[')']:
            level = level -1
            if level == 0:
                if i == left+1:
                    c = r_list[i]
                else:
                    c = convert_to_tree(r_list, left + 1, i, output_lang)
                t.add_child(c)
        elif level == 0:
            t.add_child(r_list[i])

    return t

def convert_to_string(idx_list, output_lang):
    w_list = []
    for i in range(len(idx_list)):
        w_list.append(output_lang.index2word[int(idx_list[i])])
    return " ".join(w_list)

def expression_code_validation(exp_code,output_lang):
    exp_code = [int(c) for c in exp_code]
    num_left_paren = sum(1 for c in exp_code if output_lang.index2word[int(c)]== "(")
    num_right_paren = sum(1 for c in exp_code if output_lang.index2word[int(c)]== ")")
    diff = num_left_paren - num_right_paren
    if diff > 0:
        for i in range(diff):
            exp_code.append(output_lang.word2index[")"])
    elif diff < 0:
        candidate = exp_code[:diff]
    exp_str = convert_to_string(exp_code, output_lang)
    return exp_code, exp_str
'''
def generate_expression(candidate, reference, output_lang):
    ref_str = convert_to_string(reference, output_lang)
    cand_str = convert_to_string(candidate, output_lang)
    return candidate, cand_str, reference, ref_str
'''
def convert_ni_str_to_equation(ni_str, num_list, split = False):
    ni_dict = {}
    for idx in range(len(num_list)):
        key = "N"+str(idx)
        ni_dict[key] = num_list[idx]
    eq = []
    ni_str_list = ni_str.split(' ')
    for idx in range(len(ni_str_list)):
        _str = ni_str_list[idx]
        if _str in ni_dict:
            _eq = ni_dict[_str]
            if _eq[-1] == "%":
                eq.append(str(float(_eq[:-1]) / 100))
            else:
                eq.append(_eq)
        else:
            if _str == '[':
                _str = '('
            elif _str == ']':
                _str = ')'
            eq.append(_str)
    if not split:
        eq_str = ''.join(eq)
    else:
        eq_str = ' '.join(eq)
    return eq_str


# remove the superfluous brackets
def remove_brackets(x):
    y = x
    if x[0] == "(" and x[-1] == ")":
        x = x[1:-1]
        flag = True
        count = 0
        for s in x:
            if s == ")":
                count -= 1
                if count < 0:
                    flag = False
                    break
            elif s == "(":
                count += 1
        if flag:
            return x
    return y