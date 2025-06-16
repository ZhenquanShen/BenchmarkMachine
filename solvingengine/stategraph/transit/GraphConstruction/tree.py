# tree structure in decoder side
# divide sub-node by brackets "()"

class Tree():
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = []
    
    def __str__(self, level = 0):
        ret = ""
        for child in self.children:
            if isinstance(child,type(self)):
                ret += child.__str__(level+1)
            else:
                ret += "\t"*level + str(child) + "\n"
        return ret
    
    def add_child(self,c):
        if isinstance(c,type(self)):
            c.parent = self
        self.children.append(c)
        self.num_children = self.num_children + 1

    def is_tree_instance(self, t):
        if isinstance(t,type(self)):
            return True
        else:
            return False
    
    def to_string(self):
        r_list = []
        for i in range(self.num_children):
            if isinstance(self.children[i], Tree):
                r_list.append("( " + self.children[i].to_string() + " )")
            else:
                r_list.append(str(self.children[i]))
        return "".join(r_list)
    
    def to_list(self, output_lang):
        r_list = []
        for i in range(self.num_children):
            if isinstance(self.children[i], type(self)):
                r_list.append(output_lang.symbol2idx["("])
                cl = self.children[i].to_list(output_lang)
                for k in range(len(cl)):
                    r_list.append(cl[k])
                r_list.append(output_lang.symbol2idx[")"])
            else:
                r_list.append(self.children[i])
        return r_list