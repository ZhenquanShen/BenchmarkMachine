from solvingengine import SolvingEngine
from perspective_confusion_comparison import Perspective_Confusion_Comparison
from data_utils import DataLoader
import torch

class BenchmarkMachine:
    def __init__(self):
        self.problem_list = DataLoader().loading()
        self.solving_engine = SolvingEngine()
        self.perspective_confusion_comparison = Perspective_Confusion_Comparison()
        self.time_for_one = 0
        self.total_time = 0
        self.equation_acc = 0
        self.answer_acc = 0
        self.relation_acc = 0
        self.equation_acc_1 = 0
        self.answer_acc_1 = 0
        self.relation_acc_1 = 0
        self.equation_acc_2 = 0
        self.answer_acc_2 = 0
        self.relation_acc_2 = 0
        self.equation_acc_3 = 0
        self.answer_acc_3 = 0
        self.relation_acc_3 = 0

    def get_time_for_one(self):   
        self.time_for_one =  self.solving_engine.get_elapsed_time()
        return self.time_for_one
    
    def reset_total_time(self):
        self.total_time = 0
    
    def get_total_time(self):
        return self.total_time
    
    def get_equation_acc(self):
        return self.equation_acc
    
    def get_answer_acc(self):
        return self.answer_acc
    
    def get_relation_acc(self):
        return self.relation_acc
    
    def get_equation_acc_1(self):
        return self.equation_acc_1
    
    def get_answer_acc_1(self):
        return self.answer_acc_1
    
    def get_relation_acc_1(self):
        return self.relation_acc_1

    def get_equation_acc_2(self):
        return self.equation_acc_2
    
    def get_answer_acc_2(self):
        return self.answer_acc_2
    
    def get_relation_acc_2(self):
        return self.relation_acc_2
    
    def get_equation_acc_3(self):
        return self.equation_acc_3
    
    def get_answer_acc_3(self):
        return self.answer_acc_3
    
    def get_relation_acc_3(self):
        return self.relation_acc_3
    

    def run(self):
        self.reset_total_time()

        pairs, input_lang, output_lang, generate_nums, generate_num_ids, copy_nums = DataLoader.preprocess(self.problem_list) 
        # pairs, input_lang, output_lang, generate_nums, generate_num_ids, copy_nums, tree_pairs, graph_li_test = DataLoader.preprocess(self.problem_list)

        i = 0
        i1 = 0
        i2 = 0
        i3 = 0
        answer_ac = 0
        equation_ac = 0
        relation_ac = 0
        answer_ac_1 = 0
        equation_ac_1 = 0
        relation_ac_1 = 0
        answer_ac_2 = 0
        equation_ac_2 = 0
        relation_ac_2 = 0
        answer_ac_3 = 0
        equation_ac_3 = 0
        relation_ac_3 = 0

        try:   
            for i in range(len(self.problem_list)):
            
                prob = self.problem_list[i]
                feed_input_ap = {
                    "id": str(prob.get("id")),
                    "text": prob.get("zh_text"),
                    "label": str(prob.get("label")),
                    # "text": prob.get("original_text"),
                    "segmented_tokens": prob.get("segmented_text"),
                    "gold_equation_system":prob.get("equation"),
                    "gold_answer": str(prob.get("ans")),
                    "gold_relation":prob.get("relation"),
                    "pair": pairs[i],
                    "input_lang": input_lang,
                    "output_lang": output_lang,
                    "generate_nums": generate_nums,
                    "generate_num_ids": generate_num_ids,
                    "copy_nums": copy_nums,
                    # "tree_pairs": tree_pairs[i],
                    # "graph_batch": graph_li_test[i]
                }

                try:
                    self.solving_engine.run(feed_input_ap)  
                    

                    self.time_for_one = self.get_time_for_one()
                    self.total_time = self.total_time + self.time_for_one

                    equ = self.solving_engine.get_sharedata().equation_system
                    ans = self.solving_engine.get_sharedata().equation_solution
                    rel = self.solving_engine.get_sharedata().relation_set

                    feed_input_ap["gold_equation_system"] = self.perspective_confusion_comparison.out_expression_list(feed_input_ap["pair"][2], feed_input_ap["output_lang"], feed_input_ap["pair"][4], feed_input_ap["pair"][6])
                    feed_input_ap["gold_answer"] = self.perspective_confusion_comparison.compute_prefix_expression(feed_input_ap["gold_equation_system"])

                    ans_ac = self.perspective_confusion_comparison.compare_answer(ans, feed_input_ap["gold_answer"])
                    equ_ac = self.perspective_confusion_comparison.compare_equation(equ, feed_input_ap["gold_equation_system"])
                    rel_ac = self.perspective_confusion_comparison.compare_relation(rel, feed_input_ap['gold_relation'])

                    if feed_input_ap["label"] == "type_1":
                        if ans_ac:
                            answer_ac_1 += 1
                        if equ_ac:
                            equation_ac_1 += 1
                        if rel_ac:
                            relation_ac_1 += 1
                        i1 = i1+1


                    if feed_input_ap["label"] == "type_2":
                        if ans_ac:
                            answer_ac_2 += 1
                        if equ_ac:
                            equation_ac_2 += 1
                        if rel_ac:
                            relation_ac_2 += 1
                        i2 = i2+1

                    if feed_input_ap["label"] == "type_3":
                        if ans_ac:
                            answer_ac_3 += 1
                        if equ_ac:
                            equation_ac_3 += 1
                        if rel_ac:
                            relation_ac_3 += 1
                        i3 = i3+1
                    

                    if ans_ac:
                        answer_ac += 1
                    if equ_ac:
                        equation_ac += 1
                    if rel_ac:
                        relation_ac += 1
                    i = i+1
                except:
                    continue        
        
        except:
            self.equation_acc = float(equation_ac) / i
            self.answer_acc = float(answer_ac) / i
            self.relation_acc = float(relation_ac) / i
            self.equation_acc_1 = float(equation_ac_1) / i1
            self.answer_acc_1 = float(answer_ac_1) / i1
            self.relation_acc_1 = float(relation_ac_1) / i1
            self.equation_acc_2 = float(equation_ac_2) / i2
            self.answer_acc_2 = float(answer_ac_2) / i2
            self.relation_acc_2 = float(relation_ac_2) / i2
            self.equation_acc_3 = float(equation_ac_3) / i3
            self.answer_acc_3 = float(answer_ac_3) / i3
            self.relation_acc_3 = float(relation_ac_3) / i3
            with open("output.txt") as file:
                file.write(f"{equation_ac} {answer_ac} {relation_ac} {equation_ac_1} {answer_ac_1} {relation_ac_1} {equation_ac_2} {answer_ac_2} {relation_ac_2} {equation_ac_3} {answer_ac_3} {relation_ac_3}")   # 使用f-string格式化写入 


if __name__ == "__main__":
    benchmark_machine = BenchmarkMachine()
    benchmark_machine.run()
    equation_acc = benchmark_machine.get_equation_acc()
    answer_acc = benchmark_machine.get_answer_acc()
    relation_acc = benchmark_machine.get_relation_acc()
    equation_acc_1 = benchmark_machine.get_equation_acc_1()
    answer_acc_1 = benchmark_machine.get_answer_acc_1()
    relation_acc_1 = benchmark_machine.get_relation_acc_1()
    equation_acc_2 = benchmark_machine.get_equation_acc_2()
    answer_acc_2 = benchmark_machine.get_answer_acc_2()
    relation_acc_2 = benchmark_machine.get_relation_acc_2()
    equation_acc_3 = benchmark_machine.get_equation_acc_3()
    answer_acc_3 = benchmark_machine.get_answer_acc_3()
    relation_acc_3 = benchmark_machine.get_relation_acc_3()
    time = benchmark_machine.get_total_time()
    benchmark_machine.perspective_confusion_comparison.generate_csv(relation_acc, equation_acc, answer_acc, 
                                                                    relation_acc_1, equation_acc_1, answer_acc_1,
                                                                    relation_acc_2, equation_acc_2, answer_acc_2,
                                                                    relation_acc_3, equation_acc_3, answer_acc_3,
                                                                    time)