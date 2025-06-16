import copy


class IQRClassify(object):
    def key_to_val(self, entity_list, iqr_list, key_dict, val_dict):
        temp_iqr_list = copy.deepcopy(iqr_list)
        relation_list = []
        for iqr_str in temp_iqr_list:

            relation = copy.deepcopy(iqr_str)
            var_entity = {}
            for key in key_dict:
                # 如果key存在于这个关系中
                if key in iqr_str:
                    # 查找这个key的候选值是否在eqr实体表中出现
                    for value in val_dict[key]:
                        # 匹配，将关系中的key替换为eqr中的实体
                        if value in entity_list:
                            # 记录隐含关系中在题目文本中的实体
                            for idx, ent in enumerate(entity_list):
                                if value == ent:
                                    var_entity.setdefault(str(idx), ent)
                            relation = relation.replace(key, value)
                            break
                        # 未匹配，将关系中的key替换为iqr中的实体
                        elif value == val_dict[key][-1]:
                            relation = relation.replace(key, key_dict[key])
            relation_list.append(
                {"relation": relation, "var_entity": var_entity})

        return relation_list


# 23种关系类
# 普4类
class distance_model(IQRClassify):
    def __init__(self):
        self.key_dict = {'v1': '速度', 'v2': '时间', 'v3': '路程'}
        self.val_dict = {
            'v1': ['速度', '每秒', '每分钟', '每小时', '每分', '每时', '每天'],
            'v2': ['用了', '花了', '需要', '耗时', '经过', '用', '花', '需'],
            'v3': ['行驶', '飞行', '相距', '跑了', '跑步', '走了', '距离', '通过', '长', '行', '骑', '走']
        }
        # iqr的关系初始化由key组成
        self.iqr_list = ['v3=v1*v2']

    def get_iqr_list(self, entity_list):
        return self.key_to_val(entity_list, self.iqr_list, self.key_dict,
                               self.val_dict)


class bridge_cross_model(IQRClassify):
    def __init__(self):
        self.key_dict = {'v1': '速度', 'v2': '时间', 'v3': '车长', 'v4': '桥长'}
        self.val_dict = {
            'v1': ['速度', '每秒', '每分钟', '每小时', '每分', '每时', '每天'],
            'v2': ['用了', '花了', '需要', '耗时', '经过', '用', '花', '需', '要'],
            'v3': ['列车长', '火车长', '列车', '火车', '车身'],
            'v4': ['大桥', '隧道', '桥长', '隧道长', '桥梁']
        }
        # iqr的关系初始化由key组成
        self.iqr_list = ['v3+v4=v1*v2']

    def get_iqr_list(self, entity_list):
        return self.key_to_val(entity_list, self.iqr_list, self.key_dict,
                               self.val_dict)


class interest_model(IQRClassify):
    def __init__(self):
        self.key_dict = {
            'v1': '本金',
            'v2': '年限',
            'v3': '利率',
            'v5': '利息',
            'v7': '本金和利息'
        }
        self.val_dict = {
            'v1': ['本金', '存入', '存了', '存', '贷款', '买了', '买入', '国债', '债券'],
            'v2': ['存期', '定期', '整存整取', '存'],
            'v3': ['年利率', '利率'],
            'v5': ['利息'],
            'v7': ['本金和利息', '总共获得', '一共', '总共', '取回', '拿到']
        }
        self.iqr_list = ['v5=v1*v3*v2', 'v7=v1+v5']

    def get_iqr_list(self, entity_list):
        return self.key_to_val(entity_list, self.iqr_list, self.key_dict,
                               self.val_dict)


class partwhole_model(IQRClassify):
    def __init__(self):
        self.key_dict = {'v1': '已经', 'v2': '剩下', 'v3': '一共'}
        self.val_dict = {
            'v1': ['已经', '已有'],
            'v2': ['剩下', '剩余', '还剩', '还需', '还要', '还有', '还', '剩'],
            'v3': ['一共', '总共', '共有', '距', '共', '全书', '全长']
        }
        # iqr的关系初始化由key组成
        self.iqr_list = ['v3=v1+v2']

    def get_iqr_list(self, entity_list):
        # temp_iqr_list = copy.deepcopy(self.iqr_list)
        # for index, iqr_str in enumerate(temp_iqr_list):
        #     for key in self.key_dict:
        #         # 如果key存在于这个关系中
        #         if key in iqr_str:
        #             # 查找这个key的候选值是否在eqr实体表中出现
        #             for value in self.val_dict[key]:
        #                 flag = 0
        #                 # 遍历实体
        #                 for entity in entity_list:
        #                     # 匹配，将关系中的key替换为eqr中的实体。 匹配成功后flag变为-1，跳出两层循环
        #                     if value in entity:
        #                         temp_iqr_list[index] = temp_iqr_list[index].replace(key, entity)
        #                         flag = -1s
        #                         break
        #                     # 所有实体都未匹配成功，将关系中的key替换为iqr中的实体
        #                     elif value == self.val_dict[key][-1]:
        #                         temp_iqr_list[index] = temp_iqr_list[index].replace(key, self.key_dict[key])
        #                 if flag == -1:
        #                     break
        # return temp_iqr_list
        return self.key_to_val(entity_list, self.iqr_list, self.key_dict,
                               self.val_dict)


# 长方类
class rec_circum_model(IQRClassify):
    def __init__(self):
        self.key_dict = {'v1': '长', 'v2': '宽', 'v3': '周长'}
        self.val_dict = {
            'v1': ['长'],
            'v2': ['宽'],
            'v3': ['周长', '铁丝', '篱笆', '绳子', '栏杆', '圈', '栅栏']
        }
        self.iqr_list = ['v3=(v1+v2)*2']

    def get_iqr_list(self, entity_list):
        return self.key_to_val(entity_list, self.iqr_list, self.key_dict,
                               self.val_dict)


class rec_area_model(IQRClassify):
    def __init__(self):
        self.key_dict = {'v1': '长', 'v2': '宽', 'v3': '面积'}
        self.val_dict = {'v1': ['长'], 'v2': ['宽'], 'v3': ['面积']}
        # iqr的关系初始化由key组成
        self.iqr_list = ['v3=v1*v2']

    def get_iqr_list(self, entity_list):
        return self.key_to_val(entity_list, self.iqr_list, self.key_dict,
                               self.val_dict)


class cuboid_area_model(IQRClassify):
    def __init__(self):
        self.key_dict = {'v1': '长', 'v2': '宽', 'v3': '高', 'v4': '表面积'}
        self.val_dict = {
            'v1': ['长'],
            'v2': ['宽'],
            'v3': ['高', '深'],
            'v4': ['表面积', '面积', '铁皮']
        }
        self.iqr_list = ['v4=(v1*v2+v2*v3+v3*v1)*2']

    def get_iqr_list(self, entity_list):
        return self.key_to_val(entity_list, self.iqr_list, self.key_dict,
                               self.val_dict)


class cuboid_vol_model(IQRClassify):
    def __init__(self):
        self.key_dict = {
            'v1': '长',
            'v2': '宽',
            'v3': '高',
            'v4': '底面积',
            'v5': '体积'
        }
        self.val_dict = {
            'v1': ['长'],
            'v2': ['宽'],
            'v3': ['高'],
            'v4': ['底面积', '横截面积', '底面', '横截面'],
            'v5': ['体积', '容积', '容量']
        }
        # iqr的关系初始化由key组成
        self.iqr_list = ['v5=v1*v2*v3', 'v5=v4*v3']

    def get_iqr_list(self, entity_list):
        return self.key_to_val(entity_list, self.iqr_list, self.key_dict,
                               self.val_dict)


# 正方类
class square_circum_model(IQRClassify):
    def __init__(self):
        self.key_dict = {'v1': '边长', 'v2': '周长'}
        self.val_dict = {'v1': ['边'], 'v2': ['周长', '铁丝', '篱笆', '绳子']}
        self.iqr_list = ['v2=4*v1']

    def get_iqr_list(self, entity_list):
        return self.key_to_val(entity_list, self.iqr_list, self.key_dict,
                               self.val_dict)


class square_are_model(IQRClassify):
    def __init__(self):
        self.key_dict = {'v1': '边长', 'v2': '面积'}
        self.val_dict = {'v1': ['边长'], 'v2': ['面积']}
        self.iqr_list = ['v2=v1*v1']

    def get_iqr_list(self, entity_list):
        return self.key_to_val(entity_list, self.iqr_list, self.key_dict,
                               self.val_dict)


class cube_area_model(IQRClassify):
    def __init__(self):
        self.key_dict = {'v1': '棱长', 'v2': '表面积'}
        self.val_dict = {'v1': ['棱长', '边长'], 'v2': ['表面积', '面积', '铁皮']}
        self.iqr_list = ['v2=v1*v1*6']

    def get_iqr_list(self, entity_list):
        return self.key_to_val(entity_list, self.iqr_list, self.key_dict,
                               self.val_dict)


class cube_vol_model(IQRClassify):
    def __init__(self):
        self.key_dict = {'v1': '棱长', 'v2': '体积'}
        self.val_dict = {'v1': ['棱长', '边长'], 'v2': ['体积', '容积', '容量']}
        self.iqr_list = ['v2=v1*v1*v1']

    def get_iqr_list(self, entity_list):
        return self.key_to_val(entity_list, self.iqr_list, self.key_dict,
                               self.val_dict)


# 圆类
class circular_circum_model(IQRClassify):
    def __init__(self):
        self.key_dict = {'v1': '半径', 'v2': '直径', 'v3': '周长'}
        self.val_dict = {
            'v1': ['半径'],
            'v2': ['直径'],
            'v3': ['周长', '一圈', '铁丝', '篱笆', '绳子']
        }
        self.iqr_list = ['v3=3.14*v2', 'v2=2*v1']

    def get_iqr_list(self, entity_list):
        return self.key_to_val(entity_list, self.iqr_list, self.key_dict,
                               self.val_dict)


class circular_area_model(IQRClassify):
    def __init__(self):
        self.key_dict = {'v1': '半径', 'v2': '直径', 'v3': '面积'}
        self.val_dict = {'v1': ['半径'], 'v2': ['直径'], 'v3': ['面积']}
        self.iqr_list = ['v3=3.14*v1*v1', 'v2=2*v1']

    def get_iqr_list(self, entity_list):
        return self.key_to_val(entity_list, self.iqr_list, self.key_dict,
                               self.val_dict)


class cylinder_area_model(IQRClassify):
    def __init__(self):
        self.key_dict = {
            'v1': '底面半径',
            'v2': '底面直径',
            'v3': '底面积',
            'v4': '底面周长',
            'v5': '高',
            'v6': '侧面积',
            'v7': '表面积'
        }
        self.val_dict = {
            'v1': ['底面半径', '半径'],
            'v2': ['底面直径', '直径'],
            'v3': ['底面积', '面积', '底面'],
            'v4': ['底面周长', '周长'],
            'v5': ['高', '深', '长'],
            'v6': ['侧面积', '侧面'],
            'v7': ['表面积', '铁皮'],
        }
        self.iqr_list = [
            'v7=v3+v6', 'v3=3.14*v1*v1', 'v2=2*v1', 'v4=3.14*v2', 'v6=v4*v5'
        ]

    def get_iqr_list(self, entity_list):
        return self.key_to_val(entity_list, self.iqr_list, self.key_dict,
                               self.val_dict)


class cylinder_vol_model(IQRClassify):
    def __init__(self):
        self.key_dict = {
            'v1': '底面半径',
            'v2': '底面直径',
            'v3': '底面积',
            'v4': '高',
            'v5': '体积'
        }
        self.val_dict = {
            'v1': ['底面半径', '半径'],
            'v2': ['底面直径', '直径'],
            'v3': ['底面积', '面积', '底面', '横截面积', '横截面'],
            'v4': ['高', '深', '长'],
            'v5': ['体积', '容积', '容量'],
        }
        self.iqr_list = ['v5=v3*v4', 'v3=3.14*v1*v1', 'v2=2*v1']

    def get_iqr_list(self, entity_list):
        return self.key_to_val(entity_list, self.iqr_list, self.key_dict,
                               self.val_dict)


# 多边形类
class parral_area_model(IQRClassify):
    def __init__(self):
        self.key_dict = {
            'v1': '底',
            'v2': '高',
            'v3': '面积',
        }
        self.val_dict = {'v1': ['底'], 'v2': ['高'], 'v3': ['面积']}
        self.iqr_list = ['v3=v1*v2']

    def get_iqr_list(self, entity_list):
        return self.key_to_val(entity_list, self.iqr_list, self.key_dict,
                               self.val_dict)


class triangle_area_model(IQRClassify):
    def __init__(self):
        self.key_dict = {
            'v1': '底',
            'v2': '高',
            'v3': '面积',
        }
        self.val_dict = {'v1': ['底'], 'v2': ['高'], 'v3': ['面积']}
        self.iqr_list = ['v3=v1*v2*0.5']

    def get_iqr_list(self, entity_list):
        return self.key_to_val(entity_list, self.iqr_list, self.key_dict,
                               self.val_dict)


class trapez_area_model(IQRClassify):
    def __init__(self):
        self.key_dict = {
            'v1': '上底',
            'v2': '下底',
            'v3': '高',
            'v4': '面积',
        }
        self.val_dict = {'v1': ['上底'], 'v2': ['下底'], 'v3': ['高'], 'v4': ['面积']}
        self.iqr_list = ['v4=(v1+v2)*v3*0.5']

    def get_iqr_list(self, entity_list):
        return self.key_to_val(entity_list, self.iqr_list, self.key_dict,
                               self.val_dict)


# 多分量类
class part2_model(IQRClassify):
    def __init__(self):
        self.key_dict = {'v1': '第一部分', 'v2': '第二部分', 'v3': '一共'}
        self.val_dict = {
            'v1': ['第一'],
            'v2': ['第二'],
            'v3': [
                '共', '全', '总数'
            ]  # '一共', '总共', '共有', '共修', '共看', '共长', '共用', '共重', '共需', '共付', '共运',
        }
        # iqr的关系初始化由key组成
        self.iqr_list = ['v3=v1+v2']

    def get_iqr_list(self, entity_list):
        temp_iqr_list = copy.deepcopy(self.iqr_list)
        for index, iqr_str in enumerate(temp_iqr_list):
            for key in self.key_dict:
                if key in iqr_str:
                    # 查找这个key的候选值是否被eqr实体包含（in str)
                    for value in self.val_dict[key]:
                        flag = 0
                        # 遍历所有实体
                        for entity in entity_list:
                            # 匹配，将关系中的key替换为eqr中的实体。 匹配成功后flag变为-1，跳出两层循环
                            if str(value) in str(entity):
                                temp_iqr_list[index] = temp_iqr_list[
                                    index].replace(key, entity)
                                flag = -1
                                break
                            # 当前value遍历到最后一个eqr实体也未匹配成功，将关系中的key替换为iqr中的实体
                            elif entity == entity_list[
                                    -1] and value == self.val_dict[key][-1]:
                                temp_iqr_list[index] = temp_iqr_list[
                                    index].replace(key, self.key_dict[key])
                            else:
                                pass
                        if flag == -1:
                            break
        return temp_iqr_list


class part2_remain_model(IQRClassify):
    def __init__(self):
        self.key_dict = {'v1': '第一部分', 'v2': '第二部分', 'v4': '剩下', 'v5': '一共'}
        self.val_dict = {
            'v1': ['第一'],
            'v2': ['第二'],
            'v4': ['剩', '余', '还有', '还要'],
            'v5': [
                '共', '全', '总数'
            ]  # '一共', '总共', '共有', '共修', '共看', '共长', '共用', '共重', '共需', '共付', '共运',
        }
        # iqr的关系初始化由key组成
        self.iqr_list = ['v3=v1+v2', 'v5=v1+v2+v4']

    def get_iqr_list(self, entity_list):
        temp_iqr_list = copy.deepcopy(self.iqr_list)
        for index, iqr_str in enumerate(temp_iqr_list):
            for key in self.key_dict:
                if key in iqr_str:
                    # 查找这个key的候选值是否被eqr实体包含（in str)
                    for value in self.val_dict[key]:
                        flag = 0
                        # 遍历所有实体
                        for entity in entity_list:
                            # 匹配，将关系中的key替换为eqr中的实体。 匹配成功后flag变为-1，跳出两层循环
                            if str(value) in str(entity):
                                temp_iqr_list[index] = temp_iqr_list[
                                    index].replace(key, entity)
                                flag = -1
                                break
                            # Value遍历到最后一个也未有任何实体与其匹配成功，将关系中的key替换为iqr中的实体
                            elif entity == entity_list[
                                    -1] and value == self.val_dict[key][-1]:
                                temp_iqr_list[index] = temp_iqr_list[
                                    index].replace(key, self.key_dict[key])
                            else:
                                pass
                        if flag == -1:
                            break
        return temp_iqr_list


class part3_model(IQRClassify):
    def __init__(self):
        self.key_dict = {'v1': '第一部分', 'v2': '第二部分', 'v3': '第三部分', 'v4': '一共'}
        self.val_dict = {
            'v1': ['第一'],
            'v2': ['第二'],
            'v3': ['第三'],
            'v4': [
                '共', '全', '总数'
            ]  # '一共', '总共', '共有', '共修', '共看', '共长', '共用', '共重', '共需', '共付', '共运',
        }
        # iqr的关系初始化由key组成
        self.iqr_list = ['v4=v1+v2+v3']

    def get_iqr_list(self, entity_list):
        temp_iqr_list = copy.deepcopy(self.iqr_list)
        for index, iqr_str in enumerate(temp_iqr_list):
            for key in self.key_dict:
                if key in iqr_str:
                    # 查找这个key的候选值是否被eqr实体包含（in str)
                    for value in self.val_dict[key]:
                        flag = 0
                        # 遍历所有实体
                        for entity in entity_list:
                            # 匹配，将关系中的key替换为eqr中的实体。 匹配成功后flag变为-1，跳出两层循环
                            if str(value) in str(entity):
                                temp_iqr_list[index] = temp_iqr_list[
                                    index].replace(key, entity)
                                flag = -1
                                break
                            # Value遍历到最后一个也未有任何实体与其匹配成功，将关系中的key替换为iqr中的实体
                            elif entity == entity_list[
                                    -1] and value == self.val_dict[key][-1]:
                                temp_iqr_list[index] = temp_iqr_list[
                                    index].replace(key, self.key_dict[key])
                            else:
                                pass
                        if flag == -1:
                            break
        return temp_iqr_list


class scale2real_distance_model(IQRClassify):
    def __init__(self):
        self.key_dict = {
            'v1': '比例尺',
            'v2': '距离',
            'v3': '实际距离',
        }
        self.val_dict = {
            'v1': ['比例尺'],
            'v2': ['图上距离', '距离', '图上', '长度', '长', '画'],
            'v3': ['实际距离', '实际路程', '实际长度', '实际长', '实际']
        }
        self.iqr_list = ['v3=v1*v2']

    def get_iqr_list(self, entity_list):
        return self.key_to_val(entity_list, self.iqr_list, self.key_dict,
                               self.val_dict)


class scale2real_area_model(IQRClassify):
    def __init__(self):
        self.key_dict = {
            'v1': '比例尺',
            'v2': '面积',
            'v3': '实际面积',
        }
        self.val_dict = {
            'v1': ['比例尺'],
            'v2': ['图上面积', '面积'],
            'v3': ['实际面积', '实际占地面积']
        }
        self.iqr_list = ['v3=v1*v1*v2']

    def get_iqr_list(self, entity_list):
        return self.key_to_val(entity_list, self.iqr_list, self.key_dict,
                               self.val_dict)


class IQRAcquire(object):
    model_bag = {
        0: distance_model(),
        1: bridge_cross_model(),
        2: interest_model(),
        3: partwhole_model(),
        4: rec_circum_model(),
        5: rec_area_model(),
        6: cuboid_area_model(),
        7: cuboid_vol_model(),
        8: square_circum_model(),
        9: square_are_model(),
        10: cube_area_model(),
        11: cube_vol_model(),
        12: circular_circum_model(),
        13: circular_area_model(),
        14: cylinder_area_model(),
        15: cylinder_vol_model(),
        16: parral_area_model(),
        17: triangle_area_model(),
        18: trapez_area_model(),
        19: part2_model(),
        20: part2_remain_model(),
        21: part3_model(),
        22: scale2real_distance_model(),
        23: scale2real_area_model()
    }

    @staticmethod
    def list2d_list1d(list2d):
        list1d = []
        for row in list2d:
            for line in row:
                list1d.append(line)
        return list1d

    @staticmethod
    def get_iqr_result(iqr_class_list, entity_list):
        # 22种关系, 对应22种模型
        iqr_relation_list = []
        for iqr_num in iqr_class_list:
            if iqr_num in IQRAcquire.model_bag.keys():
                r = IQRAcquire.model_bag[iqr_num].get_iqr_list(entity_list)
                iqr_relation_list.extend(r)
        return iqr_relation_list
