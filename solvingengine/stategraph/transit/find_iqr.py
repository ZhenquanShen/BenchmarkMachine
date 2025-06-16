import numpy as np

clsId2name = {
    "0": "distance_list",
    "1": "bridge_cross_list",
    "2": "interest_list",
    "3": "part_whole_list",
    "4": "rec_circum_list",
    "5": "rec_area_list",
    "6": "cuboid_area_list",
    "7": "cuboid_vol_list",
    "8": "square_circum_list",
    "9": "square_are_list",
    "10": "cube_area_list",
    "11": "cube_vol_list",
    "12": "circular_circum_list",
    "13": "circular_area_list",
    "14": "cylinder_area_list",
    "15": "cylinder_vol_list",
    "16": "parral_area_list",
    "17": "triangle_area_list",
    "18": "trapez_area_list",
    "19": "part2_list",
    "20": "part2_remain_list",
    "21": "part3_list",
    "22": "scale2real_distance_list",
    "23": "scale2real_area_list",
    "24": "times_list",
    "25": "fraction_list",
    "26": "percentage_list"
}
iqrId2name = {
    "0": ["distance", "路程"],
    "2": ["interest", "利息"],
    "8": ["square_perimeter", "正方形周长"],
    "4": ["rectangular_perimeter", "长方形周长"],
    "12": ["circular_circumference", "圆形周长"],
    "9": ["square_area", "正方形面积"],
    "5": ["rectangular_area", "长方形面积"],
    "13": ["circular_area", "圆形面积"],
    "18": ["trapezoidal_area", "梯形面积"],
    "17": ["triangle_area", "三角形面积"],
    "16": ["area_of_parallelogram", "平行四边形面积"],
    "6": ["cuboid_surface Area", "长方体表面积"],
    "14": ["cylinder_surface Area", "圆柱体表面积"],
    "11": ["cube_volume", "正方体体积"],
    "7": ["cuboid_volume", "长方体体积"],
    # ["cone_volume","圆锥体体积"],
    "15": ["cylinder_volume", "圆柱体体积"],
    "3": ["part_whole", "部分整体"],
    "19": ["part_whole", "部分整体"],
    "20": ["part_whole", "部分整体"],
    "21": ["part_whole", "部分整体"],
    "22": ["scale2real_distance", "比例尺距离"],
    "23": ["scale2real_area", "比例尺面积"],
    # ["sailing","行船"],
    "1": ["train_crossing_bridge", "火车过桥"]
}


class ProbClassify:
    clsId2name = clsId2name
    iqrId2name = iqrId2name

    def __init__(self):

        self.distance_list = 0
        self.bridge_cross_list = 0
        self.interest_list = 0
        self.part_whole_list = 0

        self.rec_circum_list = 0
        self.rec_area_list = 0
        self.cuboid_area_list = 0
        self.cuboid_vol_list = 0

        self.square_circum_list = 0
        self.square_are_list = 0
        self.cube_area_list = 0
        self.cube_vol_list = 0

        self.circular_circum_list = 0
        self.circular_area_list = 0
        self.cylinder_area_list = 0
        self.cylinder_vol_list = 0

        self.parral_area_list = 0
        self.triangle_area_list = 0
        self.trapez_area_list = 0

        self.part2_list = 0
        self.part2_remain_list = 0
        self.part3_list = 0

        self.scale2real_distance_list = 0
        self.scale2real_area_list = 0

        self.times_list = 0
        self.fraction_list = 0
        self.percentage_list = 0

        self.all_iqr = []

    def find_prob_class(self, text):
        if '每天' in text or '每小时' in text or '每分钟' in text or '每秒' in text or '速度' in text or '每时' in text:
            if '米' in text or '千米' in text or '公里' in text or 'm' in text:
                if '相遇' not in text and '相对' not in text and '相向' not in text and '相反' not in text and '相背' not in text \
                        and '返回' not in text and '半径' not in text and '直径' not in text \
                        and '正方' not in text and '长方' not in text and '平方' not in text and '立方' not in text\
                        and '工程' not in text and '修' not in text and '挖' not in text and '铺' not in text:
                    if '车长' in text or '车全长' in text or '车的长' in text or '车身长' in text or '大桥长' in text:
                        self.bridge_cross_list = 1
                        self.distance_list = 0
                    else:
                        self.bridge_cross_list = 0
                        self.distance_list = 1
                else:
                    self.bridge_cross_list = 0
                    self.distance_list = 0
            else:
                self.bridge_cross_list = 0
                self.distance_list = 0
        elif '用' in text or '花' in text or '要' in text:
            if '时' in text or '分' in text or '秒' in text:
                if '米' in text or '千米' in text or '公里' in text or 'm' in text:
                    if '相遇' not in text and '相对' not in text and '相向' not in text and '相反' not in text and '相背' not in text \
                            and '返回' not in text and '半径' not in text and '直径' not in text \
                            and '正方' not in text and '长方' not in text and '平方' not in text and '立方' not in text \
                            and '工程' not in text and '修' not in text and '挖' not in text and '铺' not in text:
                        if '车长' in text or '车全长' in text or '车的长' in text or '车身长' in text or '大桥长' in text:
                            self.bridge_cross_list = 1
                            self.distance_list = 0
                        else:
                            self.bridge_cross_list = 0
                            self.distance_list = 1
                    else:
                        self.bridge_cross_list = 0
                        self.distance_list = 0
                else:
                    self.bridge_cross_list = 0
                    self.distance_list = 0
            else:
                self.bridge_cross_list = 0
                self.distance_list = 0
        else:
            self.bridge_cross_list = 0
            self.distance_list = 0

        if '银行' in text or '存' in text or '贷款' in text or '国债' in text or '债券' in text:
            if '利率' in text or '利息率' in text:
                self.interest_list = 1
            else:
                self.interest_list = 0
        else:
            self.interest_list = 0

        # 一般部分整体类
        if ('共' in text or '总' in text or '全' in text) and '第一' not in text:
            if '已' in text or '剩' in text or '还' in text:
                self.part_whole_list = 1
            else:
                self.part_whole_list = 0
        else:
            self.part_whole_list = 0

        # 长方类
        if '长方形' in text and '长' in text and '宽' in text:
            if '周长' in text or '圈' in text or '围成' in text:
                self.rec_circum_list = 1
            else:
                self.rec_circum_list = 0
        else:
            self.rec_circum_list = 0

        if '长方形' in text and '长' in text and '宽' in text:
            if '面积' in text or '平方' in text or '公顷' in text:
                self.rec_area_list = 1
            else:
                self.rec_area_list = 0
        else:
            self.rec_area_list = 0

        if '长' in text and '宽' in text and ('高' in text or '深' in text):
            if '表面积' in text or '平方' in text:
                self.cuboid_area_list = 1
            else:
                self.cuboid_area_list = 0
        else:
            self.cuboid_area_list = 0

        if '长' in text and '宽' in text and ('高' in text or '深' in text):
            if '体积' in text or '容积' in text or '立方' in text or '升' in text:
                self.cuboid_vol_list = 1
            else:
                self.cuboid_vol_list = 0
        else:
            self.cuboid_vol_list = 0

        # 正方类
        if '正方形' in text or '方砖' in text:
            if '周长' in text or '圈' in text or '围成' in text:
                self.square_circum_list = 1
            else:
                self.square_circum_list = 0
        else:
            self.square_circum_list = 0

        if ('正方形' in text or '方砖' in text) and '边长' in text:
            if '面积' in text or '平方' in text or '公顷' in text:
                self.square_are_list = 1
            else:
                self.square_are_list = 0
        else:
            self.square_are_list = 0

        if '正方体' in text and '棱长' in text:
            if '表面积' in text or '平方' in text:
                self.cube_area_list = 1
            else:
                self.cube_area_list = 0
        else:
            self.cube_area_list = 0

        if '正方体' in text and '棱长' in text:
            if '体积' in text or '容积' in text or '立方' in text or '升' in text:
                self.cube_vol_list = 1
            else:
                self.cube_vol_list = 0
        else:
            self.cube_vol_list = 0

        # 圆类
        if '圆形' in text and ('直径' in text or '半径' in text):
            if '周长' in text or '圈' in text or '围成' in text:
                self.circular_circum_list = 1
            else:
                self.circular_circum_list = 0
        else:
            self.circular_circum_list = 0

        if '圆形' in text and ('直径' in text or '半径' in text):
            if '面积' in text or '平方' in text or '公顷' in text:
                self.circular_area_list = 1
            else:
                self.circular_area_list = 0
        else:
            self.circular_area_list = 0

        if '圆柱' in text and ('直径' in text or '半径' in text or '高' in text):
            if '面积' in text or '平方' in text:
                self.cylinder_area_list = 1
            else:
                self.cylinder_area_list = 0
        else:
            self.cylinder_area_list = 0

        if '圆柱' in text and ('直径' in text or '半径' in text or '高' in text):
            if '体积' in text or '容积' in text or '立方' in text or '升' in text:
                self.cylinder_vol_list = 1
            else:
                self.cylinder_vol_list = 0
        else:
            self.cylinder_vol_list = 0

        # 多边形类
        if '平行四边形' in text:
            if '面积' in text or '平方' in text or '公顷' in text:
                self.parral_area_list = 1
            else:
                self.parral_area_list = 0
        else:
            self.parral_area_list = 0

        if '三角形' in text:
            if '面积' in text or '平方' in text or '公顷' in text:
                self.triangle_area_list = 1
            else:
                self.triangle_area_list = 0
        else:
            self.triangle_area_list = 0

        if '梯形' in text:
            if '面积' in text or '平方' in text or '公顷' in text:
                self.trapez_area_list = 1
            else:
                self.trapez_area_list = 0
        else:
            self.trapez_area_list = 0

        # 部分整体二分量问题
        if '第一' in text and '第二' in text and '第三' not in text:
            if '剩' in text or '余下' in text or '还要' in text or '还有' in text:
                self.part2_remain_list = 1
                self.part2_list = 0
            else:
                self.part2_remain_list = 0
                if '共' in text:
                    self.part2_list = 1
                else:
                    self.part2_list = 0
        else:
            self.part2_list = 0
            self.part2_remain_list = 0

        # 部分整体三份量问题
        if '第一' in text and '第二' in text and '第三' in text:
            if '剩' in text or '余下' in text or '还要' in text or '还有' in text or '共' in text or '全' in text or '总数' in text:
                self.part3_list = 1
            else:
                self.part3_list = 0
        else:
            self.part3_list = 0

        # 比例尺问题
        if '比例尺' in text and '实际' in text:
            if '面积' in text:
                self.scale2real_area_list = 1
                self.scale2real_distance_list = 0
            else:
                self.scale2real_distance_list = 1
                self.scale2real_area_list = 0
        else:
            self.scale2real_area_list = 0
            self.scale2real_distance_list = 0

        if ('是' in text or '比' in text) and '倍' in text:
            self.times_list = 1
        else:
            self.times_list = 0

        if '几分之几' in text:
            self.fraction_list = 1
        else:
            self.fraction_list = 0

        if '百分之几' in text or '合格率' in text:
            self.percentage_list = 1
        else:
            self.percentage_list = 0

    def find_iqr_class(self, text):
        self.find_prob_class(text)
        self.all_iqr = np.array([self.distance_list,
                                 self.bridge_cross_list,
                                 self.interest_list,
                                 self.part_whole_list,
                                 self.rec_circum_list,
                                 self.rec_area_list,
                                 self.cuboid_area_list,
                                 self.cuboid_vol_list,
                                 self.square_circum_list,
                                 self.square_are_list,
                                 self.cube_area_list,
                                 self.cube_vol_list,
                                 self.circular_circum_list,
                                 self.circular_area_list,
                                 self.cylinder_area_list,
                                 self.cylinder_vol_list,
                                 self.parral_area_list,
                                 self.triangle_area_list,
                                 self.trapez_area_list,
                                 self.part2_list,
                                 self.part2_remain_list,
                                 self.part3_list,
                                 self.scale2real_distance_list,
                                 self.scale2real_area_list,
                                 self.times_list,
                                 self.fraction_list,
                                 self.percentage_list])
        total_num_result = list(np.where(self.all_iqr == 1)[0])
        return total_num_result
