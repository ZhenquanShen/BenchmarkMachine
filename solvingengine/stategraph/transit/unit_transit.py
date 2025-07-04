'''
#!/usr/bin/env: Python3
# -*- encoding: utf-8-*-
Description: 
Author: Xiaopan LYU
Date: 2023-01-01 14:02:27
LastEditTime: 2023-01-03 16:29:25
LastEditors: Xiaopan LYU
'''
UNIT_TYPES = [{
    '千米': 1,
    '公里': 1,
    '里': 2,
    '米': 1000,
    '分米': 10000,
    '厘米': 100000,
    '毫米': 1000000,
    'default': '米'
}, {
    '吨': 1,
    '千克': 1000,
    '公斤': 1000,
    '斤': 500,
    '克': 1000000,
    'default': '千克'
}, {
    '立方米': 1,
    '立方分米': 1000,
    '升': 1000,
    '毫升': 1000000,
    '立方厘米': 1000000,
    '立方毫米': 1000000000,
    'default': '立方米'
}, {
    '小时': 1,'时': 1,
    '分钟': 60,'分': 60,'秒钟': 3600,
    '秒': 3600,
    'default': '小时'
}, {
    "万亿元": 1,
    "千亿元": 10,
    "百亿元": 100,
    "十亿元": 1000,
    "亿元": 10000,
    "千万元": 100000,
    "百万元": 1000000,
    "十万元": 10000000,
    "万元": 100000000,
    "千元": 1000000000,
    "百元": 10000000000,
    "十元": 100000000000,
    "元": 1000000000000,
    "角": 10000000000000,
    "分": 100000000000000,
    'default': '元'
}, {
    '千瓦时': 1,
    '度': 1000,
    '千焦': 1000,
    '焦': 500,
    'default': '度'
}, {
    '平方公里': 1,
    '平方千米': 1,
    '顷': 15,
    '公顷': 100,
    '亩': 1500,
    '公亩': 1000,
    '平方米': 1000000,
    '平方分米': 100000000,
    '平方厘米': 10000000000,
    '平方毫米': 1000000000000,
    'default': '平方米'
},
{'1':1,'盆':1,'面':1,'盏':1,'个':1,'棵':1,'株':1,'default': '1'}
]