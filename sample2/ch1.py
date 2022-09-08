import numpy as np


# 定义朴素贝叶斯的基类
class NaiveBayes:
    # 初始化结构

    def __init__(self):
        # 训练集变量
        self._x = self._y = None
        # 核心数组，存储实际使用的条件概率的相关信息   模型核心（决策函数），能够根据输入的x,y输出相对应的后验概率
        self._data = self._func = None
        # 各个维度特征取值个数数组
        self._n_possibilities = None
        # 按类别分开后输入数据的数组   类别相关信息的数组
        self._labelled_x = self._label_zip = None
        # 第i类数据的个数（category）   数据条件概率的原始极大似然估计（conditional）
        self._cat_counter = self._con_counter = None
        # 数值化类别时的转换关系      数值化各维度特征时的转换关系
        self.label_dic = self._feat_dics = None

    # 判断实例来避免定义大量property
    def __getitem__(self, item):
        if isinstance(item, self):
            return getattr(self, "" + item)

    """
        让模型支持输入样本权重，使得模型能够应用于提升方法中
    """

    # 留下抽象方法让子类定义，并且 tar_idx参数与self._tar_idx参数的意义一致
    def feed_data(self, x, y, sample_weight=None):
        pass

    # 留下抽象方法让子类定义， sample_weight参数代表样本权重
    def feed_sample_weight(self, sample_weight=None):
        pass

    # 定义计算先验概率的函数，Laplace = 1，默认使用Laplace smoothing
    def get_prior_probability(self, laplace=1):
        return [(_c_num + laplace) / len(self._y) + laplace * len(self._cat_counter) for _c_num in self._cat_counter]

