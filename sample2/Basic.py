import numpy as np

"""
    定义朴素贝叶斯的基类（基本框架）
    
"""


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

    """
        模型训练过程
    """

    # 定义计算先验概率的函数，Laplace = 1，默认使用Laplace smoothing
    def get_prior_probability(self, laplace=1):
        return [(_c_num + laplace) / len(self._y) + laplace * len(self._cat_counter) for _c_num in self._cat_counter]

    #  定义具有普适性的训练函数
    def fit(self, x=None, y=None, sample_weight=None, laplace=1):
        # 如果有传入x，y，则使用传入的x，y初始化模型
        if x is not None and y is not None:
            self.feed_data(x, y, sample_weight)
        # 调用核心算法得到决策函数
        self._func = self._fit(laplace)

    # 留下抽象核心算法让子类定义
    def _fit(self, laplace):
        pass

    """
        模型预测与评估过程
    """

    # 定义预测单一样本的函数
    # get_raw_result控制函数是输出预测类别还是相对应的后验概率
    # get_raw_result=False输出类别， get_raw_result=True输出后验概率
    def predict_one(self, x, get_raw_result=False):
        # 进行预测前要把新的输入的数据数值化
        # 如果输入的是Numpy数组，要先将他转换成python数组来加快在数值化上的操作，不然直接对数组进行拷贝
        if isinstance(x, np.ndarray):
            x = x.tolist()
        else:
            x = x[:]
        # 调用相对应的方法进行数值化，该_transfer_x方法随具体模型的不同而不同
        x = self._transfer_x(x)
        m_arg, m_probability = 0, 0
        # 遍别各类型，找到最大的类别使得后验概率最大化
        for i in range(len(self._cat_counter)):
            p = self._func(x, i)
            if p > m_probability:
                m_arg, m_probability = i, p
        if not get_raw_result:
            return self.label_dic[m_arg]
        else:
            return m_probability

    # 留下抽象的数值化方法让子类重写
    def _transfer_x(self, x):
        pass

    # 定义预测多样本的函数，本质是不断调用上面定义的predict_one函数
    def predict(self, x, get_raw_result=False):
        return np.array([self.predict_one(_x, get_raw_result) for _x in x])

    # 定义对新数据进行评估的方法，这里暂时以简单的输出准确率作为演示
    def evaluate(self, x, y):
        y_pred = self.predict(x)
        print("ACC: {:12.6}%".format(100 * np.sum(y_pred == y) / len(y)))


