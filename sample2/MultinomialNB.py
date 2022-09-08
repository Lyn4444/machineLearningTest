from typing import List

from Basic import *

"""
    MultinomialNB实现
    使用Numpy的bincount方法计数，速度更快，但是有缺点：
        只能处理非负整数的数组
        向量中最大值即返回数组的长度
"""


class MultinomialNB(NaiveBayes):

    # 定义预处理数据的方法，对输入数据进行数值化加快训练过程，同时视具体情况不同，预处理的实现不同
    def feed_data(self, x, y, sample_weight=None):
        # 根据情况对输入的x进行转置
        if isinstance(x, list):
            features = map(list, zip(*x))
        else:
            features = x.T
        # 利用集合获取各个维度的特征与类别种类
        # 将所有特征从0开始数值化，方便使用bincount
        # 将数值化过程中的转换关系记录成字典来对新数据进行判断
        features = [set(feat) for feat in features]
        feat_dics = [{_l: i for i, _l in enumerate(feats)} for feats in features]
        label_dic = {_l: i for i, _l in enumerate(set(y))}
        # 使用转换字典更新训练集
        x = np.array([[feat_dics[i][_l] for i, _l in enumerate(sample)] for sample in feat_dics])
        y = np.array([label_dic[_y] for _y in y])
        # 使用Numpy中的bincount方法，获取各类别的数据的个数
        cat_counter = np.bincount(y)
        # 记录各维度特征的取值个数
        n_possibilities = [len(feats) for feats in features]
        # 获取各类别数组的下标
        labels = [y == value for value in range(len(cat_counter))]
        # 利用下标获取记录按类别分开后的输入数据的数组
        labelled_x = [x[ci].T for ci in labels]
        # 更新模型的各个属性
        self._x, self._y = x, y
        self._labelled_x, self._label_zip = labelled_x, list(zip(labels, labelled_x))
        (self._cat_counter, self._feat_dics, self._n_possibilities) = (cat_counter, feat_dics, n_possibilities)
        self.label_dic = {i: _l for _l, i in label_dic.items()}
        # 调用处理样本权重的函数，以更新记录条件概率的数组
        self.feed_sample_weight(sample_weight)

    # 定义处理样本权重的函数
    def feed_sample_weight(self, sample_weight=None):
        self._con_counter = []
        # 利用Numpy的bincount方法获取带权重的条件概率额极大似然估计
        for dim, _p in enumerate(self._n_possibilities):
            if sample_weight is None:
                self._con_counter.append([np.bincount(_x[dim], minlength=_p) for _x in self._labelled_x])
            else:
                self._con_counter.append(
                    [np.bincount(_x[dim], weights=sample_weight[label] / sample_weight[label].mean(), minlength=_p)
                     for label, _x in self._label_zip])

    """
        预处理进行了大量工作，使得核心函数变为调用与整合数据预处理时记录下来的信息的过程
    """
    def _fit(self, laplace):
        n_dim = len(self._n_possibilities)
        n_category = len(self._cat_counter)
        p_category = self.get_prior_probability(laplace)
