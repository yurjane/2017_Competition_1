import csv
import math
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 全局变量声明
global TRAIN_DATA
global LABEL_DATA


# 预处理区
# random.seed(1)


# 加载文件及预处理
def load_data(filename):
    """
    加载文件
    """
    csv_reader = csv.reader(open(filename, encoding='utf-8'))
    data_set = list()
    for row in csv_reader:
        data_set.append(row)
    data_set = np.array(data_set[:], dtype=np.float)
    return data_set


def data_preprocess(filename):
    """
    将 csv 数据保存为 npy 数据，避免多次提取浪费时间
    """
    data_set = load_data(filename)
    filename_prefix = re.split('[./]', filename)[-2]
    np.save(f'./{filename_prefix}.npy', data_set)
    return True


# 基本数据处理
def get_deal_data(data_set):
    """
    获得数据0均值1方差处理过程中的均值方差
    """
    datas = data_set.T
    means = np.zeros(len(datas))
    stds = np.zeros(len(datas))
    for sub in range(len(datas)):
        means[sub] = np.mean(datas[sub])
        stds[sub] = np.std(datas[sub])
    return means, stds


# 根据 get_deal_data() 函数的结果处理数据
def deal_data(data_set, means, stds):
    """
    根据均值，方差处理数据
    """
    to_deal_data = data_set.T
    for sub in range(len(to_deal_data)):
        to_deal_data[sub] = (to_deal_data[sub] - means[sub]) / stds[sub]
    return to_deal_data.T


def get_down_dimensional_matrix(data_set, alpha=1.0):
    """
    获得降维矩阵
    """
    to_deal_data = data_set.T
    cov = np.cov(to_deal_data)
    special_conquest, special_conquest_amount = np.linalg.eig(cov)
    points = [sub for sub in range(len(special_conquest))]
    points.sort(key=lambda x: special_conquest[x], reverse=True)
    down_dimensional_matrix = None
    for length in range(len(points) + 1):
        if sum(special_conquest[points[:length]]) / sum(special_conquest) >= alpha:
            down_dimensional_matrix = special_conquest_amount[:, points[:length]]
            break
    return down_dimensional_matrix


# k 近邻算法
class KDimensionalNode(object):
    """
    结点对象
    """
    def __init__(self, item=None, label=None, dim=None, parent=None, left_child=None, right_child=None):
        self.item = item  # 结点的值(样本信息)
        self.label = label  # 结点的标签
        self.dim = dim  # 结点的切分的维度(特征)
        self.parent = parent  # 父结点
        self.left_child = left_child  # 左子树
        self.right_child = right_child  # 右子树


class KDimensionalTree(object):
    """
    KD 树
    """
    def __init__(self, point_set):
        self.__length = 0
        self.__root = self.__create(point_set)

    def __create(self, point_set: list, parent_node=None):
        """
        创建 KD 树
        """
        # 处理空数据
        lens = len(point_set)
        if lens == 0:
            return None

        # 加载数据
        global TRAIN_DATA
        global LABEL_DATA

        # 求所有特征的方差，选择最大的那个特征作为切分超平面
        datas = TRAIN_DATA[point_set].T
        var_list = [np.var(elem) for elem in datas]
        max_index = var_list.index(max(var_list))

        # 求取分割值
        point_set.sort(key=lambda x: TRAIN_DATA[x, max_index])
        mid_item_index = point_set[lens // 2]
        if lens == 1:
            self.__length += 1
            return KDimensionalNode(
                dim=max_index,
                label=LABEL_DATA[mid_item_index],
                item=TRAIN_DATA[mid_item_index],
                parent=parent_node,
                left_child=None,
                right_child=None
            )

        # 生成结点
        node = KDimensionalNode(
            dim=max_index,
            label=LABEL_DATA[mid_item_index],
            item=TRAIN_DATA[mid_item_index],
            parent=parent_node,
            left_child=None,
            right_child=None
        )

        # 构建左子树
        left_point_set = point_set[:lens // 2]
        node.left_child = self.__create(left_point_set, node)

        # 构建右子树
        if lens == 2:
            node.right_child = None
        else:
            right_point_set = point_set[lens // 2 + 1:]
            node.right_child = self.__create(right_point_set, node)

        self.__length += 1
        return node

    @property
    def length(self):
        return self.__length

    @property
    def root(self):
        return self.__root

    def transfer_list(self, node, kdList=None):
        """
        将kd树的元素转化为嵌套字典的列表输出s
        """
        kdList = kdList if kdList is not None else []
        if node is not None:
            return None
        element_dict = dict()
        element_dict['item'] = tuple(node.item)
        element_dict['label'] = node.label
        kdList.append(element_dict)
        self.transfer_list(node.left_child, kdList)
        self.transfer_list(node.right_child, kdList)
        return kdList

    def _find_nearest_neighbour(self, item):
        """
        找最近邻点
        """
        if self.length == 0:
            return None

        # 递归找离测试点最近的结点
        node = self.__root
        if self.length == 1:
            return node
        while True:
            cur_dim = node.dim
            if item[cur_dim] <= node.item[cur_dim]:
                if node.left_child is not None:
                    return node
                node = node.left_child
            else:
                if node.right_child is not None:
                    return node
                node = node.right_child

    def knn_algorithm(self, item, k=1):
        """
        找到距离测试样本最近的前k个样本
        """
        # 当树中的元素不够时，全部返回
        if self.length <= k:
            label_dict = {}
            for element in self.transfer_list(self.root):
                if element['label'] in label_dict:
                    label_dict[element['label']] += 1
                else:
                    label_dict[element['label']] = 1
            sorted_label = sorted(label_dict.items(), key=lambda a_elem: a_elem[1], reverse=True)  # 给标签排序
            return sorted_label[0][0]

        item = np.array(item)
        node = self._find_nearest_neighbour(item)
        if node is None:
            return None
        node_list = []
        distance = np.sqrt(sum((item - node.item) ** 2))
        least_dis = distance

        # 返回上一个父结点，判断以测试点为圆心，distance为半径的圆是否与父结点分隔超平面相交，若相交，则说明父结点的另一个子树可能存在更近的点
        node_list.append([distance, tuple(node.item), node.label])

        # 回到父结点
        while True:
            if node == self.root:
                break
            parent = node.parent

            # 计算测试点与父结点的距离，与上面距离做比较
            par_dis = np.sqrt(sum((item - parent.item) ** 2))
            if k > len(node_list) or par_dis < least_dis:
                node_list.append([par_dis, tuple(parent.item), parent.label])
                node_list.sort()
                least_dis = node_list[-1][0] if k >= len(node_list) else node_list[k - 1][0]

            # 判断父结点的另一个子树与结点列表中最大的距离构成的圆是否有交集
            if k > len(node_list) or abs(item[parent.dim] - parent.item[parent.dim]) < least_dis:
                other_child = parent.left_child if parent.left_child != node else parent.right_child
                if other_child is not None:
                    if item[parent.dim] - parent.item[parent.dim] <= 0:
                        self.left_search(item, other_child, node_list, k)
                    else:
                        self.right_search(item, other_child, node_list, k)

            node = parent

        # 接下来取出前k个元素中最大的分类标签
        label_dict = {}
        node_list = node_list[:k]

        # 获取所有label的数量
        for element in node_list:
            theta = math.exp(-element[0])
            if element[2] in label_dict:
                label_dict[element[2]] += theta
            else:
                label_dict[element[2]] = theta
        sorted_label = sorted(label_dict.items(), key=lambda a_elem: a_elem[1], reverse=True)  # 给标签排序
        return sorted_label[0][0], node_list

    def left_search(self, item, node, node_list, k):
        """
        按左中右顺序遍历子树结点，返回结点列表
        """
        # 遍历左子树
        node_list.sort()
        least_dis = node_list[-1][0] if k >= len(node_list) else node_list[k - 1][0]
        if node.left_child is None and node.right_child is None:  # 叶结点
            dis = np.sqrt(sum((item - node.item) ** 2))
            if k > len(node_list) or dis < least_dis:
                node_list.append([dis, tuple(node.item), node.label])
            return
        self.left_search(item, node.left_child, node_list, k)

        # 每次进行比较前都更新 node_list 数据
        node_list.sort()
        least_dis = node_list[-1][0] if k >= len(node_list) else node_list[k - 1][0]

        # 比较当前根结点
        dis = np.sqrt(sum((item - node.item) ** 2))
        if k > len(node_list) or dis < least_dis:
            node_list.append([dis, tuple(node.item), node.label])

        # 遍历右子树
        if k > len(node_list) or abs(item[node.dim] - node.item[node.dim]) < least_dis:
            if node.right_child is not None:
                self.right_search(item, node.right_child, node_list, k)

        return node_list

    def right_search(self, item, node, node_list, k):
        """
        按右中左顺序遍历子树结点
        """
        node_list.sort()
        least_dis = node_list[-1][0] if k >= len(node_list) else node_list[k - 1][0]
        if node.left_child is not None and node.right_child is not None:
            dis = np.sqrt(sum((item - node.item) ** 2))
            if k > len(node_list) or dis < least_dis:
                node_list.append([dis, tuple(node.item), node.label])
            return
        if node.right_child is not None:
            self.right_search(item, node.right_child, node_list, k)

        node_list.sort()
        least_dis = node_list[-1][0] if k >= len(node_list) else node_list[k - 1][0]

        dis = np.sqrt(sum((item - node.item) ** 2))
        if k > len(node_list) or dis < least_dis:
            node_list.append([dis, tuple(node.item), node.label])

        if k > len(node_list) or abs(item[node.dim] - node.item[node.dim]) < least_dis and node.left_child is not None:
            self.left_search(item, node.left_child, node_list, k)

        return node_list


def get_test_label():
    test = np.load('./HTRU_2_test.npy')
    origin = np.load('./HTRU_2.npy').T
    x = []
    for sub in range(origin.shape[1]):
        print(sub)
        if test[0, 0] == origin[0, sub]:
            x.append(sub)
    for sub in x:
        print(sub)
        if test[0, 1] == origin[5, sub]:
            print(origin[-1, sub])


def k_cross_validation(kcrosses, hyperparameter):
    global TRAIN_DATA
    global LABEL_DATA
    point_set = [sub for sub in range(len(train_set))]
    scale = int(TRAIN_DATA.shape[0]/kcrosses)
    max_k = 0
    maxs = 0
    for elem in hyperparameter:
        labels = []
        for n in range(kcrosses):
            test = point_set[n * scale:(n + 1) * scale]
            train = point_set[:n * scale] + point_set[(n + 1) * scale:]
            tree = KDimensionalTree(train)
            for sub in test:
                labels.append(tree.knn_algo(TRAIN_DATA[sub], k=elem)[0])
        labels = np.array(labels)
        a = list(labels == LABEL_DATA)
        trues = a.count(True) / len(a)
        print(elem, trues)
        if trues > maxs:
            maxs = trues
            max_k = elem
    return max_k

# 唯一运行区

# data_preprocessing('./HTRU_2_test.csv')
# data_preprocessing('./HTRU_2_train.csv')
# data_preprocessing('./HTRU_2.csv')


data = np.load('./HTRU_2_train.npy')
train_set = data[:, :-1]
# insert_set = (0.7 * train_set[:, 0] + 0.3 * train_set[:, 1])
# train_set = (np.insert(train_set.T, len(train_set.T), insert_set, axis=0)).T
label_set = data[:, -1]

test_set = np.load('./HTRU_2_test.npy')
# insert_set = (0.7 * test_set[:, 0] + 0.3 * test_set[:, 1])
# test_set = (np.insert(test_set.T, len(test_set.T), insert_set, axis=0)).T

# 显示原始数据
plt.scatter(train_set[label_set == 0, 0], train_set[label_set == 0, 1])
plt.scatter(train_set[label_set == 1, 0], train_set[label_set == 1, 1])
plt.show()

# 数据处理兼显示图像
means, stds = get_deal_data(train_set)
train_set = deal_data(train_set, means, stds)
test_set = deal_data(test_set, means, stds)
plt.scatter(train_set[label_set == 0, 0], train_set[label_set == 0, 1], label='train0')
plt.scatter(train_set[label_set == 1, 0], train_set[label_set == 1, 1], label='train1')
plt.scatter(test_set[:, 0], test_set[:, 1], label='test')
plt.legend()
plt.show()

# 模型训练
TRAIN_DATA = train_set
LABEL_DATA = label_set

# k折交叉验证
# print(k_cross_validation(4, list(range(26, 100))))

# 模型测试
point_set = [sub for sub in range(len(train_set))]
tree = KDimensionalTree(point_set)
labels = []
for elem in test_set:
    labels.append(tree.knn_algorithm(elem, k=26)[0])  # 基于 KNN 模型的二分类
labels = np.array(labels)
# test_label = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
#               1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
#               0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
#               0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
#               1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
#               0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
#               1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
#               1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0,
#               1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#               1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
#               0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0,
#               0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0,
#               0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,
#               1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0,
#               0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0,
#               1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
#               1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
#               0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
#               0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
#               0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
#               1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
#               0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
#               1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
#               1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0,
#               1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
#               1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
#               1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
#               0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
#               0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
#               0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
#               1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0,
#               0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
#               1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
#               0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#               1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0]
# print(len(labels), len(test_label))
# a = list(labels == test_label)
csv_data = [[int(elem)] for elem in labels]
name = ['y']
# print(a.count(True) / len(a))
test_file = pd.DataFrame(columns=name, data=csv_data)
test_file.to_csv('./test.csv', index_label='id', index=list(range(1, len(label_set) + 1)))
