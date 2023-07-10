from scipy.spatial import KDTree
from dadapy.data import Data
import warnings
import numpy as np
import skdim
from treelib import Tree
import dataPreprocessing
from sklearn.datasets import make_blobs
from sklearn import preprocessing
import math
warnings.filterwarnings('ignore')


def find_max_k(data,Dthr = 23.928,omega=1,max_k=100):
    """
    自适应分配每个点对应的k近邻
    :param data: 输入数据
    :param Dthr: 判断阈值。由于密度差符合卡方分布，Dthr越大判断结果越符合点i所拥有的最大近邻。Dtrh对应置信度，因此该条件不是自由变量
    :param omega: 用于放大数据间的差异。如果数据集坐标在小范围内则需扩大omega。在数据得到适当的放缩后，omega不会影响对于结果的判断，因此不是自由变量
    :param max_k: 影响收敛速率，但不会影响判断结果，因此不是自由变量
    :return: 返回每个点对应的k近邻 密度rou 预测误差error 亮度light 近邻距离矩阵distances 近邻矩阵indices
    """
    # 计算点的内在维度
    data_twoNN = Data(data)
    data_twoNN.compute_distances(max_k)
    data_twoNN.compute_id_2NN()
    danco = skdim.id.FisherS().fit(data)
    id=danco.dimension_
    # id=5
    # 构建KD树
    tree = KDTree(data)
    distances, indices = tree.query(data, k=min(max_k, data.shape[0]))
    dissimilarity=np.power(distances,id)
    V_matrix = np.diff(dissimilarity, axis=1)*omega

    # 初始化每个点对应的k, 密度，预测误差，亮度
    list_k = [-1] * len(data)
    list_rou = [-1] * len(data)
    list_error = [-1] * len(data)
    list_light = [-1] * len(data)
    for i in range(len(data)):# 遍历每一个点
        Dk_flag = False # 判断是否有点满足密度差条件
        now_k = 0 # 当前近邻数
        while True:
            now_k += 1
            # 计算now_k
            j = indices[i][now_k]# 找到当前点的第k个近邻； indices[i][0]为i点自身
            # 计算Dk 和 Dk1
            Dk = -2 * now_k * (np.log(np.sum(V_matrix[i][:now_k])) + np.log(np.sum(V_matrix[j][:now_k])) - 2 * np.log(
                np.sum(V_matrix[i][:now_k]) + np.sum(V_matrix[j][:now_k])) + np.log(4))
            Dk1 = -2 * now_k * (np.log(np.sum(V_matrix[i][:now_k+1])) + np.log(np.sum(V_matrix[j][:now_k+1])) - 2 * np.log(
                np.sum(V_matrix[i][:now_k+1]) + np.sum(V_matrix[j][:now_k+1])) + np.log(4))
            if Dk<Dthr:# 判断是否达到过阈值
                Dk_flag=True
            if ((Dk1 >= Dthr) and (Dk_flag==True)) or (now_k==min(max_k-1,data.shape[0])) == True: #如果【达到阈值】 或者 【遍历到最大近邻数】 则停止遍历
                list_k[i] = now_k
                list_rou[i] = now_k / np.sum(V_matrix[i][:now_k])
                list_error[i] = np.sqrt((4 * now_k + 2) / ((now_k - 1) * now_k))
                list_light[i] = np.log(list_rou[i]) / list_error[i]
                break



    return list_k, list_rou, list_error, list_light, distances, indices


def find_merge_center(merge_list, visited_list, merge_array, now_position):
    """
    通过递推的方式将需要连续更新的簇放到同一个list里。比如簇0 和 1需要合并，1 和 3需要合并。则 0 1 3需要合并为同一个簇
    :param FG: frequency gap 频率差
    :param merge_list: 连续合并list。merge_list[0]=[1,2,3] 表示1 2 3号簇都需要合并为簇0
    :param visited_list: 记录已经被访问过的簇，减少计算量
    :param merge_array: 融合矩阵
    :param now_position: 新访问位置
    :return: 连续合并列表list，已被访问的列表visited_list
    """
    if visited_list[now_position]==1: #如果当前位置已经被访问则跳过
        return merge_list,visited_list
    merge_list.append(now_position) # merge_list添加当前簇
    visited_list[now_position]=1 # 记录当前簇已被访问
    res=np.where(merge_array[now_position]!=0)[0] # 获取与当前簇有同频萤火虫的簇序号
    # max_synchronization_firefly=max(merge_array[now_position])# 获得最多的同频萤火虫
    if len(res)<=0:# 如果没有同频萤火虫则返回
        return merge_list,visited_list
    else:# 如果有同频萤火虫
        for i in res: # 遍历每个可能同频簇的序号
            if merge_array[now_position][i]>0:
                merge_list.append(i) #merge_list添加同频簇
                merge_list,visited_list=find_merge_center(merge_list, visited_list, merge_array, i) # 通过同频簇递推去找更多的同频簇
        return list(set(merge_list)),visited_list


def Model(data, Dthr = 23.928, omega=1,Mthr=10,max_k=100):
    """
        自适应分配每个点对应的k近邻
        :param data: 输入数据
        :param Dthr: 判断阈值。由于密度差符合卡方分布，Dthr越大判断结果越符合点i所拥有的最大近邻。Dtrh对应置信度，因此该条件不是自由变量
        :param omega: 用于放大数据间的差异。如果数据集坐标在小范围内则需扩大omega。在数据得到适当的放缩后，omega不会影响对于结果的判断，因此不是自由变量
        :param max_k: 影响收敛速率，但不会影响判断结果，因此不是自由变量
        :param FG: frequent gap 指发光频率差。用于判断簇合并，如果发光频率差过大则簇不应该合并，此变量为模型唯一的自由变量，对模型起约束作用
        :return: 返回每个点对应标签label
        """
    # 1. 腐草生萤： 初试化每一个点
    list_k, list_rou, list_error, list_light, distances, indices = find_max_k(data,Dthr,omega,max_k) # 获得每个点对应的k近邻 密度rou 预测误差error 亮度light 近邻距离矩阵distances 近邻矩阵indices
    leading_firefly = []# 簇心萤火虫，引导萤火虫

    # 2. 萤翅流光： 根据亮度进行簇传播
    label = [-1] * data.shape[0] # 初试化标签
    label_index = 0 # 使用的标签序号
    sorted_list_light_id = sorted(range(len(list_light)), key=lambda k: list_light[k], reverse=True) # 对萤火虫亮度进行降序排序 O(NlogN)
    navigation_list=[-1]*data.shape[0]
    path_distance=[0]*data.shape[0]
    path_distance_num=[1]*data.shape[0]
    cluster_list=[]
    for i in range(0, len(sorted_list_light_id)): # 根据亮度从高到低遍历每一个点i
        leading_firefly_flag = True # 引导萤火虫标签
        higher_light_point_as_neighbor_order=-1
        for j in range(1, list_k[sorted_list_light_id[i]]): # 遍历该点的近邻j
            higher_light_point = indices[sorted_list_light_id[i]][j] #记当前近邻为亮度更高点
            if list_light[higher_light_point] > list_light[sorted_list_light_id[i]]: # 判断近邻点亮度是否高于当前点，如果是，则当前点不是引导萤火虫
                leading_firefly_flag = False
                higher_light_point_as_neighbor_order=j
                break
        if leading_firefly_flag == True: # 如果当前点是引导萤火虫
            cluster_list.append([sorted_list_light_id[i]])
            leading_firefly.append(sorted_list_light_id[i]) # 引导萤火虫list添加当前点
            label[sorted_list_light_id[i]] = label_index # 引导萤火虫自带一种新的label
            label_index+=1
        else: # 如果当前点不是引导萤火虫
            navigation_list[sorted_list_light_id[i]]= higher_light_point# 把亮度比自己高的萤火虫作为父结点，将当前萤火虫作为新结点加入到对应的同频树里
            label[sorted_list_light_id[i]] = label[higher_light_point] # 设定当前点的label为亮度更高点的label
            path_distance[sorted_list_light_id[i]]=path_distance[higher_light_point]+distances[sorted_list_light_id[i]][higher_light_point_as_neighbor_order]
            path_distance_num[sorted_list_light_id[i]]=path_distance_num[higher_light_point]+1
            cluster_list[label[higher_light_point]].append(sorted_list_light_id[i])

    # 3. 囊萤映树： 根据同频树进行簇合并

    leaves_list=list(set(sorted_list_light_id)-set(navigation_list))
    merge_array = np.zeros([len(leading_firefly), len(leading_firefly)])  # 融合矩阵。单元格内对应数值越大，表示同频萤火虫越多，说明行和列对应的簇越可能合并
    merge_array1 = np.zeros([len(leading_firefly), len(leading_firefly)])  # 融合矩阵。单元格内对应数值越大，表示同频萤火虫越多，说明行和列对应的簇越可能合并
    for i in range(0,len(leaves_list)):
        now_point=leaves_list[i]
        for j in range(0,list_k[now_point]):
            target_point=indices[now_point][j]
            if label[now_point]!=label[target_point]:
                root_leaves_distance=(path_distance[now_point]/path_distance_num[now_point]+path_distance[j]/path_distance_num[target_point])/2
                two_points_distance=distances[now_point][j]
                merge_array[label[now_point]][label[target_point]] += (two_points_distance-root_leaves_distance)/abs(list_light[now_point]-list_light[target_point])
                merge_array1[label[now_point]][label[target_point]] += 1
                merge_array[label[target_point]][label[now_point]] += (two_points_distance-root_leaves_distance)/abs(list_light[now_point]-list_light[target_point])
                merge_array1[label[target_point]][label[now_point]] += 1

    while True:
        max_value = np.max(merge_array)  # 求出矩阵最大值对应位置
        if max_value < Mthr:
            break
        max_loc = np.where(merge_array == max_value)
        row_index = max_loc[0][0]
        col_index = max_loc[1][0]
        if list_light[leading_firefly[row_index]] >= list_light[leading_firefly[col_index]]:
            cluster_list[row_index]=list(set(cluster_list[row_index]).union(set(cluster_list[col_index])))
            leading_firefly.remove(leading_firefly[col_index])
            cluster_list.remove(cluster_list[col_index])
            merge_array = np.delete(merge_array, col_index, axis=0)
            merge_array = np.delete(merge_array, col_index, axis=1)
        else:
            cluster_list[col_index] = list(set(cluster_list[col_index]).union(set(cluster_list[row_index])))
            leading_firefly.remove(leading_firefly[row_index])
            cluster_list.remove(cluster_list[row_index])
            merge_array = np.delete(merge_array, row_index, axis=0)
            merge_array = np.delete(merge_array, row_index, axis=1)




    for i in range(0, len(cluster_list)):  # 更新簇，完成合并
        for j in range(0, len(cluster_list[i])):
            label[cluster_list[i][j]] =i
    return label
