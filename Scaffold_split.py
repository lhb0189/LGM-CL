from collections import defaultdict
import logging
import random
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from data import MoleDataSet
#这里写的是骨架拆分
def generate_scaffold(mol,include_chirality=False):#从单个分子中生成Murcko骨架,Murcko骨架表示分子的核心结构框架，去除了侧链和官能团,保留了环系统和连接环的键
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol#这里避免传入的是smiles字符串
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)#include_chirality表示是否在骨架中包含手性信息
    return scaffold#最后返回得到的Murcko骨架
def scaffold_to_smiles(mol,use_indices=False):#这里输入的是一组分子 按照Murcko骨架进行分组.返回一个defaultdict,其中键是骨架smiles,值是对应的分子集合
    scaffolds = defaultdict(set)#初始化字典 值为集合,用来存储分子
    for i, one in enumerate(mol):
        scaffold = generate_scaffold(one)
        if use_indices:#use_indices=True，则存储分子的索引i
            scaffolds[scaffold].add(i)
        else:#use_indices=False 存储分子本身
            scaffolds[scaffold].add(one)
    return scaffolds#返回字典,键为骨架smiles,值为对应的分子,这样就得到一个集合,每个骨架所对应的分子集合.
def scaffold_split(data,size,seed,log):#基于Murcko骨架对分子数据集进行划分,分为训练集,验证集,测试集,确保相同骨架的分子不会同时出现在不同的集合中.
    #data是MoleDataset,一堆MoleData组成的,size为[train_ratio,val_ratio,test_ratio],seed为随机种子,log为日志记录器
    assert sum(size) == 1
    # Split
    train_size, val_size, test_size = size[0] * len(data), size[1] * len(data), size[2] * len(data)#划分比例
    train, val, test = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0
    # Map from scaffold to index in the data
    scaffold_to_indices = scaffold_to_smiles(data.mol(), use_indices=True)#传入的是索引
    index_sets = list(scaffold_to_indices.values())#骨架对应的索引集合列表
    big_index_sets = []#大骨架,如果某个骨架的分子数超过验证集或测试集大小的一半，则归为big_index_sets（避免大骨架被拆散）
    small_index_sets = []#其余的就给小骨架
    for index_set in index_sets:
        if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
            big_index_sets.append(index_set)
        else:
            small_index_sets.append(index_set)
    random.seed(seed)#随机种子
    random.shuffle(big_index_sets)
    random.shuffle(small_index_sets)#打乱大骨架和小骨架的顺序，但最终index_sets是大骨架在前
    index_sets = big_index_sets + small_index_sets#这里大骨架的列表在前面,优先处理
    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set#记录放入训练集的分子的索引
            train_scaffold_count += 1#记录训练集中不同的骨架的数目
        elif len(val) + len(index_set) <= val_size:
            val += index_set
            val_scaffold_count += 1
        else:
            test += index_set
            test_scaffold_count += 1
    log.debug(f'Total scaffolds = {len(scaffold_to_indices):,} | '
              f'train scaffolds = {train_scaffold_count:,} | '
              f'val scaffolds = {val_scaffold_count:,} | '
              f'test scaffolds = {test_scaffold_count:,}')
    # Map from indices to data
    train = [data[i] for i in train]#得到三个集合的的MoleData
    val = [data[i] for i in val]#
    test = [data[i] for i in test]
    return MoleDataSet(train), MoleDataSet(val), MoleDataSet(test)#再进行封装