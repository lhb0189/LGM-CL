import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
class MoleData():
    def __init__(self,line,text):
        self.smile=line[0]
        self.label=[float(x) if x!='' else None for x in line[1:]]
        self.text=text
    def task_num(self):
        return len(self.label)
    def change_label(self,label):
        self.label=label

class MoleDataSet(Dataset):#MoleDataset中,初始化中的参数类得是MoleData类
    def __init__(self,data):
        self.data=data
        self.scaler=None #这里确定有没有对回归中的标签进行标准化操作
    def __len__(self):
        return len(self.data)
    def __getitem__(self, key):
        return self.data[key]
    def smile(self):
        smile_list=[]
        for one in self.data:
            smile_list.append(one.smile)
        return smile_list
    def mol(self):
        mol_list=[]
        for one in self.data:
            mol_list.append(one.smile)
        return mol_list
    def label(self):
        label_list=[]
        for one in self.data:
            label_list.append(one.label)
        return label_list
    def texts(self):
        text_list=[]
        for one in self.data:
            text_list.append(one.text)
        return text_list
    def task_num(self):
        if len(self.data)>0:
            return self.data[0].task_num()
    def random_data(self,seed):#随机打乱数据
        random.seed(seed)
        random.shuffle(self.data)
    def change_label(self, label):  # 这里是用在多任务回归上,要将lable标准化，因此要更改标签
        assert len(self.data) == len(label)
        for i in range(len(label)):
            self.data[i].change_label(label[i])  # 批量修改整个数据集中所有数据点的标签,例如数据预处理或者标准化
