import rdkit
import random
from torch.utils.data.dataset import Dataset
from rdkit import Chem
#这里针对无监督学习,具体在于无标签的情况下
class MoleData():#针对CSV文件,第一行是smiles字符串
    def __init__(self,line):
        self.smile=line[0]
        self.mol=Chem.MolFromSmiles(self.smile)
class MoleDataSet(Dataset):#MoleDataset中,初始化中的参数类得是MoleData类
    def __init__(self,data):
        self.data=data
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
    def random_data(self,seed):#随机打乱数据
        random.seed(seed)
        random.shuffle(self.data)