from rdkit import Chem
import torch
import pandas as pd
import numpy as np
def one_of_k_encoding(x, allowable_set):#Onehot编码,不在allowable_set中就报错
    l=[]
    for i in allowable_set:
        if x==i:
            l+=[1]
        else:
            l+=[0]
    return l
def one_of_k_encoding_unk(x, allowable_set):#Onehot编码,不在allowable_set中的就给x弄进allowable_set中去
    l=[]
    if x not in allowable_set:
        x=allowable_set[-1]
    for i in allowable_set:
        if x==i:
            l+=[1]
        else:
            l+=[0]
    return l
#原子初始特征: 1.原子的序号,16维. 2.原子的度 6维. 3.原子的形式电荷(取的具体数值) 1维. 4.自由基电子 1维. 5.杂化轨道 6维. 6.是否在芳香键中 1维 7.连接的氢原子的个数 5维. 8.是否在环中 1维. 9.环的大小 4维. 10.手性标签 4维 11.原子质量(具体数值) 1维.
#12.隐式化合价 7维. 13.是否是酸性基团、氢键供体、氢键受体、碱性基团,一共4维.
def get_atom_features(atom,mol):
    hydrogen_donor = Chem.MolFromSmarts("[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]")  # 氢键供体
    hydrogen_acceptor = Chem.MolFromSmarts(
        "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),"
        "n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]")  # 氢键受体
    acidic = Chem.MolFromSmarts("[$([C,S](=[O,S,P])-[O;H1,-1])]")  # 酸性基团
    basic = Chem.MolFromSmarts(
        "[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);"
        "!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]")  # 碱性基团
    hydrogen_donor_match = sum(mol.GetSubstructMatches(hydrogen_donor), ())  # 氢键供体匹配的基团
    hydrogen_acceptor_match = sum(mol.GetSubstructMatches(hydrogen_acceptor), ())  # 氢键受体匹配的基团
    acidic_match = sum(mol.GetSubstructMatches(acidic), ())#酸性基团匹配
    basic_match = sum(mol.GetSubstructMatches(basic), ())#碱性基团
    atom_idx=atom.GetIdx()#原子的Id索引
    ring_info = mol.GetRingInfo()
    attributes=[]
    attributes+=one_of_k_encoding_unk(atom.GetSymbol(),['B','C','N','O','F','Si','P','S','Cl','As','Se','Br','Te','I','At','other'])#原子的序号,16维
    attributes+=one_of_k_encoding(atom.GetDegree(),[0, 1, 2, 3, 4, 5])#原子的度 6维
    attributes+=[atom.GetFormalCharge(),atom.GetNumRadicalElectrons()]#原子的形式电荷和自由基电子 2维
    attributes+=one_of_k_encoding_unk(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2,'other'])#杂化轨道 6维
    attributes+=[1 if atom.GetIsAromatic() else 0]#是否在芳香键中 1维
    attributes+=one_of_k_encoding_unk(atom.GetTotalNumHs(),[0, 1, 2, 3, 4])#氢原子个数  5维
    attributes+=[1 if atom.IsInRing() else 0]#是否在环中 1维
    attributes=(attributes+[1 if ring_info.IsAtomInRingOfSize(atom_idx, 3) else 0]+[1 if ring_info.IsAtomInRingOfSize(atom_idx, 4) else 0]
                +[1 if ring_info.IsAtomInRingOfSize(atom_idx, 5) else 0]+[1 if ring_info.IsAtomInRingOfSize(atom_idx, 6) else 0])#环的大小 4维
    attributes+=one_of_k_encoding(int(atom.GetChiralTag()),[0, 1, 2, 3])#手性标签，4维
    attributes+=[atom.GetMass()*0.01]#原子质量 1维
    #42维
    attributes+=one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6])#隐式化合价7维
    attributes=(attributes+[1 if atom_idx in hydrogen_acceptor_match else 0] +
                [1 if atom_idx in hydrogen_donor_match else 0] + [1 if atom_idx in acidic_match else 0] + [1 if atom_idx in basic_match else 0])#4维
    return attributes#分子属性总共57维
bond_fdim=15
def get_bond_features(bond):
    if bond is None:
        fbond=[1]+[0]*(bond_fdim-1)
    else:
        bt=bond.GetBondType()
        fbond= [
                0,  # 第一维表示键是否存在 1维
                1 if bt == Chem.rdchem.BondType.SINGLE else 0,#单键 1维
                1 if bt == Chem.rdchem.BondType.DOUBLE else 0,#双键 1维
                1 if bt == Chem.rdchem.BondType.TRIPLE else 0,#三键 1维
                1 if bt == Chem.rdchem.BondType.AROMATIC else 0,#芳香键 1维
                1 if bt == Chem.rdchem.BondDir.ENDUPRIGHT else 0,#二维平面上的双键方向,这里朝着右上 1维
                1 if bt == Chem.rdchem.BondDir.ENDDOWNRIGHT else 0,#二维平面上的双键方向,这里朝着右下 1维
                1 if bond.GetIsConjugated()  else 0,#是否是共轭键 1维
                1 if bond.IsInRing()  else 0 #是否在环中 1维
            ]
        fbond += one_of_k_encoding(int(bond.GetStereo()), list(range(6)))#6维，立体化学信息
    return fbond#键特征总共15维
def num_atom_features():
    mol=Chem.MolFromSmiles("CC")
    alist=mol.GetAtoms()
    a=alist[0]
    return len(get_atom_features(a,mol))#57维
def num_bond_features():#返回bond特征维度
    simple_mol = Chem.MolFromSmiles('CC')
    Chem.SanitizeMol(simple_mol)#对分子对象Mol进行化学合理性检查和标准化
    return len(get_bond_features(simple_mol.GetBonds()[0]))#15维
class GraphOne:#针对单个Graph的构建,输入为单行数据
    def __init__(self,smiles):
        mol=Chem.MolFromSmiles(smiles)
        if mol is None:#若RDKIT解析不了,直接删掉,在论文中可以说明一下RDKIT解析不了的分子直接删除.
            print(smiles)
        self.smiles=smiles
        self.atom_features=[]#存储原子特征.
        self.bond_features={}#化学键特征
        self.atom_numbers=mol.GetNumAtoms()#原子数量
        self.bond_numbers=mol.GetNumBonds()#化学键数量
        self.edge_index=[]#[2,num_edges]形式,在RDKIT解析后对应的节点序号,第一个为起点,第二个为终点.
        for i,atom in enumerate(mol.GetAtoms()):
            self.atom_features.append(get_atom_features(atom,mol))
        self.atom_features=[self.atom_features[i] for i in range(self.atom_numbers)]#按照解析后的序号,将特征重新排列
        self.adjacency_matrix=torch.eye(self.atom_numbers)#邻接矩阵
        for i,bond in enumerate(mol.GetBonds()):
            begin_atom,end_atom=bond.GetBeginAtom().GetIdx(),bond.GetEndAtom().GetIdx()#获取起点和终点对应的序号
            self.edge_index += [(begin_atom, end_atom), (end_atom, begin_atom)]#当成有向图
            self.bond_features[(begin_atom, end_atom)] = get_bond_features(bond)
            self.bond_features[(end_atom, begin_atom)] = get_bond_features(bond)
            self.adjacency_matrix[begin_atom, end_atom] = 1
            self.adjacency_matrix[end_atom, begin_atom] = 1
class GraphBatch:
    def __init__(self,graphs):
        smile_list=[]
        for graph in graphs:
            smile_list.append(graph)
        self.smile_list=smile_list
        self.smile_num=len(self.smile_list)
        self.atom_features_dim=num_atom_features()
        self.bond_features_dim=num_bond_features()
        self.atom_no=0
        self.atom_index=[]
        self.bond_features={}
        atom_features=[]
        for graph in graphs:
            atom_features.extend(graph.atom_features)
            self.atom_index.append((self.atom_no,graph.atom_numbers))
            for index,value in graph.bond_features.items():
                begin=index[0]+self.atom_no
                end=index[1]+self.atom_no
                self.bond_features[(begin,end)]=torch.FloatTensor(value)
                self.bond_features[(end,begin)]=torch.FloatTensor(value)
            self.atom_no = self.atom_no + graph.atom_numbers
        self.atom_features=torch.FloatTensor(atom_features)#转化为torch和类型
        self.adjacency_matrix=torch.eye(self.atom_no)
        for i in self.bond_features.keys():
            begin_atom,end_atom=i[0],i[1]
            self.adjacency_matrix[begin_atom,end_atom]=1
            self.adjacency_matrix[end_atom,begin_atom]=1

    def get_feature(self):  # 原子特征[total_atom_number,atom_embeddings_dim]
        return self.atom_features, self.atom_index, self.bond_features
    def get_adjacency_matrix(self):
        return self.adjacency_matrix
def create_graph(smiles):
    graphs=[]
    for one in smiles:
        graph=GraphOne(one)
        graphs.append(graph)
    return GraphBatch(graphs)

