import pandas as pd
import numpy as np
from rdkit import Chem
from graph import GraphOne
#这里得将没有RDKIT解析不了的分子去掉 然后存在没有邻居的特征原子也得去掉.
data_file="Pretrain_Datasets\\Zinc15.csv"
df=pd.read_csv(data_file)
smiles=df["smiles"].tolist()
delete_column=[]
delete_smile=[]
number=0#这里要drop的行在csv文件中对应为+2后的数字.
for smile in smiles:
    if Chem.MolFromSmiles(smile) is None:
        delete_smile.append(smile)
        delete_column.append(number)
    else:
        mol=Chem.MolFromSmiles(smile)
        g=GraphOne(smile)
        adjacency_matrix=g.adjacency_matrix
        adjacency_matrix=np.array(adjacency_matrix)
        atom_number=mol.GetNumAtoms()
        for i in range(atom_number):
            sign=0
            for j in range(atom_number):
                if i!=j:
                    if adjacency_matrix[i,j]==1:
                        sign=1
            if sign==0:
                delete_smile.append(smile)
                delete_column.append(number)
    number=number+1
    print(number)
print(len(delete_column))
print(delete_column)
print(len(delete_smile))
df=df.drop(df.index[delete_column]).reset_index(drop=True)
df.to_csv("Process_Datasets\\Zinc15.csv",index=False)