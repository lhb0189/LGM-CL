import torch
import torch.nn as nn

class GNN_model(nn.Module):
    def __init__(self,encoder1,encoder2,input_dim,output_dim,dropout):
        #5个参数,encoder1为GAT模型, encdoer2为GT模型, input_dim为输入的向量的维度 ,output_dim为projection head后得到的最终2个向量表征的维度, Dropout为projection head中的MLP层的droopout率
        super().__init__()
        self.GAT=encoder1
        self.GT=encoder2
        self.output_dim=output_dim
        self.dropout=dropout
        #projection head
        self.projection_head=nn.Sequential(
            nn.Linear(input_dim,512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256,output_dim)
        )
    def forward(self,atom_features,bond_features,mask,adjacency_matrix,atom_index):#atom_index:[2,分子数量]
        #atom_features:原子初始化后得到的特征,bond_features:化学键初始化得到的特征，用于AttentiveFP, mask:掩码矩阵, adjacency_matrix: 邻接矩阵
        #进行池化操作和Projection Head
        x1=self.GAT(atom_features,bond_features)#x1:[原子数量,特征维度]
        x2, Transformer_features = self.GT(atom_features, mask, adjacency_matrix)#x2:[原子数量,特征维度]
        embedding_GAT=[]
        embedding_GT=[]
        for index_number in atom_index:
            initial_index,atom_number=index_number[0], index_number[1]
            single_molecule_GAT=x1[initial_index:initial_index+atom_number]
            single_molecule_GT=x2[initial_index:initial_index+atom_number]
            pool_GAT=single_molecule_GAT.sum(dim=0)
            average_GAT=pool_GAT/atom_number
            pool_GT=single_molecule_GT.sum(dim=0)
            average_GT=pool_GT/atom_number
            embedding_GAT.append(average_GAT)
            embedding_GT.append(average_GT)
        embedding_stack_GAT=torch.stack(embedding_GAT,dim=0)#[分子数量,分子维度]
        embedding_stack_GT=torch.stack(embedding_GT,dim=0)
        out1=self.projection_head(embedding_stack_GAT)
        out2=self.projection_head(embedding_stack_GT)
        return x1,x2,out1,out2,Transformer_features
