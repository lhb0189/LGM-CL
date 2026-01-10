import math,copy
import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from graph import create_graph,GraphOne,GraphBatch
import torch.nn.functional as F

class GATlayerV1(nn.Module):#in_features:输入原子特征的维度, bond_features_dim:输入化学键特征的维度, output_features:想要得到的输出特征的维度,
    #dropout_gnn_ratio:GNN的dropout率 leaky_alpha:leaky_relu激活函数的alpha值, elu_alpha:ELU激活函数的alpha值
    def __init__(self,in_features,bond_features_dim,output_features,dropout_gnn_ratio,leaky_alpha,elu_alpha):
        super(GATlayerV1,self).__init__()
        self.bond_features_dim=bond_features_dim
        self.dropout_gnn=nn.Dropout(p=dropout_gnn_ratio)
        self.in_features=in_features
        self.out_features=output_features
        self.atom_fc1=nn.Linear(self.in_features,self.out_features).to("cuda")
        self.atom_fc2=nn.Linear(self.in_features+self.bond_features_dim,self.out_features).to("cuda")
        self.W=nn.Linear(2*self.out_features,1).to("cuda")
        self.leaky_alpha=leaky_alpha
        self.elu_alpha=elu_alpha
        self.attend=nn.Linear(output_features,output_features).to("cuda")
        self.GRUcell=nn.GRUCell(self.out_features,self.out_features).to("cuda")
    def forward(self,atom_features,bond_features):#atom_features:[原子数量,特征维度],bond_features:[(起点的原子索引,终点的原子索引),特征维度]
        N=atom_features.shape[0]#原子的个数
        atom_features=atom_features.to("cuda")
        new_atom_features=F.leaky_relu(self.atom_fc1(atom_features),negative_slope=self.leaky_alpha)#[原子数量,in_feautres]->[原子数量,out_features],经过一个Linear层,激活函数为LeakyReLU
        neighbor_features_transform = self.attend(self.dropout_gnn(new_atom_features))#[原子数量,out_features]->[]
        new_list=[]
        for i in range(N):
            transform=torch.zeros(self.out_features)
            transform=transform.to("cuda")
            atom_neighbor_features = []
            index=[]
            for keys,values in bond_features.items():
                if keys[1]==i:
                    index.append(keys[0])
                    neighbor_atom_features=torch.cat([atom_features[keys[0]],values],dim=0)
                    atom_neighbor_features.append(neighbor_atom_features)
            l=[]
            if len(atom_neighbor_features)==0:
                new_list.append(new_atom_features[i])
            else:
                stacked_tensor=torch.stack(atom_neighbor_features,dim=0)
                new_neighbor=F.leaky_relu(self.atom_fc2(stacked_tensor),negative_slope=self.leaky_alpha)
                for j in range(len(index)):
                    final_neighbor=torch.cat([new_atom_features[i],new_neighbor[j]],dim=0)
                    l.append(final_neighbor)
                final_embedding_1=torch.stack(l,dim=0)
                final_embedding_2=F.leaky_relu(self.W(self.dropout_gnn(final_embedding_1)),negative_slope=self.leaky_alpha)
                score=F.softmax(final_embedding_2,dim=0)
                score=score.to("cuda")
                neighbor_features_transform=neighbor_features_transform.to("cuda")
                for j in range(len(index)):
                    transform+=score[j]*neighbor_features_transform[index[j]]
                context=F.elu(transform,alpha=self.elu_alpha)
                new_list.append(context)
        output_embedding=torch.stack(new_list,dim=0)#[total_num_atoms,out_features]
        final_output_embedding=self.GRUcell(output_embedding,new_atom_features)#经过一个GRU
        return final_output_embedding,new_atom_features
class GATlayerV2(nn.Module):
    def __init__(self,output_features,dropout_gnn_ratio,leaky_alpha,elu_alpha):
        super(GATlayerV2,self).__init__()
        self.out_features=output_features
        self.dropout_gnn=nn.Dropout(p=dropout_gnn_ratio)
        self.leaky_alpha=leaky_alpha
        self.elu_alpha=elu_alpha
        self.atom_fc1=nn.Linear(output_features*2,1)
        self.atom_fc2=nn.Linear(output_features,output_features)
        self.GRUcell=nn.GRUCell(self.out_features,self.out_features)
    def forward(self,atom_features,bond_features):
        N=atom_features.shape[0]#原子个数
        neighbor_features_transform =self.atom_fc2(self.dropout_gnn(atom_features))
        new_list = []
        for i in range(N):
            transform=torch.zeros(self.out_features)
            transform=transform.to("cuda")
            index=[]
            for keys,values in bond_features.items():
                if keys[1]==i:
                    index.append(keys[0])
            l=[]
            if len(index) == 0:
                new_list.append(atom_features[i])
            else:
                for j in range(len(index)):
                    final_neighbor=torch.cat([atom_features[i],atom_features[index[j]]],dim=0)
                    l.append(final_neighbor)
                final_embedding_1=torch.stack(l,dim=0)
                final_embedding_2=F.leaky_relu(self.atom_fc1(self.dropout_gnn(final_embedding_1)),negative_slope=self.leaky_alpha)
                score=F.softmax(final_embedding_2,dim=0)
                for j in range(len(index)):
                    transform+=score[j]*neighbor_features_transform[index[j]]
                context=F.elu(transform,alpha=self.elu_alpha)
                new_list.append(context)
        output_embedding=torch.stack(new_list,dim=0)
        final_output=self.GRUcell(output_embedding,atom_features)
        return final_output
def clones(module, N):
    #Produce N identical layers
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
def get_mask(graph_batch_infor):
    #graph为graph_batch类
    length=graph_batch_infor.atom_no
    same_group=torch.full((length,length),-1e9)#[total_atom_number,total_atom_number]
    atom_index=graph_batch_infor.atom_index
    for index in atom_index:
        start,atom_number=index[0],index[1]
        same_group[start:start+atom_number,start:start+atom_number]=torch.zeros((atom_number,atom_number))
    return same_group
def graph_attention(query,key,value,mask,adjacency_matrix,lambdas,trainable_lambda,dropout=None):
    #q,k,v:[h,total_atom_number,d_k]
    d_k=query.size(-1)
    eps=1e-6
    scores=torch.matmul(query,key.transpose(-2,-1)) /math.sqrt(d_k)#[total_atom_number,total_atom_number]
    scores_shape=scores.shape
    mask_shape=mask.shape
    scores=scores+mask
    p_attn=F.softmax(scores,dim=-1).to("cuda")
    adj_matrix = adjacency_matrix / (adjacency_matrix.sum(dim=-1,keepdim=True)+ eps)#对邻接矩阵归一化(加上eps防止除以0)
    adj_matrix=adj_matrix.unsqueeze(0).repeat(query.shape[0],1,1)
    p_adj=adj_matrix.to("cuda")
    value=value.to("cuda")
    if trainable_lambda:
        softmax_attention, softmax_adjacency = lambdas.cuda()
        p_weighted = softmax_attention * p_attn + softmax_adjacency * p_adj
    else:
        lambda_attention, lambda_adjacency = lambdas
        p_weighted = lambda_attention * p_attn + lambda_adjacency * p_adj
    if dropout is not None:
        p_weighted = dropout(p_weighted)
    atom_features=torch.matmul(p_weighted,value).to("cuda")
    #atom_features:[total_num_atoms,d_k],p_weighted:[total_num_atoms,total_num_atoms]
    # p_attn:[total_num_atoms,total_num_atoms]
    return atom_features,p_weighted,p_attn
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout,lambda_attention,trainable_lambda):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h=h
        self.trainable_lambda=trainable_lambda
        if trainable_lambda:
            lambda_distance=1. -lambda_attention
            lambdas_tensor=torch.tensor([lambda_attention,lambda_distance],requires_grad=True)
            self.lambdas=torch.nn.Parameter(lambdas_tensor)
        else:
            lambdas_distance=1. - lambda_attention
            self.lambdas=(lambda_attention,lambdas_distance)
        self.linears=clones(nn.Linear(d_model, d_model).to("cuda"), 4)
        self.dropout=nn.Dropout(p=dropout)
    def forward(self,query,key,value,mask,adjacency_matrix):
        total_atom_number=query.size(0)
        query=query.to("cuda")
        key=key.to("cuda")
        value=value.to("cuda")
        q,k,v=[l(x).contiguous().view(total_atom_number,self.h,self.d_k).permute(1,0,2) for l,x in zip(self.linears,(query,key,value))]
        x,self.attn,self.self_attn=graph_attention(q,k,v,mask,adjacency_matrix,lambdas=self.lambdas,trainable_lambda=self.trainable_lambda,dropout=self.dropout)
        x=x.permute(1, 0, 2).contiguous().view(total_atom_number,self.h*self.d_k)
        return self.linears[-1](x)
class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,N_dense,dropout=0.1,leaky_relu_slope=0.0,dense_output_nonlinearity="relu"):
        super(PositionwiseFeedForward, self).__init__()
        self.N_dense = N_dense
        self.linears = clones(nn.Linear(d_model, d_model).to("cuda"), N_dense)
        self.dropout = clones(nn.Dropout(dropout).to("cuda"), N_dense)
        self.leaky_relu_slope = leaky_relu_slope
        if dense_output_nonlinearity == 'relu':
            self.dense_output_nonlinearity = lambda x: F.leaky_relu(x, negative_slope=self.leaky_relu_slope)
        elif dense_output_nonlinearity == 'tanh':
            self.tanh = torch.nn.Tanh()
            self.dense_output_nonlinearity = lambda x: self.tanh(x)
        elif dense_output_nonlinearity == 'none':
            self.dense_output_nonlinearity = lambda x: x
    def forward(self, x):
        if self.N_dense == 0:
            return x
        for i in range(len(self.linears)-1):
            x = self.dropout[i](F.leaky_relu(self.linears[i](x), negative_slope=self.leaky_relu_slope))
        return self.dropout[-1](self.dense_output_nonlinearity(self.linears[-1](x)))
class Embeddings(nn.Module):
    def __init__(self, d_model, d_atom, dropout):
        super(Embeddings, self).__init__()
        self.lin = nn.Linear(d_atom, d_model).to("cuda")
        self.dropout = nn.Dropout(dropout).to("cuda")
    def forward(self, x):
        return self.dropout(self.lin(x))
class SublayerConnection(nn.Module):#作残差连接
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, atom_dim, dropout, scale_norm):
        super(SublayerConnection, self).__init__()
        self.norm=nn.LayerNorm(atom_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        x=x.to("cuda")
        return x + self.dropout(sublayer(self.norm(x)))
class EncoderLayer(nn.Module):
    def __init__(self,h,d_model,dropout_attn,leaky_relu_slope,dropout_feedforward,lambda_attention,trainable_lambda,N_dense,scale_norm):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(h,d_model,dropout_attn,lambda_attention,trainable_lambda)
        self.feed_forward=PositionwiseFeedForward(d_model,N_dense,dropout_feedforward,leaky_relu_slope)
        self.d_model=d_model
        self.sublayer=clones(SublayerConnection(d_model,dropout_feedforward,scale_norm).to("cuda"), 2)
    def forward(self,x,mask,adjacency_matrix):
        x=self.sublayer[0](x,lambda x: self.self_attn(x,x,x,mask,adjacency_matrix))
        return self.sublayer[1](x,self.feed_forward)
class Encoder(nn.Module):
    def __init__(self,layer,N,scale_norm,atom_dim):
        super(Encoder, self).__init__()
        self.layers=clones(layer,N)
        self.norm = nn.LayerNorm(atom_dim)
    def forward(self,x,mask,adjacency_matrix):
        for layer in self.layers:
            x=layer(x,mask,adjacency_matrix)
        return self.norm(x)
class GraphTransformer(nn.Module):
    def __init__(self,d_atom,N,h,d_model,dropout_attn,leaky_relu_slope,dropout_feedforward,lambda_attention,trainable_lambda,N_dense,scale_norm):
        super(GraphTransformer, self).__init__()
        layer=EncoderLayer(h,d_model,dropout_attn,leaky_relu_slope,dropout_feedforward,lambda_attention,trainable_lambda,N_dense,scale_norm)
        self.encoder=Encoder(layer,N,scale_norm,d_model)
        self.src_embed=Embeddings(d_model,d_atom,dropout_feedforward)
    def forward(self,x,mask,adjacency_matrix):
        Transformer_features=self.encoder(self.src_embed(x),mask,adjacency_matrix)
        return Transformer_features

class GT(nn.Module):#out_features:输出的特征维度, d_atom:初始的原子维度 N:Transformer的层数 h:注意力的头数 d_model:Graph Transformer后得到的特征维度
    #dropout_attn:Graph Transformer中的Dropout leaky_relu_slope:leaky_relu的超参数 dropout_feedforward:前馈神经网络网络的dropout率
    #lambda_attention:注意力上的系数 trainable_lambda:是否参数可以训练
    #N_dense:
    def __init__(self,output_features,d_atom,N,h,d_model,dropout_attn,leaky_relu_slope,dropout_feedforward,lambda_attention,trainable_lambda,N_dense,scale_norm):
        super(GT, self).__init__()
        self.linear=nn.Linear(d_model,output_features).to("cuda")
        self.GeLU=nn.GELU().to("cuda")
        self.GraphTransformer=GraphTransformer(d_atom,N,h,d_model,dropout_attn,leaky_relu_slope,dropout_feedforward,lambda_attention,trainable_lambda,N_dense,scale_norm).to("cuda")
    def forward(self,x,mask,adjacency_matrix):
        Transformer_features=self.GraphTransformer(x,mask,adjacency_matrix)
        GeLUTransformer_features=self.GeLU(self.linear(Transformer_features))#[total_atom_number,features_dim]
        return GeLUTransformer_features,Transformer_features
class GAT(nn.Module):#in_features:初始化的原子特征. bond_features:初始化的化学键的特征 output_features:输出的特征
    #dropout_gnn_ratio:GNN中的dropout率
    def __init__(self,in_features,bond_features_dim,output_features,dropout_gnn_ratio,leaky_alpha,elu_alpha,number_layer):
        super(GAT, self).__init__()
        self.linear=nn.Linear(output_features,output_features).to("cuda")
        self.GeLU=nn.GELU()
        self.GATlayerV1=GATlayerV1(in_features,bond_features_dim,output_features,dropout_gnn_ratio,leaky_alpha,elu_alpha).to("cuda")
        self.GATlayerV2=GATlayerV2(output_features,dropout_gnn_ratio,leaky_alpha,elu_alpha).to("cuda")
        self.number_layer=number_layer
    def forward(self,atom_features,bond_features):
        all_tensors=[]
        for i in range(1,self.number_layer+1):
            if i==1:
                GAT_output_features,G_0=self.GATlayerV1(atom_features,bond_features)#[total_atom_number,features_dim]
            else:
                GAT_output_features=self.GATlayerV2(GAT_output_features,bond_features)#[total_atom_number,features_dim]
            all_tensors.append(GAT_output_features)
        all_tensors=torch.stack(all_tensors)#[n,total_atom_number,features_dim]
        averaged_tensor=all_tensors.mean(dim=0)#[total_atom_number,features_dim]
        output=self.GeLU(self.linear(averaged_tensor))
        return output#[total_atom_number,features_dim]