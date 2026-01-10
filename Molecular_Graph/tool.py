import os
import csv
import logging
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import auc, mean_squared_error, precision_recall_curve, roc_auc_score
from data import MoleDataSet, MoleData
import torch.distributed as dist
from graph import create_graph
from graph_model import get_mask
#这里要放在对比学习中,通过模型得到了每个样本的2个向量,接下来就是在batch中和其他样本进行损失函数的计算,正例是该样本通过2个模型得到的特征,负例是其他样本的这2个特征.
def load_model(model,path,cuda,log=None):
    #model是已经初始化了的模型,path是要加载的参数所在的路径,cuda:True
    if log is not None:
        debug=log.debug
    else:
        debug=print
    state=torch.load(path,map_location=lambda storage, loc: storage)
    state_dict=state['state_dict']
    model_state_dict=model.state_dict()
    load_state_dict={}
    for param in state_dict.keys():
        if param not in model_state_dict:
            debug(f'Parameter is not found: {param}.')
        elif model_state_dict[param].shape != state_dict[param].shape:
            debug(f'Shape of parameter is error: {param}.')
        else:
            load_state_dict[param] = state_dict[param]
            debug(f'Load parameter: {param}.')
    model_state_dict.update(load_state_dict)  # 更新参数,以使用训练好的参数
    model.load_state_dict(model_state_dict)  # 加载参数
    if cuda:
        model = model.to(torch.device("cuda"))#模型跑的时候放在cuda上
    return model#返回更新了参数的模型
class GatherLayer(torch.autograd.Function):#用于在多GPU/分布式环境下收集不同设备上的张量
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)
    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out
class NT_Xent(nn.Module):#batch_size:每个GPU上的样本数  tempature:NT-Xent损失函数式子中的温度参数  world_size:GPU数量
    #这个只要是输入的z_i,z_j特征维度一样,且z_i[k]和z_j[k]是同一个样本的2个不同的特征向量
    #训练前可以归一化一下,进而保证余弦相似度数值稳定
    def __init__(self, batch_size, temperature, world_size):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size
        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
    def mask_correlated_samples(self, batch_size, world_size):#返回一个掩码矩阵
        N = 2 * batch_size * world_size#总共有N=2*batch_size*world_size个样本
        mask = torch.ones((N, N), dtype=bool)#创造一个N*N的全1矩阵
        mask = mask.fill_diagonal_(0)#对角线给0,自己和自己不比较
        for i in range(batch_size * world_size):
            mask[i, batch_size + i] = 0#正样本对不记为负样本，这里得注意到第i个样本进行增强后,这个样本的位置在Batch_size+i这里
            mask[batch_size + i, i] = 0#对称
        return mask
    def forward(self, z_i, z_j):#z_i是[batch_size,feature_dim]的torch.tensor特征矩阵, z_j是样本经过数据增强后得到的[batch_size,feature_dim]的torch.tensor的特征矩阵
        N = 2 * self.batch_size * self.world_size
        z = torch.cat((z_i, z_j), dim=0)#z形状为[2*batch_size,feature_dim]
        if self.world_size > 1:#分布式的模式的话(即GPU有多个的情况下),通过GatherLayer聚合所有GPU的样本
            z = torch.cat(GatherLayer.apply(z), dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature#z.unsqueeze(1)是[N,1,feature_dim],z.unsqueeze(0)是[1,N,feature_dim]
        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss
def set_log(name, save_path):
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    if not log.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # 控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        log.addHandler(console_handler)
        # 文件输出
        os.makedirs(save_path, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(save_path, 'debug.log'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)
    return log
def load_data_for_pretrain(path):
    with open(path) as file:
        reader=csv.reader(file)
        next(reader)
        lines=[]
        for line in reader:
            lines.append(line)
        data=[]
        for line in lines:
            one=MoleData(line)
            data.append(one)
        data=MoleDataSet(data)
        fir_data_len=len(data)
        smi_exist=[]
        for i in range(fir_data_len):
            if data[i].mol is not None:
                smi_exist.append(i)
        data_val=MoleDataSet([data[i] for i in smi_exist])
        now_data_len=len(data_val)
        print('There are ', now_data_len, ' smiles in total.')  # 现在的数据
        if fir_data_len - now_data_len > 0:
            print('There are ', fir_data_len, ' smiles first, but ', fir_data_len - now_data_len,
                  ' smiles is invalid.  ')  # 记录有多少数据无效
    return data_val  # 返回MoleDataset这个类型.

def mkdir(path,isdir=True):#制造一个文件夹
    if isdir==False:
        path=os.path.dirname(path)
    if path!="":
        os.makedirs(path,exist_ok=True)

def load_data_for_finetune(path):
    with open(path) as file:
        reader=csv.reader(file)
        next(reader)
        lines=[]
        for line in reader:
            lines.append(line)
        data=[]
        for line in lines:
            one=MoleData(line)
            data.append(one)
        data=MoleDataSet(data)
        fir_data_len=len(data)
        print('There are ', fir_data_len, ' smiles in total.')
    return data

def split_data(dataset,split_type,size,seed):#这里由于参考的基准论文都是用的随机切割的 size:[0.6,0.2,0.2]
    if split_type=="random":
        dataset.random_data(seed)
        train_size=int(size[0]*len(dataset))
        val_size=int(size[1]*len(dataset))
        train_val_size=train_size+val_size
        train_data=dataset[:train_size]
        val_data=dataset[train_size:train_val_size]
        test_data=dataset[train_val_size:]
        return MoleDataSet(train_data),MoleDataSet(val_data),MoleDataSet(test_data)

def get_label_scaler(dataset):#用于回归任务中，将标签归一化处理
    label=dataset.label()
    label=np.array(label).astype(float)
    ave = np.nanmean(label, axis=0)  # 计算每列的均值 忽略NaN,这里是针对多回归任务,每列都是一个标签
    ave = np.where(np.isnan(ave), np.zeros(ave.shape), ave)  # 将NaN值替换为0
    std = np.nanstd(label, axis=0)  # 计算每列的标准差,忽略NaN
    std = np.where(np.isnan(std), np.ones(std.shape), std)  # 将NaN标准差替换为1
    std = np.where(std == 0, np.ones(std.shape), std)  # 将0标准差替换为1以避免除0
    change_1 = (label - ave) / std  # 标准化处理x-u/sigma
    label_changed = np.where(np.isnan(change_1), None, change_1)  # 将NaN结果换为None
    label_changed.tolist()  # 转回列表
    dataset.change_label(label_changed)  # 修改数据集中的标签
    return [ave, std]  # 返回均值和标准差

def get_loss(type):
    if type=='classification':
        return nn.BCEWithLogitsLoss(reduction='none')
    elif type=='regression':
        return nn.MSELoss(reduction='none')
    else:
        raise ValueError('type must be "classification" or "regression"')

def prc_auc(label,pred):#用于评价分类模型,当类别不平衡的是时候，尤其是正样本少的情况下
    prec,recall,_=precision_recall_curve(label,pred)
    result=auc(recall,prec)
    return result
def rmse(label,pred):#RMSE损失函数
    result=mean_squared_error(label,pred)
    return math.sqrt(result)

def get_metric(metric):
    if metric=='auc':
        return roc_auc_score
    elif metric=='prc_auc':
        return prc_auc
    elif metric=='rmse':
        return rmse
    else:
        raise ValueError('metric must be "auc" or "prc_auc" or "rmse"')

def save_model(path,model,scaler):
    #path：保存路径  BACE_model：使用的模型 scaler 是否数据的标签做z-score预处理
    if scaler!=None:
        state={
            "state_dict": model.state_dict(),
            'data_scaler': {
                'means': scaler[0],
                'stds': scaler[1]
            }  # 因为做了预处理 所以要记录预处理使用的均值和方差
        }
    else:
        state={
            "state_dict": model.state_dict(),
            "data_scaler": None
        }
    torch.save(state,path)

def load_model(model,path,cuda,log=None):
    #model是已经初始化了的模型,path是要加载的参数所在的路径,cuda:True
    if log is not None:
        debug=log.debug
    else:
        debug=print
    state=torch.load(path,map_location=lambda storage, loc: storage)
    state_dict=state['state_dict']
    model_state_dict=model.state_dict()
    load_state_dict={}
    for param in state_dict.keys():
        if param not in model_state_dict:
            debug(f'Parameter is not found: {param}.')
        elif model_state_dict[param].shape != state_dict[param].shape:
            debug(f'Shape of parameter is error: {param}.')
        else:
            load_state_dict[param] = state_dict[param]
            debug(f'Load parameter: {param}.')
    model_state_dict.update(load_state_dict)  # 更新参数,以使用训练好的参数
    model.load_state_dict(model_state_dict)  # 加载参数
    if cuda:
        model = model.to(torch.device("cuda"))#模型跑的时候放在cuda上
    return model#返回更新了参数的模型


def Pretrain(model,dataset,batch_size,optimizer,temperature):
    model.train()
    total_loss=0.0
    total_num=0
    data_used=0
    Batch_size=batch_size
    criterion = NT_Xent(batch_size, temperature, 1)
    for i in range(0,len(dataset),Batch_size):
        if data_used + Batch_size < len(dataset):
            data_now=MoleDataSet(dataset[i:i+Batch_size])
            data_used=data_used+Batch_size
            len_data=batch_size
            print("data used:"+str(data_used))
            smiles=data_now.smile()
            Graph_data=create_graph(smiles)
            atom_features,atom_index,bond_features=Graph_data.get_feature()#这个batch的原子特征,每个分子的原子索引,化学键特征
            bond_features = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in bond_features.items()}
            atom_features = atom_features.to("cuda")#原子特征
            adjacency_matrix = Graph_data.get_adjacency_matrix()
            adjacency_matrix = adjacency_matrix.to("cuda")
            mask_matrix = get_mask(Graph_data)#transformer的掩码矩阵
            mask_matrix = mask_matrix.to("cuda")
            x1,x2,out1,out2,Transformer_features=model(atom_features,bond_features,mask_matrix,adjacency_matrix,atom_index)
            loss=criterion(out1,out2)
            total_num=total_num+len_data
            total_loss+=loss.item() * len_data
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return total_loss/total_num
#if __name__ == "__main__":
#    batch_size = 4
#    feature_dim = 128
#    temperature = 0.5
#
#    z_i = torch.randn(batch_size, feature_dim).cuda()
 #   z_j = torch.randn(batch_size, feature_dim).cuda()

  #  criterion = NT_Xent(batch_size=batch_size, temperature=temperature, world_size=1).cuda()
   # loss = criterion(z_i, z_j)
    #print("Loss:", loss.item())