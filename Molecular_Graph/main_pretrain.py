import os
import torch
from tool import NT_Xent,Pretrain,load_data_for_pretrain,set_log
from graph_model import GAT,GT
from model_gnn_pre import GNN_model
import torch.optim as optim
#日志文件
log_path="Save_gnn_model/log"
log=set_log("Pretrain",save_path=log_path)
info=log.info
debug=log.debug
# 预训练在这里跑
#考虑一下初始化方面,超参数方面
device=torch.device("cuda")
temperature=0.1
batch_size=128
#GT的初始化超参数
output_features=110
d_atom=57
h=19
N=3
d_k=96
d_model=d_k*h
dropout_attn=0.5
leaky_relu_slope=0.01
dropout_feedward=0.05
lambda_attention=1.0
trainable_lambda="False"
N_dense=2
scale_norm="LN"
#GAT的初始化超参数
in_features=57
bond_features=15
dropout_gnn_ratio=0.5
elu_alpha=1.0
number_layer=2
batch_size=128
epochs=50
model_encoder1=GAT(in_features,bond_features,output_features,dropout_gnn_ratio,leaky_relu_slope,elu_alpha,number_layer)#GAT的初始化参数
model_encoder2=GT(output_features,d_atom,N,h,d_model,dropout_attn,leaky_relu_slope,dropout_feedward,lambda_attention,trainable_lambda,N_dense,scale_norm)#GT的初始化参数
model=GNN_model(model_encoder1,model_encoder2,output_features,output_features,dropout_feedward)
model=model.cuda()
optimizer=optim.Adam(model.parameters(),lr=0.0001,weight_decay=1e-3)
path="Pretrain_Datasets\\Zinc15.csv"
dataset=load_data_for_pretrain(path)
debug("Pretraining Model")
debug(model)

for epoch in range(1,epochs+1):
    info(f'Epoch {epoch}')
    train_loss=Pretrain(model,dataset,batch_size,optimizer,temperature)
    info(f"Average train_loss:{train_loss}")
    torch.save(model_encoder1.state_dict(),'Save_gnn_model/'+str(epoch)+"_model_encoder_GAT"+".pkl" )
    torch.save(model_encoder2.state_dict(),"Save_gnn_model/"+str(epoch)+"_model_encoder_GT"+".pkl" )
    torch.save(model.state_dict(),"Save_gnn_model/"+str(epoch)+"_model"+".pkl" )