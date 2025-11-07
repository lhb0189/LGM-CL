from train import pretrain,set_log
import torch
import torch.nn as nn
from LossFunction import NT_Xent
from Image_model import VisionTransformer,Img_model

log_path="save_Image_model/log"
log=set_log("Pretrain",save_path=log_path)

num_heads=12
attention_dropout=0.0
proj_dropout=0.1
vis=True
hidden_size=768
mlp_dim=3072
dropout_rate=0.1
transformer_layer=12#12层Transformer
img_size=224#224*224的图像
size=16#16*16
mlp_dropout=0.1
in_channels=3
VIT=VisionTransformer(num_heads,attention_dropout,proj_dropout,vis,hidden_size,mlp_dim,dropout_rate,transformer_layer,img_size,size,dropout_rate,in_channels)
model=Img_model(VIT,hidden_size)

data_path="Process_Image_Datasets\\bace_Image\\"
augment_path="Process_Image_Datasets\\bace_rotate_Image\\"
#base_lr设置为3e-4 weight_decay=0.05
base_lr=3e-4
weight_decay=0.05
device=None
len_data=1513 #这个看具体数据集的长度
num_epochs=50#50个epoch

temperature=0.1
batch_size=128
pretrain(log,model,data_path,augment_path,len_data,num_epochs,base_lr,weight_decay,temperature,batch_size,device)