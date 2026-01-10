import torch
from model import MultiViewModel,FPN
from Molecular_Graph.graph_model import GT,GAT
import numpy as np
import random
from Molecular_Language.language_model import DeBERTa
from tool import mkdir,set_log
from transformers import AutoTokenizer
import os
from train import training

Model_Name="deberta-v3-base"
tokenizer=AutoTokenizer.from_pretrained(Model_Name,use_fast=False,local_files_only=True)

log_path="Finetune/log"
log=set_log("Finetune_for_model",save_path=log_path)
info=log.info
debug=log.debug

device=torch.device("cuda")
#GT parameter
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
#Attentive FP parameter
in_features=57
bond_features=15
dropout_gnn_ratio=0.5
elu_alpha=1.0
number_layer=2
epochs=50
#DeBerta parameter=
hidden_size=768
vocab_size=128100
max_position_embeddings=512
type_vocal_size=0
layer_norm_eps=1e-7
hidden_dropout_prob=0.1
num_hidden_layers=4
intermediate_size=3072
num_attention_heads=12
attention_probs_dropout_prob=0.1
#FPN parameter 1489->512->300
dropout_FPN=0.1
FP_Linear_dim=512
FPN_hidden_size=300
# model parameter
text_hidden_dim=256
Graph_hidden_dim=256
Final_hidden_dim=256
#dataset parameter
task_num=1# 任务数量
save_path="Finetune\\Save_model\\Bace_model"#存储model.pt的路径
dataset_path="Process_Datasets\\bace.csv"#数据集路径
text_path="Process_Text_Datasets\\bace_text.csv"#增强文本路径
dataset_type="classification" #任务类型，"classification" or "regression"
val_path=None
test_path=None
split=[0.6,0.2,0.2]
split_type="random"
init_lr=1e-3
batch_size=64
total_epochs=50
task_name=["1"]#任务数量
is_class_if=1# regression:0 classification:1
metric="auc"#"rmse or auc"
seed_list=[1,2]

for seed in seed_list:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    Graph_Transformer = GT(output_features, d_atom, N, h, d_model, dropout_attn, leaky_relu_slope, dropout_feedward,lambda_attention, trainable_lambda, N_dense, scale_norm)
    AttentiveFP = GAT(in_features, bond_features, output_features, dropout_gnn_ratio, leaky_relu_slope, elu_alpha,number_layer)
    AttentiveFP_sd = torch.load("Molecular_Graph/Save_gnn_model/46_model_encoder_GAT.pkl", map_location=device)
    Graph_Transformer_sd = torch.load("Molecular_Graph/Save_gnn_model/46_model_encoder_GT.pkl", map_location=device)
    AttentiveFP.load_state_dict(AttentiveFP_sd, strict=True)
    Graph_Transformer.load_state_dict(Graph_Transformer_sd, strict=True)
    DeBERTa_model = DeBERTa(hidden_size, vocab_size, max_position_embeddings, type_vocal_size, layer_norm_eps,
                            hidden_dropout_prob, num_hidden_layers, intermediate_size, num_attention_heads,
                            attention_probs_dropout_prob)
    full_sd=torch.load("Molecular_Language/Save_DeBERTa_model/15_encoder.pkl",map_location=device)
    deberta_sd = {
        k.replace("Deberta.", ""): v
        for k, v in full_sd.items()
        if k.startswith("Deberta.")
    }
    DeBERTa_model.load_state_dict(deberta_sd, strict=True)
    Fingerprint_Network=FPN(FP_Linear_dim,dropout_FPN,FPN_hidden_size)
    Fingerprint_Network.to(device)
    Model=MultiViewModel(AttentiveFP, Graph_Transformer,DeBERTa_model,Fingerprint_Network,FPN_hidden_size,hidden_size,text_hidden_dim,output_features,Graph_hidden_dim,Final_hidden_dim,task_num,dropout_feedward)
    Model.to(device)
    info(f"Seed{seed}")
    Save_path_seed=os.path.join(save_path)
    mkdir(Save_path_seed)
    fold_score=training(log,dataset_path,text_path,dataset_type,seed,val_path,test_path,split,split_type,metric,Model,tokenizer,Save_path_seed,batch_size,init_lr,total_epochs,task_name)