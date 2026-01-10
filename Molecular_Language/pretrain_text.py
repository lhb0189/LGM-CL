import torch.nn
import torch
from transformers import AutoTokenizer
import csv
import pandas as pd
from language_model import Pretrain_DeBERTa
import torch.optim as optim
from Pretrain_tool import SmilesTextDataset,train_one_epoch,set_log
from torch.utils.data import Dataset, DataLoader

log_path="Save_DeBerta_model/log"
log=set_log("Pretrain",save_path=log_path)
info=log.info
debug=log.debug

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Model_Name="deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(Model_Name,use_fast=False)

smiles_file="Pretrain_Datasets//Zinc15.csv"
texts_file="Pretrain_Text_Datasets//Zinc15_text.csv"
smiles_df=pd.read_csv(smiles_file)
texts_df=pd.read_csv(texts_file)
smiles_data=smiles_df["smiles"].tolist()
texts_data=texts_df["text"].tolist()

Datasets=SmilesTextDataset(smiles_data,texts_data)

def collate_fn_contrastive(batch,max_len=256):
    smiles = [x["smiles"] for x in batch]
    text   = [x["text"] for x in batch]
    smiles_tok = tokenizer(smiles, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    text_tok   = tokenizer(text,   padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    return {
        "smiles_input_ids": smiles_tok["input_ids"],
        "smiles_attention_mask": smiles_tok["attention_mask"],
        "text_input_ids": text_tok["input_ids"],
        "text_attention_mask": text_tok["attention_mask"],
    }
dataloader=DataLoader(Datasets,batch_size=128,shuffle=True,drop_last=True,collate_fn=collate_fn_contrastive)


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
dropout_mlp=0.05
model=Pretrain_DeBERTa(hidden_size,vocab_size,max_position_embeddings,type_vocal_size,layer_norm_eps,hidden_dropout_prob,num_hidden_layers,intermediate_size,num_attention_heads,attention_probs_dropout_prob,dropout_mlp).to(device)

optimizer=optim.Adam(model.parameters(),lr=0.0001,weight_decay=1e-3)

batch_size=128
epochs=50
temperature=0.1

debug("Pretraining Model")
debug(model)

for epoch in range(1,epochs+1):
    info(f"Epoch{epoch}")
    train_loss=train_one_epoch(model,batch_size,dataloader,optimizer,temperature,device)
    info(f"Average Train Loss:{train_loss}")
    torch.save(model.state_dict(),"Save_DeBERTa_model/"+str(epoch)+"_encoder"+".pkl")

