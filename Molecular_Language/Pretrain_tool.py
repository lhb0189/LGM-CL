import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import logging
from loss_fc import NT_Xent
import os
class SmilesTextDataset(Dataset):
    def __init__(self, smiles_list, text_lists):
        assert len(smiles_list) == len(text_lists)
        self.smiles = smiles_list
        self.text = text_lists
    def __len__(self):
        return len(self.smiles)
    def __getitem__(self, idx):
        return {
            "smiles":self.smiles[idx],
            "text":self.text[idx]
        }
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
def train_one_epoch(model,batch_size,dataloader,optimizer,temperature,device):
    model.train()
    total_loss=0.0
    data_num=0
    data_used=0
    criterion=NT_Xent(batch_size,temperature,1)
    for batch in dataloader:
        smiles_input_ids = batch["smiles_input_ids"].to(device)
        smiles_attention_mask = batch["smiles_attention_mask"].to(device)
        text_input_ids = batch["text_input_ids"].to(device)
        text_attention_mask = batch["text_attention_mask"].to(device)
        output_MLPhead_smiles,output_smiles=model(smiles_input_ids,smiles_attention_mask)
        output_MLPHead_texts,output_texts=model(text_input_ids,text_attention_mask)
        loss=criterion(output_MLPHead_texts,output_MLPhead_smiles)
        total_num=total_num+batch_size
        total_loss+=loss.item()*batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss/total_num