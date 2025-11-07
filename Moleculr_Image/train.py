import torch
import torch.nn as nn
#将图像和增强后的图像经过VIT得到对应的向量后进行对比学习
import os
import logging
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
from LossFunction import NT_Xent
transform=transforms.Compose([transforms.ToTensor()])#[0,255]->[0,1]进行一个归一化
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
class PNGDataset(Dataset):
    def __init__(self, folder_path,data_len,transform=None):
        self.img_paths=[]
        for i in range(1,data_len+1):
            name=folder_path+str(i)+".png"
            self.img_paths.append(name)
        self.transform = transform
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img_path=self.img_paths[idx]
        img=Image.open(img_path).convert("RGB")#转换为RGB三通道的图像
        if self.transform:
            img=self.transform(img)
        return img
def pretrain(logger,model,data_path,augment_path,len_data,num_epochs,base_lr,weight_decay,temperature,batch_size,device):
    info=logger.info
    debug=logger.debug
    if device is None:
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    datasets = PNGDataset(data_path,len_data,transform=transform)
    augment_datasets=PNGDataset(augment_path,len_data,transform=transform)
    dataloader=DataLoader(datasets,batch_size=batch_size, shuffle=False)
    augmentloader=DataLoader(augment_datasets,batch_size=batch_size,shuffle=False)
    optimizer=torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=(0.9,0.999),
        eps=1e-8,
        weight_decay=weight_decay,
    )
    debug("pretrain")
    debug(model)
    for epoch in range(1,num_epochs+1):
        info(f'Epoch {epoch}')
        model.train()
        total_loss=0.0
        total_num=0
        for batch_idx,(data,augment_data) in enumerate(zip(dataloader, augmentloader)):
            len_data=len(data)
            data=data.to(device)
            augment_data=augment_data.to(device)
            x1,out1,attn_weight1=model(data)#x1是经过Vision Transformer后得到的向量，out1是再经过Projection head的 attn_weight1是该数据下的注意力分数矩阵
            x2,out2,attn_weight2=model(augment_data)#同上
            criterion=NT_Xent(batch_size,temperature,1)
            loss=criterion(out1,out2)
            total_num=total_num+len_data
            total_loss+=loss.item()*len_data
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss=total_loss/total_num
        info(f"Average train_loss:{train_loss}")
        torch.save(model.state_dict(),'Save_Image_model/'+str(epoch)+"_model_encoder_VIT"+".pkl")
