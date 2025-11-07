import torch
from rdkit import Chem
import pandas as pd
import os
from PIL import Image
#将SMILES->Image
#并将Image旋转90度作为数据增强
from rdkit.Chem import Draw
def smile2Img(smile,size,savepath):#将smiles字符串转化为图像 尺寸为224*224
    mol=Chem.MolFromSmiles(smile)
    image=Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(size,size))#这个函数需要注意的地方 mols得是列表对象 molsPerRow表示每一行中要放多少分子图像 subImgSize表示转化图像的维度
    if savepath is not None:
        image.save(savepath)
def smilestoImages(data_path):
    df=pd.read_csv(data_path)
    smiles=df["smiles"].tolist()
    path="Pretrain_Image_Datasets\\Zinc15_Image\\"
    os.makedirs(path,exist_ok=True)
    number=1
    for smile in smiles:
        savepath=path+str(number)+".png"
        number=number+1
        smile2Img(smile,224,savepath)
def Image_rotate_90(datasets_path,save_path,number):
    os.makedirs(save_path,exist_ok=True)
    for i in range(1,number+1):
        path=datasets_path+str(i)+".png"
        save_rotate_path=save_path+str(i)+".png"
        img=Image.open(path)
        rotated_90_img=img.transpose(Image.Transpose.ROTATE_90)
        rotated_90_img.save(save_rotate_path)
data_path="Pretrain_Datasets\\Zinc15.csv"
datasets_path="Pretrain_Image_Datasets\\Zinc15_Image\\"
save_path="Pretrain_Image_Datasets\\Zinc15_rotate_Image\\"

smilestoImages(data_path)#smiles批量转化为图像

length=333545
Image_rotate_90(datasets_path,save_path,length)#图像批量旋转90度
