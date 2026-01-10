import os
import csv
import logging
import math
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import auc, mean_squared_error, precision_recall_curve, roc_auc_score
from data import MoleDataSet,MoleData
import pandas as pd
def load_model(model,path,cuda,log=None):
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
    model_state_dict.update(load_state_dict)
    model.load_state_dict(model_state_dict)
    if cuda:
        model = model.to(torch.device("cuda"))
    scaler = None
    if 'data_scaler' in state and state['data_scaler'] is not None:
        scaler = [
            state['data_scaler']['means'],
            state['data_scaler']['stds']
        ]
        debug("Load label scaler (mean/std) from checkpoint.")

    return model, scaler
def set_log(name, save_path):
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    if not log.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        log.addHandler(console_handler)
        os.makedirs(save_path, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(save_path, 'debug.log'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)
    return log
def mkdir(path,isdir=True):
    if isdir==False:
        path=os.path.dirname(path)
    if path!="":
        os.makedirs(path,exist_ok=True)
def get_label_scaler_regression(dataset):
    label = dataset.label()
    label = np.array(label).astype(float)
    ave = np.nanmean(label, axis=0)
    ave = np.where(np.isnan(ave), np.zeros(ave.shape), ave)

    std = np.nanstd(label, axis=0)
    std = np.where(np.isnan(std), np.ones(std.shape), std)
    std = np.where(std == 0, np.ones(std.shape), std)

    return [ave, std]

def get_loss(type):
    if type=='classification':
        return nn.BCEWithLogitsLoss(reduction='none')
    elif type=='regression':
        return nn.MSELoss(reduction='none')
    else:
        raise ValueError('type must be "classification" or "regression"')

def prc_auc(label,pred):
    prec,recall,_=precision_recall_curve(label,pred)
    result=auc(recall,prec)
    return result
def rmse(label,pred):
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
    if scaler!=None:
        state={
            "state_dict": model.state_dict(),
            'data_scaler': {
                'means': scaler[0],
                'stds': scaler[1]
            } 
        }
    else:
        state={
            "state_dict": model.state_dict(),
            "data_scaler": None
        }
    torch.save(state,path)
def split_data(dataset,split_type,size,seed,log):
    if split_type=="random":
        dataset.random_data(seed)
        train_size=int(size[0]*len(dataset))
        val_size=int(size[1]*len(dataset))
        train_val_size=train_size+val_size
        train_data=dataset[:train_size]
        val_data=dataset[train_size:train_val_size]
        test_data=dataset[train_val_size:]
        return MoleDataSet(train_data),MoleDataSet(val_data),MoleDataSet(test_data)
    elif split_type=="scaffold":
        return scaffold_split(dataset,size,seed,log)
    else:
        raise ValueError('type must be "random" or "scaffold"')
def load_data(smiles_label_path,text_path):
    with open(smiles_label_path) as file:
        reader=csv.reader(file)
        next(reader)
        lines=[]
        df_text=pd.read_csv(text_path)
        text=df_text["text"].tolist()
        for line in reader:
            lines.append(line)
        data=[]
        number=0
        for line in lines:
            one=MoleData(line,text[number])
            number+=1
            data.append(one)
        data=MoleDataSet(data)
        fir_data_len=len(data)
        print('There are ', fir_data_len, ' smiles in total.')

    return data
