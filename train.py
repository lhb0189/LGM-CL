import torch.nn as nn
import torch
from useful_tools import *
import copy
from Molecular_Graph.graph import create_graph
from Molecular_Graph.graph_model import get_mask

def epoch_train(model,dataset,loss_f,tokenizer,optimizer,seed,batch_size,max_len=256,scaler=None):
    model.train()
    dataset.random_data(seed)
    loss_sum=0
    data_used=0
    Batch_size=batch_size
    for i in range(0,len(dataset),Batch_size):
        if data_used + Batch_size > len(dataset):
            data_now=MoleDataSet(dataset[i:])
        else:
            data_now=MoleDataSet(dataset[i:i+Batch_size])
        smiles=data_now.smile()
        label=data_now.label()
        texts=data_now.texts()
        Graph_data=create_graph(smiles)
        atom_features, atom_index, bond_features = Graph_data.get_feature()
        bond_features = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in bond_features.items()}
        atom_features = atom_features.to("cuda")#原子特征
        adjacency_matrix = Graph_data.get_adjacency_matrix()
        adjacency_matrix = adjacency_matrix.to("cuda")
        mask_matrix = get_mask(Graph_data)#transformer的掩码矩阵
        mask_matrix = mask_matrix.to("cuda")
        smiles_tok = tokenizer(smiles, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        text_tok = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        smiles_input_ids=smiles_tok["input_ids"].to("cuda")
        smiles_attention_mask=smiles_tok["attention_mask"].to("cuda")
        text_input_ids=text_tok["input_ids"].to("cuda")
        text_attention_mask=text_tok["attention_mask"].to("cuda")
        ave, std = (scaler if scaler is not None else (None, None))
        target = torch.tensor(
            [
                [
                    ((x - ave[t]) / std[t]) if (scaler is not None and x is not None) else (x if x is not None else 0.0)
                    for t, x in enumerate(tb)
                ]
                for tb in label
            ],
            dtype=torch.float32,
            device="cuda"
        )
        mask = torch.tensor(
            [[0.0 if x is None else 1.0 for x in tb] for tb in label],
            dtype=torch.float32,
            device="cuda"
        )

        optimizer.zero_grad()
        pred,embedding_stack_GAT,embedding_stack_GT,embedding_Smiles,embedding_texts,embedding_FPN,fused_Graph,fused_Text,fused_FPN_Text,fused_FPN_Graph,Mole_feature=model(smiles,atom_features,bond_features,mask_matrix,adjacency_matrix,atom_index,text_input_ids,text_attention_mask,smiles_input_ids,smiles_attention_mask)
        loss_mat = loss_f(pred, target)  # [B, T]
        loss = (loss_mat * mask).sum() / mask.sum().clamp_min(1.0)
        loss_sum+=loss.item()
        data_used+=len(smiles)
        loss.backward()
        optimizer.step()
def predict(model,dataset,batch_size,scaler,tokenizer,max_len=256):
    model.eval()
    pred=[]
    data_total=len(dataset)
    for i in range(0,data_total,batch_size):
        if i+batch_size > data_total:
            data_now=MoleDataSet(dataset[i:])
        else:
            data_now = MoleDataSet(dataset[i:i + batch_size])
        smiles = data_now.smile()
        texts=data_now.texts()
        Graph_data = create_graph(smiles)
        atom_features, atom_index, bond_features = Graph_data.get_feature()
        bond_features = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in bond_features.items()}
        atom_features = atom_features.to("cuda")  # 原子特征
        adjacency_matrix = Graph_data.get_adjacency_matrix()
        adjacency_matrix = adjacency_matrix.to("cuda")
        mask_matrix = get_mask(Graph_data)  # transformer的掩码矩阵
        mask_matrix = mask_matrix.to("cuda")
        smiles_tok = tokenizer(smiles, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        text_tok = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        smiles_input_ids=smiles_tok["input_ids"].to("cuda")
        smiles_attention_mask=smiles_tok["attention_mask"].to("cuda")
        text_input_ids=text_tok["input_ids"].to("cuda")
        text_attention_mask=text_tok["attention_mask"].to("cuda")
        with torch.no_grad():
            pred_now, embedding_stack_GAT, embedding_stack_GT, embedding_Smiles, embedding_texts, embedding_FPN, fused_Graph, fused_Text, fused_FPN_Text, fused_FPN_Graph, Mole_feature = model(
                smiles, atom_features, bond_features, mask_matrix, adjacency_matrix, atom_index, text_input_ids,
                text_attention_mask, smiles_input_ids, smiles_attention_mask)
        pred_now=pred_now.data.cpu().numpy()
        if scaler is not None:
            ave = scaler[0]
            std = scaler[1]
            pred_now = np.array(pred_now).astype(float)
            change_1 = pred_now * std + ave
            pred_now = np.where(np.isnan(change_1), None, change_1)
        pred_now = pred_now.tolist()
        pred.extend(pred_now)
    return pred

def compute_score(pred,label,metric_f,log,task_num,dataset_type):
    info=log.info
    if len(pred)==0:
        return [float('nan')] * task_num
    pred_val=[]
    label_val=[]
    for i in range(task_num):
        pred_val_i=[]
        label_val_i=[]
        for j in range(len(pred)):
            if label[j][i] is not None:
                pred_val_i.append(pred[j][i])
                label_val_i.append(label[j][i])
        pred_val.append(pred_val_i)
        label_val.append(label_val_i)
    result=[]
    for i in range(task_num):
        if dataset_type == 'classification':
            if all(one == 0 for one in label_val[i]) or all(one == 1 for one in label_val[i]):
                info('Warning: All labels are 1 or 0.')
                result.append(float('nan'))
                continue
            if all(one == 0 for one in pred_val[i]) or all(one == 1 for one in pred_val[i]):
                info('Warning: All predictions are 1 or 0.')
                result.append(float('nan'))
                continue
        re=metric_f(label_val[i],pred_val[i])
        result.append(re)
    return result


def training(log,dataset_path,text_path,dataset_type,seed,val_path,test_path,split,split_type,metric,model,tokenizer,save_path,batch_size,init_lr,total_epochs,task_name):
    info=log.info
    debug=log.debug
    debug("Starting loading data")
    dataset=load_data(dataset_path,text_path)
    task_num=dataset.task_num()
    model_path=os.path.join(save_path,"seed_"+str(seed))#每个种子一个对应的目录存储
    mkdir(model_path)
    debug(f'Splitting dataset with Seed = {seed}.')
    if val_path != None:
        val_data = load_data(val_path)
    if test_path != None:
        test_data = load_data(test_path)
    elif val_path != None:
        split_ratio = (split[0], 0, split[2])
        train_data, _, test_data = split_data(dataset, split_type, split_ratio, seed)
    elif test_path != None:
        split_ratio = (split[0], split[1], 0)
        train_data, val_data, _ = split_data(dataset, split_type, split_ratio, seed)
    else:
        train_data, val_data, test_data = split_data(dataset, split_type, split, seed)
    debug(
        f'Dataset size: {len(dataset)}    Train size: {len(train_data)}    Val size: {len(val_data)}    Test size: {len(test_data)}')  # 输出三种数据集的数量
    if dataset_type=="regression":
        label_scaler=get_label_scaler_regression(train_data)
    else:
        label_scaler=None
    loss_f=get_loss(dataset_type)
    metric_f=get_metric(metric)
    new_model=copy.deepcopy(model).to(torch.device("cuda"))
    model=model.to(torch.device("cuda"))
    optimizer=torch.optim.Adam(params=model.parameters(),lr=init_lr,weight_decay=1e-3)
    if dataset_type=="classification":
        best_score=-float("inf")
    else:
        best_score=float("inf")
    best_epoch=0
    for epoch in range(1,total_epochs+1):
        info(f'Epoch {epoch}')
        epoch_train(model,train_data,loss_f,tokenizer,optimizer,seed+epoch-1,batch_size,max_len=256,scaler=label_scaler if dataset_type=="regression" else None)
        train_pred = predict(model,train_data,batch_size,label_scaler,tokenizer)
        train_label = train_data.label()
        train_score = compute_score(train_pred, train_label, metric_f, log, task_num, dataset_type)
        val_pred=predict(model,val_data,batch_size,label_scaler,tokenizer)
        val_label=val_data.label()
        val_score=compute_score(val_pred,val_label,metric_f,log,task_num,dataset_type)
        average_train_score=np.nanmean(train_score)
        info(f'Train{metric}={average_train_score:.6f}')
        average_val_score=np.nanmean(val_score)
        info(f'Validation{metric}={average_val_score:.6f}')
        test_label = test_data.label()
        test_pred = predict(model, test_data, batch_size, label_scaler,tokenizer)
        test_score = compute_score(test_pred, test_label, metric_f, log, task_num, dataset_type)
        average_test_score = np.nanmean(test_score)
        info(f"Seed {seed} :test{metric}={average_test_score:.6f}")
        if task_num > 1:
            for one_name, one_score in zip(task_name, test_score):
                info(f'test {one_name} {metric}= {one_score:.6f}')
        if dataset_type == "classification" and average_val_score > best_score:
            best_score = average_val_score
            best_epoch = epoch
            save_model(os.path.join(model_path, 'model.pt'), model, label_scaler)
        elif dataset_type == "regression" and average_val_score < best_score:
            best_score = average_val_score
            best_epoch = epoch
            save_model(os.path.join(model_path, "model.pt"), model, label_scaler)
    info(f"Best validation {metric}={best_score:.6f} on epoch {best_epoch}")
    model, label_scaler = load_model(
        new_model,
        os.path.join(model_path, "model.pt"),
        cuda=True,
        log=log
    )
    test_label=test_data.label()
    test_pred = predict(model, test_data, batch_size, label_scaler, tokenizer)
    test_score=compute_score(test_pred,test_label,metric_f,log,task_num,dataset_type)
    average_test_score=np.nanmean(test_score)
    info(f"Seed {seed} :test{metric}={average_test_score:.6f}")
    if task_num>1:
        for one_name,one_score in zip(task_name,test_score):
            info(f'task {one_name}{metric}={one_score:.6f}')
    return test_score