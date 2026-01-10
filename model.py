#MACCS:167 bit PubChem: 881 bits Pharmacophere:441
from rdkit import Chem
from PubChemFP import GetPubChemFPs
import torch
from rdkit.Chem import AllChem,MACCSkeys
import torch.nn.functional as F
import torch.nn as nn
def smiles_to_MACCS(smiles):
    mol=Chem.MolFromSmiles(smiles)
    fp = MACCSkeys.GenMACCSKeys(mol)
    return torch.tensor(fp).view(1,-1)#1*167
def smiles_to_PubChem(smiles):
    mol=Chem.MolFromSmiles(smiles)
    mol2 = Chem.AddHs(mol)
    return torch.tensor(GetPubChemFPs(mol2)).view(1,-1)#1*881
def smiles_to_Pharmacophere(smiles):
    mol=Chem.MolFromSmiles(smiles)
    fp_phaErGfp = AllChem.GetErGFingerprint(mol,fuzzIncrement=0.3,maxPath=21,minPath=1)
    return torch.tensor(fp_phaErGfp).view(1,-1)#1*441

class FPN(nn.Module):
    def __init__(self,Linear_dim,dropout_FPN,hidden_size):
        super(FPN,self).__init__()
        self.hidden_size = hidden_size
        self.initial_dim=1489
        self.MLP=nn.Sequential(
            nn.Linear(self.initial_dim,Linear_dim),
            nn.Dropout(p=dropout_FPN),
            nn.ReLU(),
            nn.Linear(Linear_dim,hidden_size),
        )
    def forward(self,smiles):
        smiles_to_fp=[]
        for smile in smiles:
            fp_MACCS=smiles_to_MACCS(smile)#torch类型[batch_size,167]
            fp_PubChem=smiles_to_PubChem(smile)#torch类型[batch_size,881]
            fp_ErG=smiles_to_Pharmacophere(smile)#torch类型[batch_size,441]
            fp_list = torch.cat([fp_MACCS, fp_PubChem, fp_ErG], dim=1)
            fp_list = fp_list.squeeze(0)
            smiles_to_fp.append(fp_list)
        fp_list = torch.stack(smiles_to_fp, dim=0).to(device="cuda",dtype=torch.float32)
        fpn_output=self.MLP(fp_list)
        return fpn_output
def masked_mean_pooling(last_hidden,attention_mask):
    mask=attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)  # [B, H]
    denom = mask.sum(dim=1).clamp(min=1e-6)  # [B, 1]
    return summed / denom

class CrossAttention(nn.Module):
    def __init__(self, dim_q, dim_kv, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim_q, dim)
        self.k_proj = nn.Linear(dim_kv, dim)
        self.v_proj = nn.Linear(dim_kv, dim)
        self.out_proj = nn.Linear(dim, dim)
    def forward(self, query, key_value):
        """
        query:      [B,  dim_q]
        key_value: [B, dim_kv]
        return : [B,dim]
        """
        Q = self.q_proj(query)        # [B, dim]
        K = self.k_proj(key_value)    # [B, dim]
        V = self.v_proj(key_value)    # [B, dim]
        attn = (Q * K).sum(dim=-1, keepdim=True) / (Q.size(-1) ** 0.5)
        attn = torch.sigmoid(attn)
        out = attn * V
        out = self.out_proj(out)
        return out

class MultiViewModel(nn.Module):
    def __init__(self,encoder1,encoder2,encoder3,encoder4,FPN_hidden_dim,text_input_dim,text_hidden_dim,Graph_input_dim,Graph_hidden_dim,Final_hidden_dim,task_num,dropout_forward):
        super(MultiViewModel,self).__init__()
        self.AttentiveFP=encoder1
        self.GT=encoder2
        self.DeBERTa=encoder3#[batch_size,text_input_dim]
        self.FPN=encoder4#[batch_size,FPN_hidden_dim]
        self.CrossAttention1=CrossAttention(Graph_input_dim,Graph_input_dim,Graph_hidden_dim)#[Batch_size,Graph_hidden_dim]
        self.CrossAttention2=CrossAttention(text_input_dim,text_input_dim,text_hidden_dim)#[Batch_size,text_hidden_dim]
        self.CrossAttention3=CrossAttention(text_hidden_dim,FPN_hidden_dim,Final_hidden_dim)
        self.CrossAttention4=CrossAttention(Graph_hidden_dim,FPN_hidden_dim,Final_hidden_dim)
        self.MLP=nn.Sequential(
            nn.Dropout(p=dropout_forward),
            nn.Linear(in_features=2*Final_hidden_dim,out_features=2*Final_hidden_dim,bias=True),
            nn.ReLU(),
            nn.Dropout(p=dropout_forward),
            nn.Linear(in_features=2*Final_hidden_dim,out_features=task_num,bias=True)
        )
    def forward(self,smiles,atom_features,bond_features,mask,adjacency_matrix,atom_index,text_input_ids,text_attention_mask,smiles_input_ids,smiles_attention_mask,token_type_ids=None,output_all_encoded_layers=True,position_ids=None,return_att=False):
        x1=self.AttentiveFP(atom_features,bond_features)
        x2,Transformer_features=self.GT(atom_features,mask,adjacency_matrix)
        embedding_GAT=[]
        embedding_GT=[]
        for index_number in atom_index:
            initial_index,atom_number=index_number[0], index_number[1]
            single_molecule_GAT=x1[initial_index:initial_index+atom_number]
            single_molecule_GT=x2[initial_index:initial_index+atom_number]
            pool_GAT=single_molecule_GAT.sum(dim=0)
            average_GAT=pool_GAT/atom_number
            pool_GT=single_molecule_GT.sum(dim=0)
            average_GT=pool_GT/atom_number
            embedding_GAT.append(average_GAT)
            embedding_GT.append(average_GT)
        embedding_stack_GAT=torch.stack(embedding_GAT, dim=0)
        embedding_stack_GT=torch.stack(embedding_GT, dim=0)
        x3=self.DeBERTa(text_input_ids,text_attention_mask,token_type_ids,output_all_encoded_layers,position_ids,return_att)
        x4=self.DeBERTa(smiles_input_ids,smiles_attention_mask,token_type_ids,output_all_encoded_layers,position_ids,return_att)
        hidden_states_texts=x3["hidden_states"][-1]
        hidden_states_Smiles=x4["hidden_states"][-1]
        embedding_texts=masked_mean_pooling(hidden_states_texts,text_attention_mask)
        embedding_Smiles=masked_mean_pooling(hidden_states_Smiles,smiles_attention_mask)
        embedding_FPN=self.FPN(smiles)#[molecule_number,FPN_hidden_dim]

        fused_Graph=self.CrossAttention1(embedding_stack_GAT,embedding_stack_GT)#[molecules,Graph_hidden_dim]
        fused_Text=self.CrossAttention2(embedding_Smiles,embedding_texts)#[molecules,text_hidden_dim]

        fused_FPN_Graph=self.CrossAttention4(fused_Graph,embedding_FPN)#[molecules,Final_hidden_dim]
        fused_FPN_Text=self.CrossAttention3(fused_Text,embedding_FPN)#[molecules,Final_hidden_dim]
        Mole_feature=torch.cat([fused_FPN_Graph,fused_FPN_Text],dim=1)
        pred=self.MLP(Mole_feature)
        return pred,embedding_stack_GAT,embedding_stack_GT,embedding_Smiles,embedding_texts,embedding_FPN,fused_Graph,fused_Text,fused_FPN_Text,fused_FPN_Graph,Mole_feature
