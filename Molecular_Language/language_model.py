import os
import json
import torch.nn as nn
import torch.nn.functional
import torch
from packaging import version
from ops import StableDropout,MaskedLayerNorm
from disentangled_attention import DisentangledSelfAttention,build_relative_position
from torch.nn import LayerNorm
from collections.abc import Sequence
class BertSelfOutput(nn.Module):
    def __init__(self,hidden_size,layer_norm_eps,hidden_dropout_prob):#官方设置的 hidden_size:768 layer_norm_eps:1e-7 hidden_dropout_prob
        super().__init__()
        self.dense=nn.Linear(hidden_size,hidden_size)
        self.LayerNorm=nn.LayerNorm(hidden_size,layer_norm_eps)
        self.dropout = StableDropout(hidden_dropout_prob)
    def forward(self,hidden_states,input_states,mask=None):
        hidden_states=self.dense(hidden_states)
        hidden_states=self.dropout(hidden_states)
        hidden_states += input_states
        hidden_states = MaskedLayerNorm(self.LayerNorm,hidden_states)
        return hidden_states

class BertAttention(nn.Module):
    #这个是默认设置 后面可以稍微调一下 弄成轻量级的参数
    #num_attention_heads: 12 hidden_size: 768 hidden_dropout_prob:0.1 attention_probs_dropout_prob:0.1 layer_norm_eps:1e-7
    def __init__(self,num_attention_heads,hidden_size,hidden_dropout_prob,attention_probs_dropout_prob,layer_norm_eps):
        super().__init__()
        self.self=DisentangledSelfAttention(num_attention_heads,hidden_size,hidden_dropout_prob,attention_probs_dropout_prob)
        self.output=BertSelfOutput(hidden_size,layer_norm_eps,hidden_dropout_prob)
    def forward(self,hidden_states,attention_mask,return_att=False,query_states=None,relative_pos=None,rel_embeddings=None):
        output=self.self(hidden_states,attention_mask,return_att,query_states=query_states,relative_pos=relative_pos,rel_embeddings=rel_embeddings)
        self_output,att_matrix,att_logits_=output['hidden_states'],output['attention_probs'],output['attention_logits']
        if query_states is None:
            query_states=hidden_states
        attention_output=self.output(self_output,query_states,attention_mask)
        if return_att:
            return (attention_output,att_matrix)
        else:
            return attention_output

class BertIntermediate(nn.Module):#这里相当于进入了一层MLP层，激活函数用的GeLU
    def __init__(self,hidden_size,intermediate_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size,intermediate_size)
        self.intermediate_act_fn=torch.nn.functional.gelu
    def forward(self,hidden_states):
        hidden_states=self.dense(hidden_states)
        hidden_states=self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self,hidden_size,intermediate_size,layer_norm_eps,hidden_dropout_prob):
        super(BertOutput,self).__init__()
        self.dense=nn.Linear(intermediate_size,hidden_size)
        self.LayerNorm=LayerNorm(hidden_size,layer_norm_eps)
        self.dropout=StableDropout(hidden_dropout_prob)
    def forward(self,hidden_states,input_states,mask=None):
        hidden_states=self.dense(hidden_states)
        hidden_states=self.dropout(hidden_states)
        hidden_states=hidden_states+input_states
        hidden_states=MaskedLayerNorm(self.LayerNorm,hidden_states)
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self,hidden_size,intermediate_size,layer_norm_eps,hidden_dropout_prob,num_attention_heads,attention_probs_dropout_prob):
        super(BertLayer,self).__init__()
        self.attention=BertAttention(num_attention_heads,hidden_size,hidden_dropout_prob,attention_probs_dropout_prob,layer_norm_eps)
        self.intermediate=BertIntermediate(hidden_size,intermediate_size)
        self.output=BertOutput(hidden_size,intermediate_size,layer_norm_eps,hidden_dropout_prob)
    def forward(self,hidden_states,attention_mask,return_att=False,query_states=None,relative_pos=None,rel_embeddings=None):
        attention_output=self.attention(hidden_states,attention_mask,return_att=return_att,query_states=query_states,relative_pos=relative_pos,rel_embeddings=rel_embeddings)
        #进入Disengtangled_attention后出来
        #进入intermediate中
        #这个intermediate也就是一个Linear(hidden_size,intermediate_size),然后进入GeLU
        #之后再进入Output中，Linear(intermediate,hidden_size),dropout,这个中间输入再和Attention_output相加，
        if return_att:
            attention_output,att_matrix=attention_output
        intermediate_output=self.intermediate(attention_output)
        layer_output=self.output(intermediate_output,attention_output,attention_mask)
        if return_att:
            return (layer_output,att_matrix)
        else:
            return layer_output

class ConVLayer(nn.Module):#这个用个卷积来增加局部建模能力
    #给的配置是 conv_kernel_size=3, conv_groups=1 conv_act=gelu
    def __init__(self,hidden_size,layer_norm_eps,hidden_dropout_prob):
        super().__init__()
        kernel_size=3
        groups=1
        self.conv_act=torch.nn.functional.gelu
        self.conv = torch.nn.Conv1d(hidden_size,hidden_size,kernel_size,padding=(kernel_size-1)//2,groups=groups)
        self.LayerNorm=LayerNorm(hidden_size,layer_norm_eps)
        self.dropout=StableDropout(hidden_dropout_prob)
    def forward(self,hidden_states,residual_states,input_mask):
        out= self.conv(hidden_states.permute(0,2,1).contiguous()).permute(0,2,1).contiguous()
        rmask=(1-input_mask).bool()
        out.masked_fill_(rmask.unsqueeze(-1).expand(out.size()), 0)
        out=self.conv_act(self.dropout(out))
        output_states=MaskedLayerNorm(self.LayerNorm,residual_states+out,input_mask)
        return output_states

class BertEncoder(nn.Module):
    #默认的num_hidden_layers=12 可以稍微调少点
    def __init__(self,num_hidden_layers,hidden_size,intermediate_size,layer_norm_eps,hidden_dropout_prob,num_attention_heads,attention_probs_dropout_prob):
        super().__init__()
        self.layer=nn.ModuleList([BertLayer(hidden_size,intermediate_size,layer_norm_eps,hidden_dropout_prob,num_attention_heads,attention_probs_dropout_prob) for _ in range(num_hidden_layers)])
        self.relative_attention=True
        self.max_relative_positions=512
        self.position_buckets=256
        pos_ebd_size=self.max_relative_positions*2
        if self.position_buckets>0:
            pos_ebd_size=self.position_buckets*2
        self.rel_embeddings=nn.Embedding(pos_ebd_size,hidden_size)
        self.norm_rel_bed=["layer_norm"]
        self.LayerNorm=LayerNorm(hidden_size,layer_norm_eps,elementwise_affine=True)
        kernel_size=3
        self.with_conv=True
        self.conv=ConVLayer(hidden_size,layer_norm_eps,hidden_dropout_prob)

    def get_rel_embedding(self):
        rel_embeddings=self.rel_embeddings.weight
        rel_embeddings=self.LayerNorm(rel_embeddings)
        return rel_embeddings
    def get_attention_mask(self,attention_mask):
        if attention_mask.dim() <= 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
            attention_mask = attention_mask.byte()
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)
        return attention_mask

    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        if self.relative_attention and relative_pos is None:
            q = query_states.size(-2) if query_states is not None else hidden_states.size(-2)
            relative_pos = build_relative_position(q, hidden_states.size(-2), bucket_size=self.position_buckets, max_position=self.max_relative_positions,device=hidden_states.device)
        return relative_pos
    def forward(self,hidden_states, attention_mask, output_all_encoded_layers=True, return_att=False, query_states = None, relative_pos=None):
        if attention_mask.dim() <= 2:
            input_mask = attention_mask
        else:
            input_mask = (attention_mask.sum(-2) > 0).byte()
        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)
        all_encoder_layers = []
        att_matrices = []
        if isinstance(hidden_states, Sequence):
            next_kv = hidden_states[0]
        else:
            next_kv = hidden_states
        rel_embeddings = self.get_rel_embedding()
        for i, layer_module in enumerate(self.layer):
            output_states = layer_module(next_kv, attention_mask, return_att, query_states=query_states,
                                         relative_pos=relative_pos, rel_embeddings=rel_embeddings)
            if return_att:
                output_states, att_m = output_states
            if i == 0 and self.with_conv:
                prenorm = output_states  # output['prenorm_states']
                output_states = self.conv(hidden_states, prenorm, input_mask)
            if query_states is not None:
                query_states = output_states
                if isinstance(hidden_states, Sequence):
                    next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
            else:
                next_kv = output_states
            if output_all_encoded_layers:
                all_encoder_layers.append(output_states)
                if return_att:
                    att_matrices.append(att_m)
        if not output_all_encoded_layers:
            all_encoder_layers.append(output_states)
            if return_att:
                att_matrices.append(att_m)
        return {
            'hidden_states': all_encoder_layers,
            'attention_matrices': att_matrices
        }
#这里要拿最终的输出后的结果就是直接  outputs = encoder(hidden_states, attention_mask) final_hidden = outputs['hidden_states'][-1]   # 这才是多层 DeBERTa 编码后的最终表示

class BertEmbeddings(nn.Module):
    #vocal_size:128100  hidden_size:768 max_position_embeddings:512 type_vocal_size:0 layer_norm_eps:1e-7
    def __init__(self,hidden_size,vocab_size,max_position_embeddings,type_vocal_size,layer_norm_eps,hidden_dropout_prob):
        super(BertEmbeddings, self).__init__()
        padding_idx=0
        self.embedding_size=hidden_size
        self.word_embeddings=nn.Embedding(vocab_size,hidden_size)
        self.position_biased_input=True
        self.position_embeddings=nn.Embedding(max_position_embeddings,self.embedding_size)
        if type_vocal_size>0:
            self.token_type_embeddings=nn.Embedding(type_vocal_size,self.embedding_size)
        self.LayerNorm=LayerNorm(hidden_size,layer_norm_eps)
        self.dropout=StableDropout(hidden_dropout_prob)
        self.output_to_half=False
        self.type_vocab_size = type_vocal_size
    def forward(self,input_ids,token_type_ids=None,position_ids=None,mask=None):
        seq_length=input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids.long())
        embeddings = words_embeddings
        if self.type_vocab_size > 0:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings
        if self.position_biased_input:
            embeddings += position_embeddings
        embeddings = MaskedLayerNorm(self.LayerNorm, embeddings, mask)
        embeddings = self.dropout(embeddings)
        return {
            'embeddings': embeddings,
            'position_embeddings': position_embeddings}

class DeBERTa(torch.nn.Module):
    def __init__(self,hidden_size,vocab_size,max_position_embeddings,type_vocal_size,layer_norm_eps,hidden_dropout_prob,num_hidden_layers,intermediate_size,num_attention_heads,attention_probs_dropout_prob):
        super().__init__()
        #这个encoder是进入注意力层里面的
        #embedding是进入前的
        self.embeddings=BertEmbeddings(hidden_size,vocab_size,max_position_embeddings,type_vocal_size,layer_norm_eps,hidden_dropout_prob)
        self.encoder=BertEncoder(num_hidden_layers,hidden_size,intermediate_size,layer_norm_eps,hidden_dropout_prob,num_attention_heads,attention_probs_dropout_prob)
    def forward(self,input_ids,attention_mask=None,token_type_ids=None,output_all_encoded_layers=True,position_ids=None,return_att=False):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        ebd_output = self.embeddings(input_ids.to(torch.long), token_type_ids.to(torch.long), position_ids,
                                     attention_mask)
        embedding_output = ebd_output['embeddings']
        encoder_output = self.encoder(embedding_output,
                                      attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers, return_att=return_att)
        encoder_output.update(ebd_output)
        return encoder_output
def masked_mean_pooling(last_hidden,attention_mask):
    mask=attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)  # [B, H]
    denom = mask.sum(dim=1).clamp(min=1e-6)  # [B, 1]
    return summed / denom
class Pretrain_DeBERTa(torch.nn.Module):
    def __init__(self,hidden_size,vocab_size,max_position_embeddings,type_vocal_size,layer_norm_eps,hidden_dropout_prob,num_hidden_layers,intermediate_size,num_attention_heads,attention_probs_dropout_prob,dropout_mlp):
        super().__init__()
        self.Deberta=DeBERTa(hidden_size,vocab_size,max_position_embeddings,type_vocal_size,layer_norm_eps,hidden_dropout_prob,num_hidden_layers,intermediate_size,num_attention_heads,attention_probs_dropout_prob)
        self.MLPHead=nn.Sequential(
            nn.Linear(hidden_size,hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_mlp),
            nn.Linear(hidden_size,256),
        )
    def forward(self,input_ids,attention_mask=None,token_type_ids=None,output_all_encoded_layers=True,position_ids=None,return_att=False):
        output=self.Deberta(input_ids,attention_mask,token_type_ids,output_all_encoded_layers,position_ids,return_att)
        last_hidden_states=output['hidden_states'][-1]#[batch_size,seq_len,hidden_size]
        sent=masked_mean_pooling(last_hidden_states,attention_mask)
        proj=self.MLPHead(sent)
        return proj,sent

#要取最后一层的输出就是 last_hidden = encoder_output['hidden_states'][-1] shape为：[batch_size,seq_len,hidden_size]
#还要注意要池化操作
