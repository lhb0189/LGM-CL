import torch.nn as nn
import torch
import transformers
import math
from ops import MaskedLayerNorm,StableDropout,XSoftmax
from functools import lru_cache
@lru_cache(maxsize=128)
def make_log_bucket_dict(bucket_size,max_position,device=None):
    relative_pos=torch.arange(-max_position,max_position,device=device)
    sign = torch.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = torch.where((relative_pos < mid) & (relative_pos > -mid), torch.tensor(mid - 1).to(relative_pos),
                          torch.abs(relative_pos))
    log_pos = torch.ceil(torch.log(abs_pos / mid) / math.log((max_position - 1) / mid) * (mid - 1)) + mid
    bucket_pos = torch.where(abs_pos <= mid, relative_pos, (log_pos * sign).to(relative_pos)).to(torch.long)
    return bucket_pos

def make_log_bucket_position(relative_pos, bucket_size, max_position):
    relative_pos = torch.clamp(relative_pos,-max_position+1, max_position-1) + max_position
    bucket_dict = make_log_bucket_dict(bucket_size, max_position, relative_pos.device)
    for d in range(relative_pos.dim()-1):
        bucket_dict = bucket_dict.unsqueeze(0)
        bucket_pos = torch.gather(bucket_dict.expand(list(relative_pos.size())[:-1] + [bucket_dict.size(-1)]), index=relative_pos.long(), dim=-1)
    return bucket_pos

@lru_cache(maxsize=128)
def build_relative_position(query_size,key_size,bucket_size=-1,max_position=-1,device=None):
    q_ids=torch.arange(0,query_size)#[0,1,2,...,query_size-1]
    k_ids=torch.arange(0,key_size)#[0,1,2,...key_size-1]
    if device is not None:
        q_ids=q_ids.to(device)
        k_ids=k_ids.to(device)
    rel_pos_ids=q_ids.view(-1,1) - k_ids.view(-1,1)
    if bucket_size>0 and max_position>0:
        rel_pos_ids=make_log_bucket_position(rel_pos_ids,bucket_size,max_position)
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = rel_pos_ids.unsqueeze(0)
    return rel_pos_ids

class DisentangledSelfAttention(nn.Module):
    def __init__(self,num_attention_heads,hidden_size,hidden_dropout_prob,attention_probs_dropout_prob):
        super().__init__()
        self.num_attention_heads=num_attention_heads
        self.attention_head_size=int(hidden_size/num_attention_heads)
        self.all_head_size=self.num_attention_heads * self.attention_head_size
        self.query=nn.Linear(hidden_size,self.all_head_size,bias=True)
        self.key=nn.Linear(hidden_size,self.all_head_size,bias=True)
        self.value=nn.Linear(hidden_size,self.all_head_size,bias=True)
        self.share_att_key=True
        self.pos_att_type=["p2c","c2p"]
        self.relative_attention=True
        self.positon_buckets=256
        self.max_relative_positions=512
        self.pos_ebd_size=256
        self.pos_dropout=StableDropout(hidden_dropout_prob)
        self.dropout=StableDropout(attention_probs_dropout_prob)
        self._register_load_state_dict_pre_hook(self._pre_load_hook)
    def transpose_for_scores(self,x,attention_heads):
        new_x_shape=x.size()[:-1]+(attention_heads,-1)
        x=x.view(*new_x_shape)
        return x.permute(0,2,1,3).contiguous().view(-1,x.size(1),x.size(-1))
    def forward(self,hidden_states,attention_mask,return_att=False,query_states=None,relative_pos=None,rel_embeddings=None):
        if query_states is None:
            query_states=hidden_states
        query_layer = self.transpose_for_scores(self.query(query_states), self.num_attention_heads).float()
        key_layer = self.transpose_for_scores(self.key(hidden_states), self.num_attention_heads).float()
        value_layer = self.transpose_for_scores(self.value(hidden_states), self.num_attention_heads)
        rel_att=None
        scale_factor=3
        scale=1/math.sqrt(query_layer.size(-1) * scale_factor)
        attention_scores = torch.bmm(query_layer,key_layer.transpose(-1,-2)*scale)
        rel_embeddings=self.pos_dropout(rel_embeddings)
        rel_att=self.disentangled_attention_bias(query_layer,key_layer,relative_pos,rel_embeddings,scale_factor)
        if rel_att is not None:
            attention_scores=(attention_scores + rel_att)
        attention_scores = (attention_scores - attention_scores.max(dim=-1, keepdim=True).values.detach()).to(hidden_states)
        attention_scores = attention_scores.view(-1, self.num_attention_heads, attention_scores.size(-2),attention_scores.size(-1))
        _attention_probs = XSoftmax.apply(attention_scores,attention_mask,-1)
        attention_probs = self.dropout(_attention_probs)
        context_layer = torch.bmm(attention_probs.view(-1,attention_probs.size(-2),attention_probs.size(-1)),value_layer)
        context_layer = context_layer.view(-1,self.num_attention_heads,context_layer.size(-2),context_layer.size(-1)).permute(0,2,1,3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        context_layer=context_layer.view(*new_context_layer_shape)
        return {
            'hidden_states': context_layer,
            'attention_probs': _attention_probs,
            'attention_logits': attention_scores
        }

    def disentangled_attention_bias(self,query_layer,key_layer,relative_pos,rel_embeddings,scale_factor):
        if relative_pos is None:
            q = query_layer.size(-2)
            relative_pos=build_relative_position(q,key_layer.size(-2),bucket_size=self.positon_buckets,max_position=self.max_relative_positions,device=query_layer.device)
        if relative_pos.dim()==2:
            relative_pos=relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim()==3:
            relative_pos=relative_pos.unsqueeze(1)
        elif relative_pos.dim()==4:
            raise ValueError(f'Relative postion ids must be of dim 2 or 3 or 4. {relative_pos.dim()}')
        att_span=self.pos_ebd_size
        relative_pos = relative_pos.long().to(query_layer.device)
        rel_embeddings = rel_embeddings[self.pos_ebd_size - att_span:self.pos_ebd_size + att_span, :].unsqueeze(0)
        if self.share_att_key:
            pos_query_layer=self.transpose_for_scores(self.query(rel_embeddings),self.num_attention_heads).repeat(query_layer.size(0)//self.num_attention_heads,1,1)
            pos_key_layer=self.transpose_for_scores(self.key(rel_embeddings),self.num_attention_heads).repeat(query_layer.size(0)//self.num_attention_heads,1,1)
        score=0
        #content - >position
        if "c2p" in self.pos_att_type:
            scale=1/math.sqrt(pos_key_layer.size(-1)*scale_factor)
            c2p_att = torch.bmm(query_layer, pos_key_layer.transpose(-1, -2).to(query_layer) * scale)
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1).squeeze(0).expand([query_layer.size(0), query_layer.size(1), relative_pos.size(-1)])
            c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_pos)
            score += c2p_att
        #position -> content
        if "p2c" in self.pos_att_type or 'p2p' in self.pos_att_type:
            scale=1/math.sqrt(pos_query_layer.size(-1)*scale_factor)
        if "p2c" in self.pos_att_type:
            p2c_att=torch.bmm(pos_query_layer.to(key_layer)*scale,key_layer.transpose(-1,-2))
            p2c_att=torch.gather(p2c_att,dim=-2,index=c2p_pos)
            score+=p2c_att
        return score
    def _pre_load_hook(self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs):
        self_state = self.state_dict()
        if ((prefix + 'query_proj.weight') not in state_dict) and ((prefix + 'in_proj.weight') in state_dict):
            v1_proj = state_dict[prefix+'in_proj.weight']
            v1_proj = v1_proj.unsqueeze(0).reshape(self.num_attention_heads, -1, v1_proj.size(-1))
            q,k,v=v1_proj.chunk(3, dim=1)
            state_dict[prefix + 'query_proj.weight'] = q.reshape(-1, v1_proj.size(-1))
            state_dict[prefix + 'key_proj.weight'] = k.reshape(-1, v1_proj.size(-1))
            state_dict[prefix + 'key_proj.bias'] = self_state['key_proj.bias']
            state_dict[prefix + 'value_proj.weight'] = v.reshape(-1, v1_proj.size(-1))
            v1_query_bias = state_dict[prefix + 'q_bias']
            state_dict[prefix + 'query_proj.bias'] = v1_query_bias
            v1_value_bias = state_dict[prefix +'v_bias']
            state_dict[prefix + 'value_proj.bias'] = v1_value_bias
            v1_pos_key_proj = state_dict[prefix + 'pos_proj.weight']
            state_dict[prefix + 'pos_key_proj.weight'] = v1_pos_key_proj
            v1_pos_query_proj = state_dict[prefix + 'pos_q_proj.weight']
            state_dict[prefix + 'pos_query_proj.weight'] = v1_pos_query_proj
            v1_pos_query_proj_bias = state_dict[prefix + 'pos_q_proj.bias']
            state_dict[prefix + 'pos_query_proj.bias'] = v1_pos_query_proj_bias
            state_dict[prefix + 'pos_key_proj.bias'] = self_state['pos_key_proj.bias']
            del state_dict[prefix + 'in_proj.weight']
            del state_dict[prefix + 'q_bias']
            del state_dict[prefix + 'v_bias']
            del state_dict[prefix + 'pos_proj.weight']
            del state_dict[prefix + 'pos_q_proj.weight']
            del state_dict[prefix + 'pos_q_proj.bias']
