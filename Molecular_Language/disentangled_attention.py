import torch.nn as nn
import torch
import transformers
import math
from ops import MaskedLayerNorm,StableDropout,XSoftmax
from functools import lru_cache
#DeBERTa的创新点主要体在两方面，1.Disentangled Attention Mechanism 2.Enhanced Masker Decoder
#由于预训练手段使用的是对比学习，因此Enhanced Masker Decoder不使用，主要使用的技术是这个Disentangled Attention Mechanism
#Disentangled Attention Mechanism 主要体现的创新点是在于将位置信息和文本内容分成两个向量进行编码了，由此对传统的Transformer架构也在数学表达式上进行了创新。

@lru_cache(maxsize=128)
def make_log_bucket_dict(bucket_size,max_position,device=None):#做relative position bucketing，把相对位置信息映射到bucket索引中，之后用于得到相对位置信息的编码向量
    relative_pos=torch.arange(-max_position,max_position,device=device)
    sign = torch.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = torch.where((relative_pos < mid) & (relative_pos > -mid), torch.tensor(mid - 1).to(relative_pos),
                          torch.abs(relative_pos))
    log_pos = torch.ceil(torch.log(abs_pos / mid) / math.log((max_position - 1) / mid) * (mid - 1)) + mid
    bucket_pos = torch.where(abs_pos <= mid, relative_pos, (log_pos * sign).to(relative_pos)).to(torch.long)
    return bucket_pos

def make_log_bucket_position(relative_pos, bucket_size, max_position):#这个还是给出bucket id
    relative_pos = torch.clamp(relative_pos,-max_position+1, max_position-1) + max_position
    bucket_dict = make_log_bucket_dict(bucket_size, max_position, relative_pos.device)
    for d in range(relative_pos.dim()-1):
        bucket_dict = bucket_dict.unsqueeze(0)
        bucket_pos = torch.gather(bucket_dict.expand(list(relative_pos.size())[:-1] + [bucket_dict.size(-1)]), index=relative_pos.long(), dim=-1)
    return bucket_pos

@lru_cache(maxsize=128)
def build_relative_position(query_size,key_size,bucket_size=-1,max_position=-1,device=None):#论文里面的relative position embedding项 作为位置信息
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
        self.num_attention_heads=num_attention_heads#注意力头数
        self.attention_head_size=int(hidden_size/num_attention_heads)#每个注意力头的维度
        self.all_head_size=self.num_attention_heads * self.attention_head_size#这个就是看返回后的维度，怕遇到除不尽的情况下，正常设置都是除的尽的，所以还是hidden_size的数值
        self.query=nn.Linear(hidden_size,self.all_head_size,bias=True)#Query的数值，这里disentangled Attention 下就是代表Context内容用的Query
        self.key=nn.Linear(hidden_size,self.all_head_size,bias=True)#Key的数值
        self.value=nn.Linear(hidden_size,self.all_head_size,bias=True)
        self.share_att_key=True#跨层共享参数，这里是DeBERTa的技巧，专门在 W_K上实现每层一样的参数
        self.pos_att_type=["p2c","c2p"]#论文里面的p2c和c2p
        self.relative_attention=True
        self.positon_buckets=256
        self.max_relative_positions=512
        self.pos_ebd_size=256
        self.pos_dropout=StableDropout(hidden_dropout_prob)
        self.dropout=StableDropout(attention_probs_dropout_prob)
        self._register_load_state_dict_pre_hook(self._pre_load_hook)
    def transpose_for_scores(self,x,attention_heads):#将文本信息的格式对准
        new_x_shape=x.size()[:-1]+(attention_heads,-1)
        x=x.view(*new_x_shape)
        return x.permute(0,2,1,3).contiguous().view(-1,x.size(1),x.size(-1))
    #这里通常query_states都是None，除非想做Cross-attention那样的情况下
    #这个rel_embeddiings貌似后面会给
    def forward(self,hidden_states,attention_mask,return_att=False,query_states=None,relative_pos=None,rel_embeddings=None):
        if query_states is None:
            query_states=hidden_states
        query_layer = self.transpose_for_scores(self.query(query_states), self.num_attention_heads).float()#Q
        key_layer = self.transpose_for_scores(self.key(hidden_states), self.num_attention_heads).float()#K
        value_layer = self.transpose_for_scores(self.value(hidden_states), self.num_attention_heads)#V
        rel_att=None
        scale_factor=3
        scale=1/math.sqrt(query_layer.size(-1) * scale_factor)
        attention_scores = torch.bmm(query_layer,key_layer.transpose(-1,-2)*scale)#标准的计算注意力分数Qk/根号3k，这里计算按论文里面来看就是c2c的内容信息的计算
        rel_embeddings=self.pos_dropout(rel_embeddings)
        rel_att=self.disentangled_attention_bias(query_layer,key_layer,relative_pos,rel_embeddings,scale_factor)#这里计算的是 c2p和p2c信息
        if rel_att is not None:
            attention_scores=(attention_scores + rel_att)#公式里面是A_{i,j}= c2c +c2p + p2c 上面就是正常的c2c内容计算
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
            q = query_layer.size(-2)#按照query_layer的shape应该是[Batch_size*seq_len, attention_heads, head_dim]
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
            #这个score就是论文里面的那两个附加项，就是通过矩阵P得到的，而矩阵P表示的是token的相对位置信息
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