import torch
import torch.nn as nn
import math
from torch.nn.modules.utils import _pair
import copy
#model for Vision Transformer
#由于Transformer原本是用于文本的
#最初的输入是[batch_size,每句话的文本信息]，要转化为[batch_size,seq_len,hidden_size]
#其中seq_len是文本中最长的分词（token）的长度，每个分词会被映射到一个具体的数值id，长度不够的话就填充0
#之后再蒋每个token的数值id映射到具体的向量（即为hidden_size尺寸的向量），这样去得到一个[batch_size,seq_len,hidden_size]
#CLS是表示全局语义的信息
#position_embeedding表示图像的位置信息
def swwish(x):
    return x * torch.sigmoid(x)
#Transformer的多头注意力层
class Attention(nn.Module):#num_heads:注意力的头数 hidden_size:输入进注意力的矩阵的维度 attention_dropout:attention层的dropout率 proj_dropout:映射层的dropout率 vis:是否可视化注意力分数矩阵，表示那个地方的权重分配更多
    def __init__(self,num_heads,hidden_size,attention_dropout,proj_dropout,vis):
        super(Attention,self).__init__()
        self.vis=vis
        self.num_attention_heads = num_heads#注意力头的个数
        self.attention_head_size=int(hidden_size/ self.num_attention_heads)#每个头分得到的隐藏层的维度
        self.all_head_size=self.num_attention_heads *  self.attention_head_size
        self.query=nn.Linear(hidden_size,self.all_head_size)
        self.key=nn.Linear(hidden_size,self.all_head_size)
        self.value=nn.Linear(hidden_size,self.all_head_size)
        self.out=nn.Linear(hidden_size,hidden_size)
        self.attn_dropout=nn.Dropout(p=attention_dropout)
        self.proj_dropout=nn.Dropout(p=proj_dropout)
        self.softmax=nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):#将输入形状从[batch_size,seq_len,hidden_size]->[batch_size,num_heads,seq_len,head_size]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    def forward(self,x):
        mixed_query_layer=self.query(x)
        mixed_key_layer=self.key(x)
        mixed_value_layer=self.value(x)
        query_layer=self.transpose_for_scores(mixed_query_layer)
        key_layer=self.transpose_for_scores(mixed_key_layer)
        value_layer=self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))#Q点积K
        attention_scores=attention_scores / math.sqrt(self.attention_head_size)#这里除以根号K
        attention_probs=self.softmax(attention_scores)
        weights=attention_probs if self.vis else None#可视化与否
        attention_probs=self.attn_dropout(attention_probs)#attn层的dropout
        context_layer=torch.matmul(attention_probs,value_layer)#score*V
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)#将[batch_size,num_heads,seq_len,head_size]->[batch_size,seq_len,hidden_size]
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights
class MLP(nn.Module):
    #hidden_size:隐藏层的维度 mlp_dim:经过MLP层通过Linear中间所得到的转换层的维度
    def __init__(self,hidden_size,mlp_dim,dropout_rate):
        super(MLP,self).__init__()
        self.fc1=nn.Linear(hidden_size,mlp_dim)
        self.fc2=nn.Linear(mlp_dim,hidden_size)
        self.act_fn=nn.functional.gelu
        self.dropout=nn.Dropout(p=dropout_rate)
        self._init_weights()
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias,std=1e-6)
        nn.init.normal_(self.fc2.bias,std=1e-6)
    def forward(self,x):
        x=self.fc1(x)
        x=self.act_fn(x)
        x=self.dropout(x)
        x=self.fc2(x)
        x=self.dropout(x)
        return x
class Embedddings(nn.Module):#把输入图像的[batch_size,224,224,3]切分成小块path，再通过卷积或者混合CNN特征提取，拼接上CLS全局语义信息标记和位置编码positional encoding，生成供Transformer编码器使用的输入序列
    #img_size:图像的尺寸 size:patch的尺寸,按照VIT原论文的话就是16*16 hidden_size：卷积后得到的维度数 dropout:dropout率 in_channesl:图像的通道数
    def __init__(self,img_size,size,hidden_size,dropout_rate,in_channels=3):
        super(Embedddings,self).__init__()
        img_size=_pair(img_size)#将224->[224,224],如果输入的是[224,224],那么就还是[224,224]
        patch_size=_pair(size)
        n_patches=(img_size[0]// patch_size[0])*(img_size[1]//patch_size[1])#图像被分成了一个14*14的网格 每个格子patch是16*16像素，总共有196个patch作为token
        self.patch_embeddings=nn.Conv2d(in_channels=in_channels,out_channels=hidden_size,kernel_size=patch_size,stride=patch_size)
        self.positon_embeddings=nn.Parameter(torch.zeros(1,n_patches+1,hidden_size))
        self.cls_token=nn.Parameter(torch.zeros(1,1,hidden_size))
        self.dropout=nn.Dropout(p=dropout_rate)
    def forward(self,x):
        batch_size=x.shape[0]
        cls_token=self.cls_token.expand(batch_size,-1,-1)#将[1,1,hidden_size]->[batch_size,1,hidden_size]
        x=self.patch_embeddings(x)#[batch_size,3,224,224]->[batch_size,768,14,14]
        x=x.flatten(2)#[batch_size,768,14,14]->[batch_size,768,196]
        x=x.transpose(-1,-2)#[batch_size,196,768]
        x=torch.cat((cls_token,x),dim=1)
        embeddings=x+self.positon_embeddings
        embeddings=self.dropout(embeddings)
        return embeddings
class Block(nn.Module):
    def __init__(self,num_heads,attention_dropout,proj_dropout,vis,hidden_size,mlp_dim,dropout_rate):
        super(Block,self).__init__()
        self.hidden_size=hidden_size
        self.attention_norm=nn.LayerNorm(hidden_size,eps=1e-6)
        self.ffn_norm=nn.LayerNorm(hidden_size,eps=1e-6)
        self.ffn=MLP(hidden_size,mlp_dim,dropout_rate)
        self.attn=Attention(num_heads,hidden_size,attention_dropout,proj_dropout,vis)
    def forward(self,x):
        h=x
        x=self.attention_norm(x)
        x,weights=self.attn(x)
        x=x+h
        h=x
        x=self.ffn_norm(x)
        x=self.ffn(x)
        x=x+h
        return x, weights
class Encoder(nn.Module):
    def __init__(self,num_heads,attention_dropout,proj_dropout,vis,hidden_size,mlp_dim,dropout_rate,transformer_layer):
        super(Encoder,self).__init__()
        self.vis=vis
        self.layer=nn.ModuleList()
        self.encoder_norm=nn.LayerNorm(hidden_size,eps=1e-6)
        for i in range(transformer_layer):
            layer=Block(num_heads,attention_dropout,proj_dropout,vis,hidden_size,mlp_dim,dropout_rate)
            self.layer.append(copy.deepcopy(layer))
    def forward(self,x):
        attn_weights=[]
        for layer_block in self.layer:
            x,weights=layer_block(x)
            if self.vis:
                attn_weights.append(weights)
        encoded=self.encoder_norm(x)
        return encoded,attn_weights
class VisionTransformer(nn.Module):
    def __init__(self,num_heads,attention_dropout,proj_dropout,vis,hidden_size,mlp_dim,dropout_rate,transformer_layer,img_size,size,mlp_dropout,in_channels):
        super(VisionTransformer,self).__init__()
        self.embeddings=Embedddings(img_size,size,hidden_size,mlp_dropout,in_channels)
        self.encoder=Encoder(num_heads,attention_dropout,proj_dropout,vis,hidden_size,mlp_dim,dropout_rate,transformer_layer)
    def forward(self,x):
        embedding_output=self.embeddings(x)
        encoded,attn_weights=self.encoder(embedding_output)
        return encoded,attn_weights
class Img_model(nn.Module):#图像经过Transformer得到一个768维度的向量,通过projection head得到用于对比学习的向量
    def __init__(self,encoder1,input_dim):
        super(Img_model,self).__init__()
        self.transformer=encoder1
        #Projection head 这个按照SimCLR的设置来的  论文里面可以陈述这点
        self.projection_head=nn.Sequential(
            nn.Linear(input_dim,512),
            nn.ReLU(),
            nn.Linear(512,128),
        )
    def forward(self,x):
        x,attn_weights=self.transformer(x)
        out1=self.projection_head(x)
        return x,out1,attn_weights
