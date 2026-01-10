import torch.nn as nn
import torch
import torch.distributed as dist
#损失函数NT_Xent以进行对比学习
class GatherLayer(torch.autograd.Function):#用于在多GPU/分布式环境下收集不同设备上的张量
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)
    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out
class NT_Xent(nn.Module):#batch_size:每个GPU上的样本数  tempature:NT-Xent损失函数式子中的温度参数  world_size:GPU数量
    #这个只要是输入的z_i,z_j特征维度一样,且z_i[k]和z_j[k]是同一个样本的2个不同的特征向量
    #训练前可以归一化一下,进而保证余弦相似度数值稳定
    def __init__(self, batch_size, temperature, world_size):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size
        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
    def mask_correlated_samples(self, batch_size, world_size):#返回一个掩码矩阵
        N = 2 * batch_size * world_size#总共有N=2*batch_size*world_size个样本
        mask = torch.ones((N, N), dtype=bool)#创造一个N*N的全1矩阵
        mask = mask.fill_diagonal_(0)#对角线给0,自己和自己不比较
        for i in range(batch_size * world_size):
            mask[i, batch_size + i] = 0#正样本对不记为负样本，这里得注意到第i个样本进行增强后,这个样本的位置在Batch_size+i这里
            mask[batch_size + i, i] = 0#对称
        return mask
    def forward(self, z_i, z_j):#z_i是[batch_size,feature_dim]的torch.tensor特征矩阵, z_j是样本经过数据增强后得到的[batch_size,feature_dim]的torch.tensor的特征矩阵
        N = 2 * self.batch_size * self.world_size
        z = torch.cat((z_i, z_j), dim=0)#z形状为[2*batch_size,feature_dim]
        if self.world_size > 1:#分布式的模式的话(即GPU有多个的情况下),通过GatherLayer聚合所有GPU的样本
            z = torch.cat(GatherLayer.apply(z), dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature#z.unsqueeze(1)是[N,1,feature_dim],z.unsqueeze(0)是[1,N,feature_dim]
        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss