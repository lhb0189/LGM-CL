import torch
from torch.nn import LayerNorm
from packaging import version
import torch.onnx.symbolic_helper as sym_help
from torch.onnx.symbolic_opset9 import masked_fill,softmax


class XSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(self,input,mask,dim):
        self.dim=dim
        rmask = ~(mask.bool())
        output = input.masked_fill(rmask, float('-inf'))
        output = torch.softmax(output,self.dim)
        output.masked_fill_(rmask, 0)
        self.save_for_backward(output)
        return output
    @staticmethod
    def backward(self,grad_output):#这里由于torch新版本那个库不外放了，这个地方改了原代码
        output, = self.saved_tensors
        dim = self.dim
        grad_output=grad_output.contiguous()
        output=output.contiguous()
        sum=(grad_output * output).sum(dim=dim,keepdim=True)
        grad_input=output*(grad_output - sum)
        return grad_input, None,None
    @staticmethod
    def symbolic(g,self,mask,dim):
        mask_cast_value = g.op("Cast", mask, to_i=sym_help.cast_pytorch_to_onnx['Long'])
        r_mask = g.op("Cast",
                      g.op("Sub", g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64)), mask_cast_value),
                      to_i=sym_help.cast_pytorch_to_onnx['Byte'])
        output = masked_fill(g, self, r_mask, g.op("Constant", value_t=torch.tensor(float('-inf'))))
        output = softmax(g, output, dim)
        return masked_fill(g, output, r_mask, g.op("Constant", value_t=torch.tensor(0, dtype=torch.uint8)))

class DropoutContext(object):
    def __init__(self):
        self.dropout=0
        self.mask=None
        self.scale=1
        self.reuse_mask=True
def get_mask(input,local_context):
    #当 local_context 不是DropoutContext的时候， 这里也不知道他源代码这个地方的不是的时候觉得会输入啥东西
    if not isinstance(local_context,DropoutContext):
        dropout=float(local_context)
        mask=None
    #这里如果是的话，就是首先得到的Dropout概率
    else:
        dropout = local_context.dropout
        dropout = dropout * local_context.scale #这里乘以一个Scale来动态调整实际的Dropout比例子
        mask = local_context.mask if local_context.reuse_mask else None
    if dropout > 0 and mask is None:
        keep = torch.empty_like(input).bernoulli_(1 - dropout).to(torch.bool)
        mask = ~keep  # True 表示丢弃
    if isinstance(local_context,DropoutContext):
        if local_context.mask is None:
            local_context.mask=mask
    #mask:None或者一个和input形状一样的布尔张量，表示那些位置被丢弃
    #Dropout：使用的Dropout概率
    return mask,dropout

#这里这个XDropout就是一个nn.Dropout的一个增强版,前向和反向都用的同一个mask来掩藏信息以对某些神经元进行隐藏来减少过拟合，
class XDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input,local_ctx):
        mask,dropout=get_mask(input,local_ctx)
        ctx.scale=1.0/(1-dropout)
        if dropout>0:
            ctx.save_for_backward(mask)
            return input.masked_fill(mask,0)*ctx.scale
        else:
            return input
    @staticmethod
    def backward(ctx,grad_output):
        if ctx.scale>1:
            (mask,)=ctx.saved_tensors
            return grad_output.masked_fill(mask,0)*ctx.scale, None
        else:
            return grad_output,None

class StableDropout(torch.nn.Module):
    def __init__(self,drop_prob):
        super().__init__()
        self.drop_prob=drop_prob
        self.count=0
        self.context_stack=None
    def forward(self,x):
        if self.training and self.drop_prob>0:
            return XDropout.apply(x,self.get_context())
        return x
    def clear_context(self):
        self.count=0
        self.context_stack=None
    def init_context(self,reuse_mask=True,scale=1):
        if self.context_stack is None:
            self.context_stack=[]
        self.count=0
        for c in self.context_stack:
            c.reuse_mask=reuse_mask
            c.scale=scale
    def get_context(self):
        if self.context_stack is not None:
            if self.count >= len(self.context_stack):
                self.context_stack.append(DropoutContext())
            ctx=self.context_stack[self.count]
            ctx.dropout=self.drop_prob
            self.count+=1
            return ctx
        else:
            return self.drop_prob

def MaskedLayerNorm(layerNorm, x, mask=None):
    # layerNorm: nn.LayerNorm 实例
    y = layerNorm(x)
    if mask is None:
        return y
    # mask 预期 [B, S]；如果传成 [B,1,S,S] 之类，尽量压回 [B,S]
    if mask.dim() == 4:
        # 常见 attention mask: [B,1,S,S] -> 取对角线/或 squeeze 后再取任意一维
        mask = mask.squeeze(1).squeeze(1)  # 可能变成 [B,S,S] 或 [B,S]
        if mask.dim() == 3:
            # 退而求其次：用任意一行判断 token 是否有效
            mask = (mask.sum(-1) > 0)
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)          # [B,S,1]
    mask = mask.to(dtype=y.dtype)
    return y * mask
