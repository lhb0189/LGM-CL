import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
import transformers
import torch.nn as nn
from peft import LoraConfig,get_peft_model
class LoraMistralModel(nn.Module):
    def __init__(self,local_model_path,use_lora):
        super().__init__()
        self.local_model_path = local_model_path
        self.model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            torch_dtype=torch.float16,
            device_map=None
        ).to("cuda")
        self.tokenizer=AutoTokenizer.from_pretrained(local_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lora_config=LoraConfig(
            r=4,
            lora_alpha=16,
            target_modules=["q_proj","v_proj"],
            lora_dropout=0.05
        )
        if use_lora:
            self.peft_model=get_peft_model(self.model,self.lora_config)
        else:
            self.peft_model=self.model
            for param in self.peft_model.parameters():
                param.requires_grad=False
    def template_generate(self,prompts,max_new_tokens,temperature,top_p):
        self.peft_model.eval()
        single_input = isinstance(prompts, str)
        if single_input:
            prompt_list = [prompts]
        else:
            prompt_list = prompts
        inputs=self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        ).to(self.peft_model.device)
        with torch.no_grad():
            output_ids=self.peft_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        new_texts=[]
        for i,ids in enumerate(output_ids):
            full_text=self.tokenizer.decode(ids,skip_special_tokens=True)
            p=prompt_list[i]
            idx=full_text.find(p)
            if idx != -1:
                start = idx + len(p)
                new_part = full_text[start:]
            else:
                new_part = full_text
            new_texts.append(new_part.strip())
        result = new_texts[0] if single_input else new_texts
        return result
    def forward(self, prompt, **generate_kwargs):
        return self.template_generate(prompt, **generate_kwargs)
