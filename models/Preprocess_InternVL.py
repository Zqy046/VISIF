import torch
import torch.nn as nn

from models.xllm.configuration_internvl_chat import InternVLChatConfig
from models.xllm.modeling_internvl_chat import InternVLChatModel


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        if configs.use_multi_gpu:
            self.device = f"cuda:{configs.local_rank}"
        else:
            self.device = f"cuda:{configs.gpu}"
        print(self.device)
        
        self.internvl = InternVLChatModel.from_pretrained(
            configs.llm_ckp_dir,
            torch_dtype=torch.float16,
            use_flash_attn=True,
            device_map=self.device,
        )
        self.internvl_tokenizer = None

        for name, param in self.internvl.named_parameters():
            param.requires_grad = False

    def tokenizer(self, x):
        input_ids = self.internvl_tokenizer(x, return_tensors="pt")['input_ids'].to(self.device)
        return input_ids   
    
    def forecast(self, text_prompts, images):
        """
        Process text prompts and images to generate embeddings.
        
        Args:
            text_prompts: List of text strings like "This is Time Series from {start_time} to {end_time}"
            images: List of PIL Images or torch.Tensor images corresponding to the time range
        
        Returns:
            embeddings: [batch_size, hidden_dim] tensor
        """
        text_prompts = torch.cat([self.tokenizer(text_prompts[i]) for i in range(len(text_prompts))], 0)
        with torch.no_grad():
            outputs = self.internvl(input_ids=text_prompts)
            text_outputs = outputs.hidden_states[:, -1, :]
        
        return text_outputs
    
    def forward(self, text_prompts, images=None):
        return self.forecast(text_prompts, images)

