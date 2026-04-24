import torch
import torch.nn as nn
from transformers import AutoModel
from layers.mlp import MLP
from models.xllm.modeling_internvl_chat import InternVLChatModel


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.token_len = configs.token_len
        if configs.use_multi_gpu:
            self.device = f"cuda:{configs.local_rank}"
        else:
            self.device = f"cuda:{configs.gpu}"
        print(self.device)
        
        self.internvl = InternVLChatModel.from_pretrained(
            configs.llm_ckp_dir,
            torch_dtype=torch.float16 if configs.use_amp else torch.float32,
            use_flash_attn=True,
            device_map=self.device,
        )
        
        if hasattr(configs, 'context_channels') and configs.context_channels != 3:
            num_channels = len(configs.context_channels)
            
            self.internvl.config.vision_config.num_channels = num_channels
            
            embed_dim = self.internvl.vision_model.embeddings.embed_dim
            patch_size = self.internvl.vision_model.embeddings.patch_size
            self.internvl.vision_model.embeddings.patch_embedding = nn.Conv2d(
                in_channels=num_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size
            ).to(self.device)

        self.hidden_dim_of_internvl = self.internvl.config.llm_config.hidden_size
        self.mix = configs.mix_embeds
        if self.mix:
            self.add_scale = nn.Parameter(torch.ones([]))
        
        for name, param in self.internvl.named_parameters():
            if 'patch_embedding' in name or 'mlp1' in name:
                param.requires_grad = True
                param.data = param.data.float()
            else:
                param.requires_grad = False

        if configs.mlp_hidden_layers == 0:
            if not configs.use_multi_gpu or (configs.use_multi_gpu and configs.local_rank == 0):
                print("use linear as tokenizer and detokenizer")
            self.encoder = nn.Linear(self.token_len, self.hidden_dim_of_internvl)
            self.decoder = nn.Linear(self.hidden_dim_of_internvl, self.token_len)
        else:
            if not configs.use_multi_gpu or (configs.use_multi_gpu and configs.local_rank == 0):
                print("use mlp as tokenizer and detokenizer")
            self.encoder = MLP(self.token_len, self.hidden_dim_of_internvl, 
                            configs.mlp_hidden_dim, configs.mlp_hidden_layers, 
                            configs.dropout, configs.mlp_activation)
            self.decoder = MLP(self.hidden_dim_of_internvl, self.token_len,
                            configs.mlp_hidden_dim, configs.mlp_hidden_layers,
                            configs.dropout, configs.mlp_activation)
    
    def merge_patches_to_big_images(self, t, H_big=448, W_big=448):
        B, N, C, H, W = t.shape
        
        patches_per_row = W_big // W
        patches_per_col = H_big // H
        patches_per_big = patches_per_row * patches_per_col

        assert N % patches_per_big == 0, \
            f"{N} patches cannot form full big images of size {patches_per_big}"

        t = t.reshape(B, -1, patches_per_big, C, H, W)
        t = t.reshape(B, -1, patches_per_col, patches_per_row, C, H, W)
        t = t.permute(0, 1, 4, 2, 5, 3, 6).reshape(
            B, -1, C, H_big, W_big
        )

        return t.contiguous()

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, pixel_values, input_ids):
        means = x_enc.mean(1, keepdim=True).detach()    
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        bs, _, n_vars = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1)
        x_enc = x_enc.reshape(x_enc.shape[0] * x_enc.shape[1], -1)
        fold_out = x_enc.unfold(dimension=-1, size=self.token_len, step=self.token_len)
        token_num = fold_out.shape[1]
        times_embeds = self.encoder(fold_out)
        if self.mix and x_mark_enc is not None:
            times_embeds = times_embeds / times_embeds.norm(dim=2, keepdim=True)
            x_mark_enc = x_mark_enc / x_mark_enc.norm(dim=2, keepdim=True)
            times_embeds = times_embeds + self.add_scale * x_mark_enc

        pixel_values = self.merge_patches_to_big_images(pixel_values)
        outputs = self.internvl(
            pixel_values=pixel_values,
            times_embeds=times_embeds,
            input_ids=input_ids,
            output_hidden_states=True,
        ).hidden_states

        dec_out = self.decoder(outputs)
        dec_out = dec_out.reshape(bs, n_vars, -1)
        dec_out = dec_out.permute(0, 2, 1)
        
        dec_out = dec_out * \
            (stdev[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))
        dec_out = dec_out + \
            (means[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))
        
        return dec_out
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, pixel_values, input_ids):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, pixel_values, input_ids)

