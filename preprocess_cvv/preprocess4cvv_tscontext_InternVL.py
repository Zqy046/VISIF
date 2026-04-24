# Jupyter cell
import warnings
warnings.filterwarnings('ignore', message='.*convert_dtype.*', category=FutureWarning)

import torch
import numpy as np
import argparse
import datetime
import pandas as pd
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from data_provider.tscontext_dataset import TSContextDataset
from models.Preprocess_InternVL import Model
from models.xllm.tokenization_internlm2 import InternLM2Tokenizer


class TSContextDataset_Preprocess(Dataset):
    def __init__(self, tscontext_dataset, token_len=8):
        self.tscontext_dataset = tscontext_dataset
        self.token_len = token_len
        
    def __getitem__(self, index):
        sample = self.tscontext_dataset[index]
        
        time_utc_list = sample["time_utc"]
        start_time = time_utc_list[0]
        start = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        end_time = (start + datetime.timedelta(minutes=30*(self.token_len-1))).strftime('%Y-%m-%d %H:%M:%S')
        seq_x_mark = f"This is Time Series from {start_time} to {end_time}"

        image_pil = None
                    
        return seq_x_mark, time_utc_list[0]
    
    def __len__(self):
        return len(self.tscontext_dataset)


parser = argparse.ArgumentParser(description='AutoTimes Preprocess with InternVL')
parser.add_argument('--gpu', type=int, default=0, help='gpu id')
parser.add_argument('--llm_ckp_dir', type=str, 
                    default='',
                    help='InternVL checkpoints dir')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--local_rank', type=int, default=0, help='local rank for multi-gpu')
# parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision', default=False)
parser.add_argument('--dataset_time_range', type=str, required=True,
                    help='dataset time range, e.g., 20082022')
parser.add_argument('--site', type=str, required=True,
                    help='dataset from site to preprocess, options:[IZA, CNR, PAL]')
parser.add_argument('--save_dir_path', type=str, 
                    default='./data_stamp/',
                    help='directory path to save the processed results')
args = parser.parse_args()
print(args.dataset_time_range)

model = Model(args)
tokenizer = InternLM2Tokenizer.from_pretrained(args.llm_ckp_dir, use_fast=False)
special_tokens_dict = {'additional_special_tokens': ['<TIME_SERIES>']}
tokenizer.add_special_tokens(special_tokens_dict)
model.internvl.img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
model.internvl.ts_context_token_id = tokenizer.convert_tokens_to_ids('<TIME_SERIES>')
num_image_token = model.internvl.num_image_token
model.internvl_tokenizer = tokenizer

data_dir = ''
stats_path = 'stats_multi.json'

dataset_kwargs = {
    'data_dir': data_dir,
    'stats_path': stats_path,
    'context_channels': ['IR_039', 'IR_087', 'IR_108', 'VIS006', 'VIS008', 'WV_062', 'WV_073'],
    'optflow_channels': ['IR_039_vx', 'IR_039_vy', 'IR_087_vx', 'IR_087_vy', 'IR_108_vx', 'IR_108_vy', 'WV_062_vx', 'WV_062_vy', 'WV_073_vx', 'WV_073_vy'],
    'ts_channels': ['GHI', 'DIF [W/m**2]', 'DIR [W/m**2]', 'PoPoPoPo [hPa]', 'dhi', 'dni', 'ghi'],
    'ts_target_channels': ['GHI'],
    'years': {
        'train': [
            '2008_nonhrv', '2009_nonhrv', '2010_nonhrv', '2011_nonhrv', '2012_nonhrv',
            '2013_nonhrv', '2014_nonhrv', '2015_nonhrv', '2016_nonhrv', '2017_nonhrv',
            '2018_nonhrv', '2019_nonhrv', '2020_nonhrv', '2021_nonhrv', '2022_nonhrv',
        ],
    },
    'stations': {
        'train': [f'PCCI_20082022_{args.site}'],
    },
    'image_size': (112, 112),
    'crop': None,
    'seq_len': 1,
    'label_len': 0,
    'pred_len': 0,
    'use_target': True,
    'token_len': 8,
    'tokenizer': tokenizer,
    'num_image_token': num_image_token,
}

tscontext_dataset = TSContextDataset(**dataset_kwargs, mode='train')

token_len = 8

data_set = TSContextDataset_Preprocess(
    tscontext_dataset=tscontext_dataset,
    token_len=token_len
)

data_loader = DataLoader(
    data_set,
    batch_size=128,
    shuffle=False,
    num_workers=8,
)

from tqdm import tqdm

save_dir_path = args.save_dir_path
os.makedirs(save_dir_path, exist_ok=True)

tensor_list = []
timestamp_list = []
for idx, (seq_x_batch, x_mark_batch) in tqdm(enumerate(data_loader), desc="Processing"):
    output = model(seq_x_batch)
    tensor_list.append(output.detach().cpu())
    timestamp_list.extend(x_mark_batch)

if tensor_list:
    result = torch.cat(tensor_list, dim=0)
    assert result.shape[0] == len(timestamp_list), f"Mismatch: result.shape[0]={result.shape[0]}, len(timestamp_list)={len(timestamp_list)}"
    print(f"Shape of the result: {result.shape}")
    torch.save(result, save_dir_path + f'/cvv_timestamps_forecast_{args.site}_{args.dataset_time_range}.pt')
    print(f"Saved to: {save_dir_path}/cvv_timestamps_forecast_{args.site}_{args.dataset_time_range}.pt")

    timestamp_save_path = save_dir_path + f'/cvv_timestamps_forecast_{args.site}_{args.dataset_time_range}.json'
    with open(timestamp_save_path, 'w', encoding='utf-8') as f:
        json.dump(timestamp_list, f, ensure_ascii=False, indent=2)
    print(f"Saved to: {timestamp_save_path}")
else:
    print("No data processed.")

