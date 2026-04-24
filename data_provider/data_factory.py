from data_provider.data_loader import Dataset_ETT_hour, Dataset_Custom, Dataset_M4, Dataset_Solar, Dataset_TSF, Dataset_TSF_ICL
from data_provider.forecast_dataset import TSDataset
from data_provider.tscontext_dataset import TSContextDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'Solar': Dataset_Solar,
    'tsf': Dataset_TSF,
    'tsf_icl': Dataset_TSF_ICL,
    'cvv': TSDataset,
    'cvv_tscontext': TSContextDataset
}


def data_provider(args, flag, tokenizer=None, num_image_token=0):
    Data = data_dict[args.data]

    if flag in ['test', 'test1', 'test2', 'test3']:
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size 
    elif flag == 'val':
        shuffle_flag = args.val_set_shuffle
        drop_last = False
        batch_size = args.batch_size 
    else:
        shuffle_flag = True
        drop_last = args.drop_last
        batch_size = args.batch_size

    # CVV dataset with context (images) uses TSContextDataset
    if args.data == 'cvv_tscontext':
        if flag in ['train', 'val']:
            data_set = Data(
                data_dir=args.data_dir,
                stats_path=args.stats_path,
                context_channels=args.context_channels,
                optflow_channels=args.optflow_channels,
                ts_channels=args.ts_channels,
                ts_target_channels=args.ts_target_channels,
                years=args.years,
                stations=args.stations,
                mode=flag,
                seq_len=args.seq_len,
                label_len=args.label_len,
                pred_len=args.token_len,
                image_size=args.image_size,
                crop=args.crop,
                use_target=True,
                timestamp_dir=args.timestamp_dir,
                timestamp_range=args.timestamp_range if hasattr(args, 'timestamp_range') else "20082022",
                token_len=args.token_len,
                tokenizer=tokenizer,
                num_image_token=num_image_token,
            )
        else:
            data_set = Data(
                data_dir=args.data_dir,
                stats_path=args.stats_path,
                context_channels=args.context_channels,
                optflow_channels=args.optflow_channels,
                ts_channels=args.ts_channels,
                ts_target_channels=args.ts_target_channels,
                years=args.years,
                stations=args.stations,
                mode=flag,
                seq_len=args.test_seq_len,
                label_len=args.test_label_len,
                pred_len=args.test_pred_len,
                image_size=args.image_size,
                crop=args.crop,
                use_target=True,
                timestamp_dir=args.timestamp_dir,
                timestamp_range=args.timestamp_range if hasattr(args, 'timestamp_range') else "20082022",
                token_len=args.token_len,
                tokenizer=tokenizer,
                num_image_token=num_image_token,
            )
    # CVV dataset uses different parameters
    elif args.data == 'cvv':
        if flag in ['train', 'val']:
            data_set = Data(
                data_dir=args.data_dir,
                stats_path=args.stats_path,
                ts_channels=args.ts_channels,
                years=args.years,
                stations=args.stations,
                mode=flag,
                seq_len=args.seq_len,
                label_len=args.label_len,
                pred_len=args.token_len,
                freq=args.freq,
                time_encoding=args.time_encoding,
                timestamp_dir=args.timestamp_dir,
                token_len=args.token_len,
            )
        else:
            data_set = Data(
                data_dir=args.data_dir,
                stats_path=args.stats_path,
                ts_channels=args.ts_channels,
                years=args.years,
                stations=args.stations,
                mode=flag,
                seq_len=args.test_seq_len,
                label_len=args.test_label_len,
                pred_len=args.test_pred_len,
                freq=args.freq,
                time_encoding=args.time_encoding,
                timestamp_dir=args.timestamp_dir,
                token_len=args.token_len,
            )
    else:
        # Standard datasets
        if flag in ['train', 'val']:
            data_set = Data(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.token_len],
                seasonal_patterns=args.seasonal_patterns,
                drop_short=args.drop_short,
            )
        else:
            data_set = Data(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.test_seq_len, args.test_label_len, args.test_pred_len],
                seasonal_patterns=args.seasonal_patterns,
                drop_short=args.drop_short,
            )
    if (args.use_multi_gpu and args.local_rank == 0) or not args.use_multi_gpu:
        print(flag, len(data_set))
    
    # For test sets, do not use DistributedSampler to avoid data splitting
    # Each process should see the full test dataset
    if args.use_multi_gpu and flag not in ['test', 'test1', 'test2', 'test3']:
        train_datasampler = DistributedSampler(data_set, shuffle=shuffle_flag)
        data_loader = DataLoader(data_set, 
            batch_size=batch_size,
            sampler=train_datasampler,
            num_workers=args.num_workers,
            persistent_workers=True,
            pin_memory=True,
            drop_last=drop_last,
            )
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
    return data_set, data_loader