from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import json
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

warnings.filterwarnings('ignore')


class Exp_CVV_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_CVV_Forecast, self).__init__(args)
        
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args)
        if self.args.use_multi_gpu:
            self.device = torch.device('cuda:{}'.format(self.args.local_rank))
            model = DDP(model.cuda(), device_ids=[self.args.local_rank])
        else:
            self.device = self.args.gpu
            model = model.to(self.device)
        return model

    def _get_data(self, flag):
        """
        Get dataset and dataloader for a specific flag.
        For CVV dataset, supports multiple test sets (test1, test2, test3).
        """
        if flag == 'test' and self.args.data == 'cvv':
            # Support multiple test sets for CVV dataset
            test_sets = []
            test_loaders = []
            for test_flag in ['test1', 'test2', 'test3']:
                data_set, data_loader = data_provider(self.args, test_flag)
                test_sets.append(data_set)
                test_loaders.append(data_loader)
            return test_sets, test_loaders
        else:
            data_set, data_loader = data_provider(self.args, flag)
            return data_set, data_loader

    def _select_optimizer(self):
        p_list = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            else:
                p_list.append(p)
                if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                    print(n, p.dtype, p.shape)
        model_optim = optim.Adam([{'params': p_list}], lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
            print('next learning rate is {}'.format(self.args.learning_rate))
        return model_optim

    def _select_criterion(self):
        criterion = nn.L1Loss()
        return criterion

    def prepare_batch(self, batch):
        """
        Prepare batch data from dictionary format to model input format.
        Extract GHI channel only for AutoTimes_Llama model.
        """
        ghi_id = batch["ghi"].long()[0].item()
        x_ts = batch["x"][..., ghi_id].unsqueeze(-1).float()
        y_ts = batch["y"][..., ghi_id].unsqueeze(-1).float()
        x_timestamp = batch["x_timestamp"]
        y_timestamp = batch["y_timestamp"]
        return x_ts, y_ts, x_timestamp, y_timestamp, ghi_id

    def vali(self, vali_data, vali_loader, criterion, is_test=False):
        total_loss = []
        total_count = []
        time_now = time.time()
        test_steps = len(vali_loader)
        iter_count = 0
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                iter_count += 1
                batch_x, batch_y, batch_x_mark, batch_y_mark, ghi_id = self.prepare_batch(batch)
                
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.to(self.device)
                batch_y_mark = batch_y_mark.to(self.device)
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                
                if is_test:
                    outputs = outputs[:, -self.args.token_len:, :]
                    batch_y = batch_y[:, -self.args.token_len:, :].to(self.device)
                else:
                    outputs = outputs[:, -self.args.token_len:, :]
                    batch_y = batch_y[:, -self.args.token_len:, :].to(self.device)

                loss = criterion(outputs, batch_y)

                loss = loss.detach().cpu()
                total_loss.append(loss)
                total_count.append(batch_x.shape[0])
                if (i + 1) % 100 == 0:
                    if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * (test_steps - i)
                        print("\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(i + 1, speed, left_time))
                        iter_count = 0
                        time_now = time.time()
        if self.args.use_multi_gpu:
            total_loss = torch.tensor(np.average(total_loss, weights=total_count)).to(self.device)
            dist.barrier()   
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss = total_loss.item() / dist.get_world_size()
        else:
            total_loss = np.average(total_loss, weights=total_count)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
            if not os.path.exists(path):
                os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(self.args, verbose=True)
        
        model_optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)
        criterion = self._select_criterion()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0

            loss_val = torch.tensor(0., device="cuda")
            count = torch.tensor(0., device="cuda")
            
            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x, batch_y, batch_x_mark, batch_y_mark, ghi_id = self.prepare_batch(batch)
                
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_x_mark = batch_x_mark.to(self.device)
                batch_y_mark = batch_y_mark.to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                        outputs = outputs[:, -self.args.token_len:, :]
                        batch_y = batch_y[:, -self.args.token_len:, :]
                        loss = criterion(outputs, batch_y)                        
                        loss_val += loss.item()
                        count += 1
                else:
                    outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                    outputs = outputs[:, -self.args.token_len:, :]
                    batch_y = batch_y[:, -self.args.token_len:, :]
                    loss = criterion(outputs, batch_y)
                    loss_val += loss.item()
                    count += 1
                
                if (i + 1) % 100 == 0:
                    if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
            if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))   
            if self.args.use_multi_gpu:
                dist.barrier()   
                dist.all_reduce(loss_val, op=dist.ReduceOp.SUM)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)      
            train_loss = loss_val.item() / count.item()

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            
            if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                print("Epoch: {}, Steps: {} | Train Loss: {:.7f} Vali Loss: {:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss))
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                    print("Early stopping")
                break
            if self.args.cosine:
                scheduler.step()
                if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                    print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            if self.args.use_multi_gpu:
                train_loader.sampler.set_epoch(epoch + 1)
                
        best_model_path = path + '/' + 'checkpoint.pth'
        if self.args.use_multi_gpu:
            dist.barrier()
            self.model.load_state_dict(torch.load(best_model_path), strict=False)
        else:
            self.model.load_state_dict(torch.load(best_model_path), strict=False)
        return self.model

    def test(self, setting, test=0):
        self.test_results = {"IZA":{}, "CNR":{}, "PAL":{}}
        test_data, test_loader = self._get_data(flag='test')

        print("info:", self.args.test_seq_len, self.args.test_label_len, self.args.token_len, self.args.test_pred_len)
        if test:
            print('loading model')
            setting = self.args.test_dir
            best_model_path = self.args.test_file_name

            print("loading model from {}".format(os.path.join(self.args.checkpoints, setting, best_model_path)))
            load_item = torch.load(os.path.join(self.args.checkpoints, setting, best_model_path))
            self.model.load_state_dict({k.replace('module.', ''): v for k, v in load_item.items()}, strict=False)

        # Handle multiple test sets for CVV dataset
        if self.args.data == 'cvv' and isinstance(test_loader, list):
            test_names = ['test1', 'test2', 'test3']
            # Mapping from test_name to station name
            test_to_station = {'test1': 'IZA', 'test2': 'CNR', 'test3': 'PAL'}
            for idx, (test_data_single, test_loader_single) in enumerate(zip(test_data, test_loader)):
                print(f"\n{'='*50}")
                print(f"Testing on {test_names[idx]}")
                print(f"{'='*50}")
                station_name = test_to_station[test_names[idx]]
                self._test_single(test_loader_single, setting, test_names[idx], station_name)
        else:
            self._test_single(test_loader, setting, 'test', 'test')
        
        # Save JSON results
        if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
            if self.args.data == 'cvv':
                file_path = f"./test_results/{setting}"
                os.makedirs(file_path, exist_ok=True)

                import re
                training_station_name = ""
                station_names = ['IZA', 'CNR', 'PAL']
                for name in station_names:
                    if name in setting.split('_'):
                        training_station_name = name
                        break
                if "Llama" in setting:
                    filename = os.path.join(file_path, f"test_Llama{training_station_name}.json")
                elif re.search(r'InternLM', setting, re.IGNORECASE):
                    # Find all matches starting with internlm till the end or next separator
                    matches = re.findall(r'(internlm[^/\\]*)', setting, flags=re.IGNORECASE)
                    if matches:
                        internlm_str = matches[-1]
                        filename = os.path.join(file_path, f"test_{internlm_str}{training_station_name}.json")
                else:
                    filename = os.path.join(file_path, "test_results.json")
                with open(filename, "w") as f:
                    json.dump(self.test_results, f, indent=4)
                print(f"\nTest results saved to {filename}")
    
    def _test_single(self, test_loader, setting, test_name='test', station_name='test'):
        """Test on a single test set"""
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/' + test_name + '/'
        if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        time_now = time.time()
        test_steps = len(test_loader)
        iter_count = 0
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                iter_count += 1
                batch_x, batch_y, batch_x_mark, batch_y_mark, ghi_id = self.prepare_batch(batch)
                
                # Store original batch data for JSON saving (only on rank 0)
                if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                    if self.args.data == 'cvv' and station_name in ['IZA', 'CNR', 'PAL']:
                        # Get idx, time_utc, and input timeseries from batch
                        idx = batch["idx"].cpu().detach().tolist() if "idx" in batch else list(range(i * batch_x.shape[0], (i + 1) * batch_x.shape[0]))
                        time_utc = batch["time_utc"] if "time_utc" in batch else None
                        # Fix time_utc format: DataLoader's default collate transposes lists
                        # If time_utc is [seq_len, batch_size], transpose it to [batch_size, seq_len]
                        if time_utc is not None and len(time_utc) > 0:
                            test_seq_len = len(time_utc)
                            batch_size = len(time_utc[0]) if isinstance(time_utc[0], list) else len(idx)
                            time_utc = [[time_utc[j][i] for j in range(test_seq_len)] for i in range(batch_size)]
                        # Get input timeseries (GHI channel) from timseries_data_wo_mean
                        if "timseries_data_wo_mean" in batch:
                            input_timeseries = batch["timseries_data_wo_mean"][..., ghi_id].cpu().detach().tolist()
                        else:
                            # Fallback: use batch_x (already extracted GHI channel)
                            input_timeseries = batch_x.cpu().detach().squeeze(-1).tolist()
                
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_x_mark = batch_x_mark.to(self.device)
                batch_y_mark = batch_y_mark.to(self.device)

                # Autoregressive multi-step prediction
                inference_steps = self.args.test_pred_len // self.args.token_len
                dis = self.args.test_pred_len - inference_steps * self.args.token_len
                if dis != 0:
                    inference_steps += 1
                pred_y = []
                for j in range(inference_steps):
                    if len(pred_y) != 0:
                        # Slide window: use previous prediction to extend input
                        batch_x = torch.cat([batch_x[:, self.args.token_len:, :], pred_y[-1]], dim=1)
                        tmp = batch_y_mark[:, j-1:j, :]
                        batch_x_mark = torch.cat([batch_x_mark[:, 1:, :], tmp], dim=1)
                        
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                    pred_y.append(outputs[:, -self.args.token_len:, :])
                
                pred_y = torch.cat(pred_y, dim=1)
                if dis != 0:
                    pred_y = pred_y[:, :-(self.args.token_len - dis), :]
                batch_y = batch_y[:, -self.args.test_pred_len:, :].to(self.device)
                outputs = pred_y.detach().cpu()
                batch_y = batch_y.detach().cpu()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                
                # Save to test_results for JSON (only on rank 0)
                if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                    if self.args.data == 'cvv' and station_name in ['IZA', 'CNR', 'PAL']:
                        # Convert predictions and ground truth to lists
                        # pred and true are already on CPU and detached
                        if pred.dim() == 3:
                            y_hat = pred.squeeze(-1).tolist()  # [batch_size, pred_len]
                        elif pred.dim() == 2:
                            y_hat = pred.tolist()
                        else:
                            y_hat = [pred.item()] if pred.numel() == 1 else pred.tolist()
                        
                        if true.dim() == 3:
                            y_ts = true.squeeze(-1).tolist()  # [batch_size, pred_len]
                        elif true.dim() == 2:
                            y_ts = true.tolist()
                        else:
                            y_ts = [true.item()] if true.numel() == 1 else true.tolist()

                        # Store results for each sample in the batch
                        for batch_idx in range(len(idx)):
                            sample_idx = idx[batch_idx]
                            
                            # Get predictions for this sample
                            assert len(y_ts) > batch_idx or len(input_timeseries) > batch_idx
                            if isinstance(y_hat, list):
                                pred_list = y_hat[batch_idx] if isinstance(y_hat[batch_idx], list) else [y_hat[batch_idx]]
                            if isinstance(y_ts, list):
                                gt_list = y_ts[batch_idx] if isinstance(y_ts[batch_idx], list) else [y_ts[batch_idx]]
                            if isinstance(input_timeseries, list):
                                input_ts_list = input_timeseries[batch_idx] if isinstance(input_timeseries[batch_idx], list) else [input_timeseries[batch_idx]]
                            sample_time_utc = time_utc[batch_idx] if isinstance(time_utc[batch_idx], list) else []

                            
                            self.test_results[station_name][sample_idx] = {
                                "predictions": pred_list,
                                "ground_truth": gt_list,
                                "input_timeseries": input_ts_list,
                                "time_utc": sample_time_utc
                            }
                
                if (i + 1) % 100 == 0:
                    if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * (test_steps - i)
                        print("\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(i + 1, speed, left_time))
                        iter_count = 0
                        time_now = time.time()
                    
                if self.args.visualize and i == 0:
                    gt = np.array(true[0, :, -1])
                    pd = np.array(pred[0, :, -1])
                    lookback = batch_x[0, :, -1].detach().cpu().numpy()
                    gt = np.concatenate([lookback, gt], axis=0)
                    pd = np.concatenate([lookback, pd], axis=0)
                    dir_path = folder_path + f'{self.args.test_pred_len}/'
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    visual(gt, pd, os.path.join(dir_path, f'{i}.png'))
        
        # Only process and output results on rank 0 in multi-gpu mode
        if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
            preds = torch.cat(preds, dim=0).numpy()
            trues = torch.cat(trues, dim=0).numpy()
            
            mae, mse, rmse, mape, mspe = metric(preds, trues)
            print(f'\n{test_name} Results:')
            print(f'mse: {mse:.4f}, mae: {mae:.4f}, rmse: {rmse:.4f}, mape: {mape:.4f}, mspe: {mspe:.4f}')
            
            f = open("result_long_term_forecast.txt", 'a')
            f.write(f"{setting} - {test_name}\n")
            f.write(f'mse: {mse:.4f}, mae: {mae:.4f}, rmse: {rmse:.4f}, mape: {mape:.4f}, mspe: {mspe:.4f}\n')
            f.write('\n')
            f.close()
        
        # Synchronize all processes before returning
        if self.args.use_multi_gpu:
            dist.barrier()
        return
