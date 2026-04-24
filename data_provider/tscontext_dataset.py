import json
import os
from typing import Dict, List, Tuple, Union

import deeplake
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from data_provider.dataset_utils import calculate_possible_starts

"""
This prevents the following error occuring from the interaction between deeplake and wandb:
wandb.errors.UsageError: problem
"""
deeplake.constants.WANDB_INTEGRATION_ENABLED = False



def build_transform(input_size):
    # MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        # T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        # T.ToTensor(),
        # T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image, input_size=448, max_num=12):
    # image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def build_transform_tensor(input_size):
    """
    Tensor version of build_transform - only does resize using F.interpolate.
    Returns a function that takes a tensor [C, H, W] and returns [C, input_size, input_size].
    """
    def transform_tensor(img_tensor):
        # img_tensor: [C, H, W]
        # Resize to (input_size, input_size) using bicubic interpolation
        img_tensor = img_tensor.unsqueeze(0)  # [1, C, H, W]
        img_tensor = F.interpolate(
            img_tensor, 
            size=(input_size, input_size), 
            mode='bicubic', 
            align_corners=False
        )
        img_tensor = img_tensor.squeeze(0)  # [C, input_size, input_size]
        return img_tensor
    return transform_tensor

def dynamic_preprocess_tensor(img_tensor, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """
    Tensor version of dynamic_preprocess.
    Args:
        img_tensor: [C, H, W] - normalized tensor (already in [0, 1] range)
    Returns:
        List of tensor tiles: [tile1, tile2, ...], each tile is [C, image_size, image_size]
    """
    C, orig_height, orig_width = img_tensor.shape
    aspect_ratio = orig_width / orig_height
    
    # Calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) 
        for i in range(1, n + 1) for j in range(1, n + 1) 
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    
    # Find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    
    # Calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    
    # Resize the image using F.interpolate
    img_tensor_4d = img_tensor.unsqueeze(0)  # [1, C, H, W]
    resized_img = F.interpolate(
        img_tensor_4d,
        size=(target_height, target_width),
        mode='bicubic',
        align_corners=False
    )
    resized_img = resized_img.squeeze(0)  # [C, target_height, target_width]
    
    # Split the image into tiles
    processed_images = []
    num_cols = target_width // image_size
    num_rows = target_height // image_size
    
    for i in range(blocks):
        row = i // num_cols
        col = i % num_cols
        # Extract tile: [C, image_size, image_size]
        tile = resized_img[
            :,
            row * image_size:(row + 1) * image_size,
            col * image_size:(col + 1) * image_size
        ]
        processed_images.append(tile)
    
    assert len(processed_images) == blocks
    
    # Add thumbnail if needed
    if use_thumbnail and len(processed_images) != 1:
        # Use original img_tensor for thumbnail
        thumbnail = F.interpolate(
            img_tensor_4d,
            size=(image_size, image_size),
            mode='bicubic',
            align_corners=False
        ).squeeze(0)  # [C, image_size, image_size]
        processed_images.append(thumbnail)
    
    return processed_images

def load_image_tensor(img_tensor, input_size=448, max_num=12):
    """
    Tensor version of load_image.
    Args:
        img_tensor: [C, H, W] - normalized tensor (already in [0, 1] range)
        input_size: Target size for each tile
        max_num: Maximum number of tiles
    Returns:
        pixel_values: [num_tiles, C, input_size, input_size]
    """
    transform_tensor = build_transform_tensor(input_size=input_size)
    images = dynamic_preprocess_tensor(img_tensor, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform_tensor(img) for img in images]
    pixel_values = torch.stack(pixel_values)  # [num_tiles, C, input_size, input_size]
    return pixel_values

class TSContextDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        stats_path: str,
        context_channels: Union[Tuple[str], List[str]],
        optflow_channels: Union[Tuple[str], List[str]],
        ts_channels: Union[Tuple[str], List[str]],
        ts_target_channels: Union[Tuple[str], List[str]],
        years: Dict[str, Union[int, str]],
        stations: Dict[str, str],
        mode: str = "train",
        use_target: bool = True,
        image_size: Tuple[int] = None,
        crop: Tuple[int] = None,
        seq_len: int = 24 * 2,
        label_len: int = 12,
        pred_len: int = 24 * 2,
        timestamp_dir: str = "",
        timestamp_range: str = "20082022",
        token_len: int = 0,
        tokenizer=None,
        num_image_token: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        data_dir : str
            Absolute directory path where the deeplake dataset is located.
        stats_path : str
            Absolute directory path where the stats json file is located.
        context_channels: Union(Tuple[str], List[str])
            Selects the provided channels for the context as input. If ``None`` is given,
            then it takes all available channels.
        optflow_channels: Union(Tuple[str], List[str])
            Selects the provided channels for the optical flow as input. If ``None`` is given,
            then it takes all available channels.
        ts_channels: Union(Tuple[str], List[str])
            Selects the provided channels for the timeseries as input. If ``None`` is given,
            then it takes all available channels.
        ts_target_channels: Union(Tuple[str], List[str])
            Selects the provided channels for the timeseries as output. If ``None`` is given,
            then it takes all available channels.
        years: Dict[str, Union[int, str]]
            Dictionary containing for each mode, which years to select.
        stations: Dict[str, str]
            Dictionary containing for each mode, which stations to select.
        mode : str
            Indicates which dataset is it: train, val or test.
        use_target: bool
            Indicates whether to output target or not. In case it's not output, the input timeseries will be 2*n_steps long.
        image_size : Tuple[int]
            Interpolate to desired image size. If set to None, no interpolation is done
            and the size is kept as is.
        crop: Tuple[int]
            If not None, ``crop`` is expected to be in the following format:
                (lat_upper_left, lon_upper_left, lat_bottom_right, lon_bottom_right)
            And the context is cropped to the given parameters. Note that the context is first resized
            using ``image_size`` argument and then cropped.
        seq_len: int
            Number of frames in the input sequence.
        label_len: int
            Number of frames in the label sequence.
        pred_len: int
            Number of frames in the prediction sequence.
        """

        self.data_dir = data_dir
        self.stats_path = stats_path
        self.context_channels = context_channels
        self.optflow_channels = optflow_channels
        self.ts_channels = ts_channels
        self.ts_target_channels = ts_target_channels
        self.years = years[mode]
        self.stations = stations[mode]
        self.crop = crop
        self.mode = mode
        self.image_size = tuple(image_size) if image_size is not None else None
        self.use_target = use_target
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.timestamp_dir = timestamp_dir
        self.timestamp_range = timestamp_range
        self.token_len = token_len
        self.tokenizer = tokenizer
        self.input_ids = None
        self.num_image_token = num_image_token
        self.n_samples = []
        self.year_mapping = {}

        if self.stats_path is not None:
            with open(self.stats_path, "r") as fp:
                self.stats = json.load(fp)
        else:
            self.stats = None
        self.deeplake_ds = deeplake.load(self.data_dir, read_only=True)

        for year in self.years:
            for station in self.stations:
                print(f"{year} {station}")
                (
                    possible_starts_context,
                    possible_starts_station,
                ) = calculate_possible_starts(
                    self.deeplake_ds[f"{year}/context/time_utc"].numpy()[0],
                    self.deeplake_ds[f"{year}/{station}/time_utc"].numpy()[0],
                    frames_total=self.seq_len + self.pred_len,
                )
                n_in_group = len(possible_starts_context)
                self.n_samples.append(n_in_group)

                for i in range(self.n_samples[-1]):
                    step_id_station = possible_starts_station[i]
                    step_id_context = possible_starts_context[i]
                    self.year_mapping[i + sum(self.n_samples[:-1])] = (
                        str(year),
                        station,
                        step_id_station,
                        step_id_context,
                    )
        # Load timestamp embeddings if timestamp_dir is provided
        if self.timestamp_dir:
            self.timestamp_emb = {}
            for station in self.stations:
                timestamps_path = os.path.join(self.timestamp_dir,
                                              f"cvv_timestamps_forecast_{station.split('_')[-1]}_{self.timestamp_range}.json")
                timestamp_embs_path = os.path.join(self.timestamp_dir,
                                                  f"cvv_timestamps_forecast_{station.split('_')[-1]}_{self.timestamp_range}.pt")
                if os.path.exists(timestamps_path) and os.path.exists(timestamp_embs_path):
                    timestamps = json.load(open(timestamps_path))
                    timestamp_embs = torch.load(timestamp_embs_path)
                    timestamp_to_idx = {t: i for i, t in enumerate(timestamps)}
                    assert timestamp_embs.shape[0] == len(timestamps), f"Mismatch: timestamp_embs.shape[0]={timestamp_embs.shape[0]}, len(timestamps)={len(timestamps)}"
                    self.timestamp_emb[station] = {"timestamp_embs": timestamp_embs, "timestamp_to_idx": timestamp_to_idx}
                else:
                    print(f"Warning: Timestamp embedding files not found for station {station}")
                    self.timestamp_emb[station] = None
        else:
            self.timestamp_emb = {}
        
        self._coords = None
        self._elevation = None
        self._mean = {}
        self._std = {}
        self._context_channel_ids = None
        self._optflow_channel_ids = None
        self._ts_channel_ids = None
        self._ts_target_channel_ids = None
        self._lat_slice = None
        self._lon_slice = None
        print(f"Number of {mode} samples: {sum(self.n_samples)}")

    def get_stats(self, station):
        if station in self._mean:
            return self._mean[station], self._std[station]

        if self.stats is not None:
            mean = []
            std = []
            station_channels = (
                self.ts_channels
                if self.ts_channels is not None
                else self.stats[station]
            )
            for i, chan in enumerate(station_channels):
                mean.append(float(self.stats[station][chan]["mean"]))
                std.append(float(self.stats[station][chan]["std"]))
            mean = torch.tensor(mean).float().view(1, -1)
            std = torch.tensor(std).float().view(1, -1)
            self._mean[station] = mean
            self._std[station] = std
            return mean, std
        return None, None

    def get_coords(self, year):
        if self._coords is None:
            lat = self.deeplake_ds[f"{str(year)}/context/latitude"].numpy()

            lat = 2 * ((lat + 90) / 180) - 1
            lon = self.deeplake_ds[f"{str(year)}/context/longitude"].numpy()
            lon = 2 * ((lon + 180) / 360) - 1
            self._coords = np.stack(np.meshgrid(lat, lon), axis=0)
        return self._coords

    def get_elevation(self, year):
        if self._elevation is None:
            self._elevation = self.deeplake_ds[f"{str(year)}/context/elevation"].numpy()
        return self._elevation

    def get_channel_ids(
        self,
        context_tensor: deeplake.Tensor,
        optflow_tensor: deeplake.Tensor,
        timeseries_tensor: deeplake.Tensor,
    ):
        """
        Get the list of channel indices to use for the context, optical flow and timeseries.
        Args:
            context_tensor (deeplake.Tensor): Context tensor to extract channel ids from.
            optflow_tensor (deeplake.Tensor): Optical flow tensor to extract channel ids from.
            timeseries_tensor (deeplake.Tensor): Timeseries tensor to extract channel ids from.
        """
        if self._context_channel_ids is None:
            if self.context_channels is not None:
                self._context_channel_ids = [
                    i
                    for i, k in enumerate(context_tensor.info["context_channels"])
                    for c in self.context_channels
                    if c == k
                ]
            else:
                self._context_channel_ids = [
                    i for i, k in enumerate(context_tensor.info["context_channels"])
                ]
            self._context_channel_ids = sorted(self._context_channel_ids)

        if self._optflow_channel_ids is None:
            if self.optflow_channels is not None:
                self._optflow_channel_ids = [
                    i
                    for i, k in enumerate(optflow_tensor.info["optflow_channels"])
                    for c in self.optflow_channels
                    if c == k
                ]
            else:
                self._optflow_channel_ids = [
                    i for i, k in enumerate(optflow_tensor.info["optflow_channels"])
                ]
            self._optflow_channel_ids = sorted(self._optflow_channel_ids)

        if self.ts_channels is not None:
            self._ts_channel_ids = [
                i
                for i, k in enumerate(timeseries_tensor.info["timeseries_channels"])
                for c in self.ts_channels
                if c == k
            ]
        else:
            self._ts_channel_ids = [
                i
                for i, k in enumerate(timeseries_tensor.info["timeseries_channels"])
            ]
        self._ts_channel_ids = sorted(self._ts_channel_ids)

        return (
            self._context_channel_ids,
            self._optflow_channel_ids,
            self._ts_channel_ids,
        )

    def get_target_channel_ids(self, ts_tensor):
        if self.ts_target_channels is not None:
            self._ts_target_channel_ids = [
                i
                for i, k in enumerate(ts_tensor.info["timeseries_channels"])
                for c in self.ts_target_channels
                if c == k
            ]
        else:
            self._ts_target_channel_ids = [
                i for i, k in enumerate(ts_tensor.info["timeseries_channels"])
            ]
        self._ts_target_channel_ids = sorted(self._ts_target_channel_ids)
        return self._ts_target_channel_ids

    def __len__(self) -> int:
        return sum(self.n_samples)

    def __getitem__(self, idx: int) -> dict:
        year, station, step_idx_station, step_idx_context = self.year_mapping[idx]

        x_begin_index_ctx = step_idx_context
        x_end_index_ctx = x_begin_index_ctx + self.seq_len

        x_begin_index_ts = step_idx_station
        x_end_index_ts = x_begin_index_ts + self.seq_len
        y_begin_index_ts = x_end_index_ts - self.label_len
        y_end_index_ts = y_begin_index_ts + self.label_len + self.pred_len

        if self.use_target:
            time_utc = pd.Series(
                self.deeplake_ds[f"{year}/{station}/time_utc"][
                    0,
                    x_begin_index_ts:x_end_index_ts,
                ].numpy()
            )
        else:
            time_utc = pd.Series(
                self.deeplake_ds[f"{year}/{station}/time_utc"][
                    0,
                    x_begin_index_ts : x_end_index_ts + self.seq_len,
                ].numpy()
            )

        time_utc_original = time_utc.copy()
        x_time_coords = None
        if self.timestamp_dir:
            x_time_utc_list = time_utc_original.dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
            x_time_utc_begin = x_time_utc_list[0]
            x_mark_begin_idx = self.timestamp_emb[station]["timestamp_to_idx"][x_time_utc_begin]
            x_mark_end_idx = x_mark_begin_idx + self.seq_len
            y_mark_begin_idx = x_mark_end_idx - self.label_len
            y_mark_end_idx = x_mark_begin_idx + self.label_len + self.pred_len
            x_time_coords = self.timestamp_emb[station]["timestamp_embs"][x_mark_begin_idx:x_mark_end_idx:self.token_len]

        mean, std = self.get_stats(station)

        ts_tensor = self.deeplake_ds[f"{year}/{station}/data"]
        context_tensor = self.deeplake_ds[f"{year}/context/data"]
        optflow_tensor = self.deeplake_ds[f"{year}/ctx_opt_flow/data"]
        context_channel_ids, optflow_channel_ids, ts_channel_ids = self.get_channel_ids(
            context_tensor, optflow_tensor, ts_tensor
        )
        target_channel_ids = self.get_target_channel_ids(ts_tensor)
        if self.use_target:
            timseries_data = torch.from_numpy(
                ts_tensor[
                    0,
                    x_begin_index_ts:x_end_index_ts,
                    ts_channel_ids,
                ].numpy()
            )
        else:
            timseries_data = torch.from_numpy(
                ts_tensor[
                    0,
                    x_begin_index_ts : x_end_index_ts + self.seq_len,
                    ts_channel_ids,
                ].numpy()
            )
        if mean is not None:
            timseries_data_wo_mean = timseries_data
            timseries_data = (timseries_data - mean) / std
        if self.use_target:
            target = torch.from_numpy(
                ts_tensor[
                    0,
                    y_begin_index_ts:y_end_index_ts,
                    target_channel_ids,
                ].numpy()
            )
            target_previous = torch.from_numpy(
                ts_tensor[
                    0, x_begin_index_ts:x_end_index_ts, target_channel_ids
                ].numpy()
            )
        if self.use_target:
            context_data = torch.from_numpy(
                context_tensor[
                    x_begin_index_ctx:x_end_index_ctx,
                    context_channel_ids,
                ].numpy()
            )
            optflow_data = torch.from_numpy(
                optflow_tensor[
                    x_begin_index_ctx:x_end_index_ctx,
                    optflow_channel_ids,
                ].numpy()
            )
        else:
            context_data = torch.from_numpy(
                context_tensor[
                    x_begin_index_ctx : x_end_index_ctx + self.seq_len,
                    context_channel_ids,
                ].numpy()
            )
            optflow_data = torch.from_numpy(
                optflow_tensor[
                    x_begin_index_ctx : x_end_index_ctx + self.seq_len,
                    optflow_channel_ids,
                ].numpy()
            )
        coords = torch.from_numpy(self.get_coords(year))
        elevation = torch.from_numpy(self.get_elevation(year))

        station_elevation = (
            torch.Tensor(
                [
                    ts_tensor.info["elevation"],
                ]
            )
            - elevation.mean()
        ) / elevation.std()
        station_elevation = station_elevation.unsqueeze(0)
        station_coords = [
            2 * (ts_tensor.info["coordinates"][0] + 90) / 180 - 1,
            2 * (ts_tensor.info["coordinates"][1] + 180) / 360,
        ]
        station_coords = torch.Tensor(station_coords)[(...,) + (None,) * 2]
        elevation = (elevation - elevation.mean()) / elevation.std()

        if self.image_size is not None:
            optflow_data = F.interpolate(
                optflow_data, size=self.image_size, mode="bicubic", align_corners=True
            )
            context_data = F.interpolate(
                context_data, size=self.image_size, mode="bicubic", align_corners=True
            )

        H, W = context_data.shape[-2:]

        months = torch.from_numpy(time_utc.dt.month.values)[
            (...,) + (None,) * 3
        ].repeat(1, 1, H, W)
        days = torch.from_numpy(time_utc.dt.day.values)[(...,) + (None,) * 3].repeat(
            1, 1, H, W
        )
        hours = torch.from_numpy(time_utc.dt.hour.values)[(...,) + (None,) * 3].repeat(
            1, 1, H, W
        )
        minutes = torch.from_numpy(time_utc.dt.minute.values)[
            (...,) + (None,) * 3
        ].repeat(1, 1, H, W)

        time = torch.cat([months, days, hours, minutes], dim=1)

        if self.input_ids is None:
            query = '<image>\n' + '<TIME_SERIES>' * (self.seq_len // self.token_len)
            assert (448 % self.image_size[0] == 0) and (448 % self.image_size[1] == 0)
            num_patches = (self.image_size[0] / 448) * (self.image_size[1] / 448)
            image_tokens = '<img>' + '<IMG_CONTEXT>' * round(self.num_image_token * num_patches) * self.seq_len + '</img>'
            query = query.replace('<image>', image_tokens, 1)
            self.input_ids = self.tokenizer(query, return_tensors='pt')['input_ids'].squeeze(0)

        return_tensors = {
            "context": context_data,
            "optical_flow": optflow_data,
            "timeseries": timseries_data,
            "time_coordinates": time,
            "station_elevation": station_elevation,
            "station_coords": station_coords,
        }
        return_tensors["timseries_data_wo_mean"] = timseries_data_wo_mean
        return_tensors["time_utc"] = time_utc_original.dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
        return_tensors["idx"] = idx
        
        if x_time_coords is not None:
            return_tensors["x_timestamp"] = x_time_coords
        
        if self.use_target:
            return_tensors["target"] = target
            return_tensors["target_previous"] = target_previous # 输入序列的target_channel_ids维度 [seq_len, 1]

        return_tensors["input_ids"] = self.input_ids
        return_tensors["ghi"] = target_channel_ids

        return return_tensors
