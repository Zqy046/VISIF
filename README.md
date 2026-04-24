# VISIF

## Usage

1. Install Pytorch and necessary dependencies.

```
pip install -r requirements.txt
```

2. Download the large language models from [Hugging Face](https://huggingface.co/). The default LLM is InternVL-8B, you can change the `llm_ckp_dir` in `run.py` to use other LVLMs.

3. Generate the position embedding from textual timestamps.
```
bash ./preprocess_cvv/preprocess4cvv_tscontext_InternVL.sh
```

4. Train and evaluate the model.
```
bash ./scripts/time_series_forecasting/cvv/AutoTimes_InternVL_CVV.sh
```

## Dataset
You can view the dataset [here](https://app.activeloop.ai/crossvivit/SunLake) and can you access it as follows:
```python
import deeplake

ds = deeplake.load('hub://crossvivit/SunLake')
```
If you wish to download it you can do the following:
```python
import deeplake

ds = deeplake.load('hub://crossvivit/SunLake')
local_dataset = ds.copy('/path/to/local/storage', num_workers=4)
```
It is recommended to use a custom dataset to train the model.

## Acknowledgement

We appreciate the following GitHub repos a lot for their valuable code and dataset.
- Time-Series-Library (https://github.com/thuml/Time-Series-Library)
- AutoTimes (https://github.com/thuml/AutoTimes)
- CrossViVit (https://github.com/gitbooo/CrossViVit)
