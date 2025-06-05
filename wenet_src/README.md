# FireRedASR Fine-tuning

## QuickStart

### Env

Please use `Deep Learning OSS Nvidia Driver AMI GPU PyTorch` AMI, with g6e.48xlarge.

```bash
conda create -n wenet python=3.10 -y
conda activate wenet
```

```bash
git clone THIS_REPO.git
pip install -e .  # requirement list can be found in setup.py
```

### Model

```bash
huggingface-cli download FireRedTeam/FireRedASR-AED-L --local-dir ./weights/FireRedASR-AED-L
```

```bash
python wenet/firered/convert_FireRed_AED_L_to_wenet_config_and_ckpt.py --firered_model_dir weights/FireRedASR-AED-L --output_dir weights/FireRedASR-AED-L_wenet
```


### Data

Format, saved in `YOUR_NAME.list`:

```json
{"key": "BAC009S0002W0122", "wav": "/export/data/asr-data/OpenSLR/33/data_aishell/wav/train/S0002/BAC009S0002W0122.wav", "txt": "而对楼市成交抑制作用最大的限购"}
{"key": "BAC009S0002W0123", "wav": "/export/data/asr-data/OpenSLR/33/data_aishell/wav/train/S0002/BAC009S0002W0123.wav", "txt": "也成为地方政府的眼中钉"}
{"key": "BAC009S0002W0124", "wav": "/export/data/asr-data/OpenSLR/33/data_aishell/wav/train/S0002/BAC009S0002W0124.wav", "txt": "自六月底呼和浩特市率先宣布取消限购后"}
```

### Test Base Model

```bash
# Test
python wenet/bin/recognize.py --config weights/FireRedASR-AED-L_wenet/train.yaml --test_data data/zh/test.list --gpu 0 --device cuda --checkpoint weights/FireRedASR-AED-L_wenet/wenet_firered.pt --result_dir ./results --modes attention
```

### Train and Test

#### Training

```bash
# Full-parameter
torchrun --nproc_per_node=8 wenet/bin/train.py --config weights/FireRedASR-AED-L_wenet/train.yaml --model_dir results/training --train_data data/zh/train.list --cv_data data/zh/test.list --ddp.dist_backend 'nccl' --prefetch 16 --num_workers 16 --checkpoint weights/FireRedASR-AED-L_wenet/wenet_firered.pt --use_amp

# LoRA
... --use_lora True
```

#### Testing

```bash
# Full-parameter
python wenet/bin/recognize.py --config results/training/epoch_2.yaml  --test_data data/zh/test.list --gpu 0 --device cuda --checkpoint  results/training/epoch_2.pt --result_dir ./results --modes attention

# LoRA
... --use_lora True 
```
