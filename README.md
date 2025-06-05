# wenet_on_sagemaker
使用 Wenet 框架在 Sagemaker training job 进行训练


## 步骤
1. 克隆代码库到本地
```
git clone https://github.com/xqun3/wenet_on_sagemaker.git && cd wenet_on_sagemaker 
```

2. 打包训练环ne传 ECR
```
bash build_and_push.sh
```

3. 准备好数据，打开 submit_training_job.ipynb，按步骤提交训练任务


## 数据格式准备
Format, saved in `YOUR_NAME.list`:

```json
{"key": "BAC009S0002W0122", "wav": "/export/data/asr-data/OpenSLR/33/data_aishell/wav/train/S0002/BAC009S0002W0122.wav", "txt": "而对楼市成交抑制作用最大的限购"}
{"key": "BAC009S0002W0123", "wav": "/export/data/asr-data/OpenSLR/33/data_aishell/wav/train/S0002/BAC009S0002W0123.wav", "txt": "也成为地方政府的眼中钉"}
{"key": "BAC009S0002W0124", "wav": "/export/data/asr-data/OpenSLR/33/data_aishell/wav/train/S0002/BAC009S0002W0124.wav", "txt": "自六月底呼和浩特市率先宣布取消限购后"}