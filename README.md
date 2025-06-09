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

4. 如果是需要使用 lora 训练，需要修改 wenet_src/train_script_sagemaker.sh 文件，在启动训练命令加上 ```--use_lora True ``` 


## 数据格式准备
具体格式如下所示，数据文件名为 `YOUR_NAME.list`:

**实际的音频路径要修改为训练时落盘到实际路径：```/opt/ml/input/data/```**


假设提交 SageMaker Training job 任务时，数据路径为
```
input_channel = {'zh': "s3://audio-train-datasets/wenet/zh/"}
estimator.fit(input_channel)
```
则对应的音频数据实际路径需要整理为
```json
{"key": "BAC009S0002W0122", "wav": "/opt/ml/input/data/zh/BAC009S0002W0122.wav", "txt": "而对楼市成交抑制作用最大的限购"}
{"key": "BAC009S0002W0123", "wav": "/opt/ml/input/data/zh/BAC009S0002W0123.wav", "txt": "也成为地方政府的眼中钉"}
{"key": "BAC009S0002W0124", "wav": "/opt/ml/input/data/zh/BAC009S0002W0124.wav", "txt": "自六月底呼和浩特市率先宣布取消限购后"}
```


## 训练完以后转回 FieredASR 模型官方格式

```bash
python wenet_src/wenet/firered/convert_wenet_to_FireRed_AED_L_ckpt.py --wenet_config_path results/training/epoch_2.yaml --wenet_pt_path results/training/epoch_2.pt --original_fireredaed_dir weights/FireRedwenet_src/--output_dir weights/full_epoch2
```