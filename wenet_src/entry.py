import os
import json
import socket
import yaml
import time
import subprocess
# import sagemaker_ssh_helper
# sagemaker_ssh_helper.setup_and_start_ssh()

if __name__ == "__main__":
   
    hosts = json.loads(os.environ['SM_HOSTS'])
    current_host = os.environ['SM_CURRENT_HOST']
    host_rank = int(hosts.index(current_host))
    
    #Parse the IP address of the master node in the multiple nodes cluster of SageMaker training.
    master = json.loads(os.environ['SM_TRAINING_ENV'])['master_hostname']
    master_addr = socket.gethostbyname(master)
    
    # os.environ['DS_BUILD_FUSED_ADAM'] = '1'
    os.environ['NODE_INDEX'] = str(host_rank)
    os.environ['SM_MASTER'] = str(master)
    os.environ['SM_MASTER_ADDR'] = str(master_addr)
    # os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
    
    # backend env config
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
   #  os.environ['FI_PROVIDER'] = 'efa'
   #  os.environ['NCCL_PROTO'] = 'simple'
   # # os.environ['FI_EFA_USE_DEVICE_RDMA'] = '1'
   #  os.environ['NCCL_DEBUG'] = 'INFO'
   #  os.environ['HCCL_OVER_OFI'] = '1'
    os.environ['NCCL_SOCKET_IFNAME'] = os.environ["SM_NETWORK_INTERFACE_NAME"]


    os.environ['NCCL_DEBUG'] = 'ERROR'
    # os.environ['NCCL_OVER_OFI'] = '1'
    os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
    os.system("chmod +x ./train_script_sagemaker.sh")
    os.system("chmod +x ./s5cmd")

    print("*****************start cp foundation model*****************************")
    os.system("./s5cmd sync {0}/* {1}".format(os.environ['MODEL_S3_PATH'], os.environ["MODEL_LOCAL_PATH"]))
    print(f'-----finished cp-------')
    # os.environ['SM_MODEL_DIR']

    # time.sleep(14400)
    # os.system("/bin/bash -c ./train_script_sagemaker.sh")
    train_cmd = "bash train_script_sagemaker.sh"
    print(f"Executing command: {train_cmd}")

    print("*****************finished training, start cp finetuned model*****************************")
    # os.system("ls -l /opt/ml/checkpoints")
    # os.system('find /opt/ml/checkpoints -maxdepth 1 -type f ! -name "*.sagemaker-uploaded" ! -name "checkpoint-*" -print')
    # os.system('find /opt/ml/checkpoints -maxdepth 1 -type d ! -name "." ! -name ".." ! -name "checkpoint-*" -print')
    try:
        # Run the training command and check return code
        result = subprocess.run(train_cmd, shell=True, check=True)
        print("Training completed successfully")
        # os.system('python grpo/scripts/model_merger.py --local_dir "/opt/ml/checkpoints/global_step_$(cat /opt/ml/checkpoints/latest_global_step.txt)"/actor/')
        # os.system(f'cp -r "/opt/ml/checkpoints/" {os.environ["SM_MODEL_DIR"]}')
        os.system(f'ls -l {os.environ["SM_MODEL_DIR"]}')

    except subprocess.CalledProcessError as e:
        # Training failed
        print(f"ERROR: Training failed with return code {e.returncode}")
        sys.exit(1) 

    # os.system('find /opt/ml/checkpoints -maxdepth 1 -type f ! -name "*.sagemaker-uploaded" ! -name "checkpoint-*" -exec mv {} "/opt/ml/model" \;')
    # os.system('find /opt/ml/checkpoints -maxdepth 1 -type d ! -name "." ! -name ".." ! -name "checkpoint-*" -exec mv {} "/opt/ml/model" \;')

    # os.system('aws s3 sync --exclude "/opt/ml/checkpoints/checkpoint-*" --exclude "*.sagemaker-uploaded" {0} {1}'.format("/opt/ml/checkpoints", os.environ['OUTPUT_MODEL_S3_PATH']))
    # print(f'-----finished cp-------')
