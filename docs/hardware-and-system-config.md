## Hardware and System Configuration
We apply a limit on the hardware available to each participant to run their solutions. Specifically, 

- All solutions will be run on [AWS g4dn.12xlarge](https://aws.amazon.com/ec2/instance-types/g4/) instances equipped with [NVIDIA T4 GPUs](https://www.nvidia.com/en-us/data-center/tesla-t4/). 
- Solutions for Phase 1 will have access to :
    - `2` x [NVIDIA T4 GPU](https://www.nvidia.com/en-us/data-center/tesla-t4/s). 
    - `20` x vCPU (`10` physical CPU cores)
    - `90GB` RAM 
- Solutions for Phase 2 will have access to: 
    - `4` x [NVIDIA T4 GPU](https://www.nvidia.com/en-us/data-center/tesla-t4/s). 
    - `40` x vCPU (`20` physical CPU cores)
    - `180GB` RAM 

**Note**: When running in `gpu:false` mode, you will have access to `4` x vCPUs (`2` physical cores) and `8GB` RAM. 

Please note that NVIDIA T4 uses a somewhat outdated architectures and is thus not compatible with certain acceleration toolkits (e.g. Flash Attention), so please be careful about compatibility.

Besides, the following restrictions will also be imposed: 

- Network connection will be disabled. 
- Each submission will be assigned a certain amount of time to run. Submissions that exceed the time limits will be killed and will not be evaluated. The tentative time limit is set as follows. 

| Phase  | Track 1 | Track 2 | Track 3 | Track 4 | Track 5 |
| ------ | ------- | ------- | ------- | ------- | ------- |
| **Phase 1**| 140 minutes | 40 minutes | 60 minutes | 60 minutes | 5 hours |

- Each team will be able to make up to **2 submissions per week** per track for Tracks 1-4, and **1 submission per week** for track 5 all-around. 

Based on the hardware and system configuration, we recommend participants to begin with 7B models. According to our experiments, 7B models like Vicuna-7B and Mistral can perform inference smoothly on 2 NVIDIA T4 GPUs, while 13B models will result in OOM. 
