# How to Run Intel® AI Visual Inference Samples Performance Measurements on Intel® Data Center GPU Flex Series

## 1. Install Required Software on Host machine
* Get your system ready to run media analytics workloads with Intel® Data Center GPU Flex Series, follow the instructions [here](https://dgpu-docs.intel.com/driver/installation.html#ubuntu-install-steps)
* Install Intel® XPU Manager
  ```bash
  sudo apt install -y  xpu-smi
  ```
* Enable performance mode
  ```bash
  sudo /usr/lib/linux-tools-<version>/cpupower frequency-set --governor performance
  ```

## 2. Build and Run Docker image
Build and run docker image for desired backend using [these instructions](running-sample-apps.md)

## 3. Run Intel® AI Visual Inference Samples

### Using Intel® Extension for PyTorch* as inference framework

Navigate to the samples directory inside the docker container:
```bash
cd samples/pytorch
```

Run your desired sample using `run_multi.sh` script as shown later in this section.

Run each sample with `--n-procs` with values 4,6, or 8 on Intel® Data Center GPU Flex Series 140 and 170. `--n-procs` specifies how many process instances will be executed.
To reach optimal performance run 8 processes.  
* You may encounter the `PI_ERROR_OUT_OF_RESOURCES` error when running the ResNet50-v1.5 workload on Intel® Data Center GPU Flex Series 140 systems. In that case, reduce the number of processes to 6.  

By default, the duration of sample execution is selected according to each workload performance and takes around 2 minutes, if you want to change the duration you can modify the number of frames to process via `--sample-args "--num-frames <fill_here>"` option.

| Name     | Precis. | Command |
|----------|-----------|---------|
| SwinTransfomer</br>[224x224 bs=4] | FP16 | `./run_multi.sh --sample-name SwinTransformer --sample-args "--num-frames 40000" --multi-device --n-procs <fill_here>`  |
| ResNet50-v1.5 </br>[224x224 bs=64] | INT8 | `./run_multi.sh --sample-name ResNet50 --sample-args "--num-frames 400000" --multi-device --n-procs <fill_here>`  |

### Using OpenVINO™ Toolkit as inference framework

Navigate to the samples directory inside the docker container:
```bash
cd samples/openvino
```

Run your desired sample using auxiliary `run_multi.sh` script as shown later in this section.

OpenVINO™ Toolkit based samples are optimized to perform within a single process per GPU.
So, for Intel® Data Center GPU Flex Series 170 it's reccomended to use 1 process (`--n-procs 1`) and for Intel® Data Center GPU Flex Series 140 – 2 processes (`--n-procs 2`) to reach optimal throughput. 

> ℹ️ When running the samples on Intel® Data Center GPU Flex Series 170 using a single process, you can omit `run_multi.sh` script and invoke sample directly by going to the sample directory and calling `python3 main.py`.
>
> Please refer to ["Running Sample Applications"](running-sample-apps.md) document for details.

| Name     | Precis. | Command |
|----------|-----------|---------|
| FBNet</br>[224x224 bs=64]   | FP16 | `./run_multi.sh --sample-name FBNet --n-procs <fill_here> --multi-device` |
| ResNet50</br>[224x224 bs=64] | INT8 | `./run_multi.sh --sample-name ResNet50 --n-procs <fill_here> --multi-device` |
| SwinTransformer</br>[224x224 bs=64] | FP16 | `./run_multi.sh --sample-name SwinTransformer --n-procs <fill_here> --multi-device` |
| YOLOv5m</br>[320x320 bs=64] | INT8 | `./run_multi.sh --sample-name YOLOv5m --n-procs <fill_here> --multi-device` | 
| YOLOv5m + AVC</br>[320x320 bs=64] | INT8 | `./run_multi.sh --sample-name YOLOv5m --n-procs <fill_here> --multi-device --sample-args "--input ~/ma/data/media/20230104_dog_bark_1920x1080_3mbps_30fps_ld_h264.mp4"` | 
| SSD MobileNet</br>[300x300 bs=64] | INT8 | `./run_multi.sh --sample-name ssd_mobilenet_v1_coco --n-procs <fill_here> --multi-device` |


### Heterogeneous pipelines using CPU & GPU
To run heterogeneous pipelines where media processing is executed on GPU and inference on CPU please use `run_multi.py`. The tool is designed to work with multi-device GPUs and optimally configures CPU threading parameters. 

| Name     | Precis. | Command |
|----------|-----------|---------|
| ResNet50_GPU_CPU </br>[224x224 bs=4] | INT8 | `python3 run_multi.py --backend openvino --sample-name ResNet50_GPU_CPU --num-processes <fill_here> --num-devices <fill_here> --install-requirements --use-taskset`  |
| SSD MobileNet</br>[300x300 bs=64] | INT8 | `python3 run_multi.py --backend openvino --sample-name ssd_mobilenet_v1_coco_GPU_CPU --num-processes <fill_here> --num-devices <fill_here> --install-requirements --use-taskset` |
| YOLOv5m_GPU_CPU</br>[320x320 bs=4] | INT8 | `python3 run_multi.py --backend openvino --sample-name YOLOv5m_GPU_CPU --num-processes <fill_here> --num-devices <fill_here> --install-requirements --use-taskset`  |

## Profiling
Profiling can be done using Intel® XPU Manager for any workloads one for each run. To get more information about available metrics see [Intel® XPU Manager](https://github.com/intel/xpumanager/blob/master/doc/smi_user_guide.md)

Example Profiling
```bash
sudo xpu-smi dump -d 0 -m 0,1,2,3,5,9,10,11,18,22,24,26,27,35,36,19,20 >> xpu_smi_dump.csv
```

## Notes on Intel® AI Visual Inference Samples Performance Measurements on Intel® Data Center GPU Flex Series
It is normal to have some variance between any two SoCs of the same design and the same manufacturing process. Tiny unavoidable differences change the amount of leakage current in each chip. Usually, this variance is negligible, but it can manifest as performance differences under identical power limits. This is more noticeable at higher operating temperatures. You may see some small performance differences between GPUs under identical operating conditions.

In general, as temperature increases peak performance decreases. As temperatures rise, leakage current increases and power management must lower the voltage to stay within designed power limits. Lower voltage means lower clock speeds, which will directly translate to lower performance.

These GPUs do not have active cooling fans, so you must ensure that your server is set up with proper airflow to the GPUs. Performance is measured after starting from idle state and running for ~1-2 minutes. With room temperature ambient air and proper airflow, the GPUs will heat up to 55-60℃ range after ~1-2 minutes.

In addition to temperature impacts, the voltage and clock frequency will change depending on the utilization of the media engines and compute engines at any given time. In Intel® Data Center GPU Flex Series, the media engines and compute engine share a GPU clock but have different voltage and frequency needs/limits. Therefore, changing the ratio of media engine utilization vs. compute engine utilization can significantly change the clock speed and end-to-end performance.
For compute-bound workloads like Swin_transfomer it is expected that compute engine frequency is x2 more than media engine frequency, in opposite to that media-bound workloads like ResNet50-v1.5 have 1:1 compute/media frequency ratio. To check actual media and compute frequencies you can use Intel® XPU Manager with the following cmd line:
```
sudo xpu-smi dump -d 0 -m 2, 36
```
where `-m 2` shows compute engine and `-m 36` – media engine frequencies. **It may require waiting for some time (up to several minutes) to get frequencies converged to the targeted ratio.**

The media engine and compute engine utilization can change depending on batch size, input video details (codec, resolution, bitrate, etc.), the inference models used, the number of objects in the video content to be detected/classified, etc. To reproduce the results make sure to follow the instructions, use the exact command lines, and input video provided.

## Specific of running workloads on Intel® Data Center GPU Flex Series 140
This card has 2 GPUs onboard and requires to use of both of them to show the best performance. This is why `run_multi.sh` script has option `--multi-device` to split loading and running half of processes on each GPU. It is default parameter and doesn't impact execution on Intel® Data Center GPU Flex Series 170.

## Specific of running GPU plus CPU workloads
To optimize and balance loading it is possible to execute media analytics workloads using host Intel® Xeon® CPU for AI inference. It may give additional performance gains for platform configurations with 4th/5th Gen Intel® Xeon® Scalable Processors and Intel® Data Center GPU Flex Series 140. You need to use both GPUs with the help of `--num-devices 2` option for `run_multi.sh` script. For the platform with one GPU card the optimal command lines:
| Name     | Precis. | Command |
|----------|-----------|---------|
| ResNet50_GPU_CPU </br>[224x224 bs=4] | INT8 | `python3 run_multi.py --backend openvino --sample-name ResNet50_GPU_CPU --num-processes 2 --num-devices 2 --install-requirements --use-taskset`  |
| SSD MobileNet_GPU_CPU </br>[300x300 bs=64] | INT8 | `python3 run_multi.py --backend openvino --sample-name ssd_mobilenet_v1_coco_GPU_CPU --num-processes 2 --num-devices 2 --install-requirements --use-taskset` |
| YOLOv5m_GPU_CPU</br>[320x320 bs=4] | INT8 | `python3 run_multi.py --backend openvino --sample-name YOLOv5m_GPU_CPU --num-processes 2 --num-devices 2 --install-requirements --use-taskset`  |
