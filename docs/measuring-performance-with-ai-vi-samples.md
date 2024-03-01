# How to Run Intel® AI Visual Inference Samples Performance Measurements on Intel® Data Center GPU Flex Series

## 1. Install Required Software on Host machine
* To get your system ready to run media analytics workloads with Intel® Data Center GPU Flex Series, follow the instructions [here](https://dgpu-docs.intel.com/installation-guides/ubuntu/ubuntu-jammy-dc.html)
* Install Intel® XPU Manager
  ```bash
  sudo apt install -y  xpu-smi
  ```
* Install Linux tools and enable performance mode
  ```bash
  sudo apt install -y  linux-tools-common linux-tools-generic
  sudo /usr/lib/linux-tools-<version>/cpupower frequency-set --governor performance
  ```

## 2. Build and Run Docker image
Build and run docker image according to the Step 2 from instructions in [README](../README.md)

## 3. Run Intel® AI Visual Inference Samples

Navigate to samples directory by
```bash
cd samples/pytorch
```
Then run desired sample using auxiliary `run_multi.sh` script as shown later in this section.

Run each sample for `--n-procs` 4,6,8 on Intel® Data Center GPU Flex Series 140 and 170. The `--n-procs` specifies how many process instances will be executed.
To reach optimal performance you have to run 8 processes. It is possible to have `PI_ERROR_OUT_OF_RESOURCES` error status on Intel® Data Center GPU Flex Series 140 with ResNet50-v1.5 workload. In that case, you have to reduce the number of processes to 6.

| Name| Command|
|----------|------------------|
| Swin_transfomer FP16 224x224 bs=4 | `./run_multi.sh --sample-name SwinTransformer --sample-args "--iterations 1 --frames-per-iteration 40000" --multi-device --n-procs <fill_here> `  |
| ResNet50-v1.5 INT8 224x224 bs=64 | `./run_multi.sh --sample-name ResNet50 --sample-args "--iterations 1 --frames-per-iteration 400000" --multi-device --n-procs <fill_here> `  |


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
