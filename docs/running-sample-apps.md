# Running Sample Applications using Docker

Intel® AI Visual Inference Samples provides applications for multiple AI inference frameworks.
A separate Docker file is provided for every AI inference framework.

## Sample Application Usage

Application usage and help can be accessed with any of the following command line options: `-?`, `-h`, `--help`.  This will provide a full list of supported options for the sample in use as well as basic information on the sample.

## Supported AI inference frameworks

- [OpenVINO™ Toolkit](#openvino-toolkit)
- [Intel® Extension for PyTorch*](#intel-extension-for-pytorch) 

### OpenVINO™ Toolkit

Follow steps below to build Docker image and run samples.

#### Step 1: Install host drivers

Install host drivers for Intel® Data Center GPU Flex Series according to the instructions
[here](https://dgpu-docs.intel.com/driver/installation.html)

#### Step 2: Build Docker image

Clone this repository locally
```bash
git clone --recursive <link-to-repo>
```

Navigate to cloned repository folder and build docker image:
```bash
docker build -t ov -f docker/Dockerfile.ov .
```

#### Step 3: Run the samples

Run docker container:
```bash
docker run -it --rm --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -1) ov
```

Once inside Docker container, run sample by:
```bash
cd samples/openvino/<sample_name>
# Skip next step if requirements.txt doesn't exist
pip install -r requirements.txt
python main.py
```


### Intel® Extension for PyTorch*

Follow steps below to build Docker image and run samples.

#### Step 1: Install host drivers

Install host drivers for Intel® Data Center GPU Flex Series according to the instructions
[here](https://dgpu-docs.intel.com/driver/installation.html)


#### Step 2: Build Docker image

Clone this repository locally
```bash
git clone --recursive <link-to-repo>
```

Navigate to cloned repository folder and build docker image:
```bash
docker build -t ipex -f docker/Dockerfile.ipex .
```

#### Step 3: Run the samples

Run docker container:
```bash
docker run -it --rm --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -1) ipex
```

Once inside Docker container, run sample by:
```bash
cd samples/pytorch/<sample_name>
# Skip next step if requirements.txt doesn't exist
pip install -r requirements.txt
python main.py
```

---
<b>*</b>Other names and brands may be claimed as the property of others.