# Intel® AI Visual Inference Samples

Intel® AI Visual Inference Samples are easy to use python* scripts implementations of workloads with media and AI inference pipelines. The samples are based on media and AI inference frameworks to demonstrate performance of Intel® Data Center GPU Flex Series. 

## Contents
- [Requirements](#requirements)
- [Known Isssues](#known-issues)
- [Running Sample Applications using Docker](#running-sample-applications-using-docker)
- [Measuring Performance with Intel® AI Visual Inference Samples](docs/measuring-performance-with-ai-vi-samples.md)
- [Get Support](#get-support)
- [License](#license)
- [How to Contribute](#how-to-contribute)
- [Security Policy](#security-policy)
- [Code of Conduct](#code-of-conduct)

## Requirements

### Operating System Requirements
- Ubuntu 22.04

### Hardware Requirements
- Intel® Data Center GPU Flex Series on Intel® Xeon® platform hosts

## Known Issues
There are currently no known issues

## Sample Application Usage

Application usage and help can be accessed with any of the following command line options: '-?', '-h', '--help'.  This will provide a full list of command lines supported for the sample you are using and basic information on the sample.  

## Running Sample Applications using Docker

### Step 0: Install host drivers
Install host drivers for Intel® Data Center GPU Flex Series according to the instructions [here](https://dgpu-docs.intel.com/driver/installation.html)

### Step 1: Clone the repository
Clone this repository locally
```bash
git clone --recursive <link to repo>
```

### Step 2: Build and Run Docker Container
1. Build docker image:
   ```bash
   docker build -t ipex -f docker/Dockerfile.ipex .
   ```

2. Run docker container
   ```bash
   docker run -it --rm --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -1) ipex
   ```

### Step 3: Run the samples

```bash
cd samples/pytorch/<sample_name>
# Skip next step if requirements.txt doesn't exist
pip install -r requirements.txt
python main.py
```

## Get support

Report questions, issues and suggestions, using:

* [GitHub* Issues](https://github.com/intel/ai-visual-inference-samples/issues)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file
for details.

## How to Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md) for more information.  

## Security Policy

See [SECURITY.md](SECURITY.md) for more information.  

## Code of Conduct

For information on the contributor code of coduct see [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for more information. 

<br><br>* Other names and brands may be claimed as the property of others.
