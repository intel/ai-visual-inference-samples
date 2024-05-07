# Sample for illustrating latency calls
import torch
import torchvision.io
from torchvision.models import resnet50, ResNet50_Weights
import av.datasets
from metrics import latency_timer, Metrics

# Add decorator to enable latency calculation for the function


@latency_timer
def decode_frames(content):
    """Decodes all video content"""
    return torchvision.io.VideoReader(content, "video")


# Add decorator to enable latency calculation for the function


@latency_timer
def preprocess_frame(frame_data):
    """Preprocess single frame"""
    return preprocess(frame_data)


# Add decorator to enable latency calculation for the function


@latency_timer
def inference(img_input):
    """Run inference on single frame of input image"""
    with torch.no_grad():
        output = torch.nn.functional.softmax(model(img_input.unsqueeze(0)), dim=1)
        ind = torch.argmax(output)
        # print("Class: " + str(ind.item()))


# Execute pipeline


def run_pipeline(content):
    """Decode + Preprocess + Inference + Postprocess and returns number of frames"""
    video_frames = decode_frames(content)
    num_frames = 0
    for frame in video_frames:
        num_frames += 1
        # Apply preprocessing to the input frame
        img_transformed = preprocess_frame(frame["data"])
        inference(img_transformed)

    return num_frames


# Initialize the Weight Transforms
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)

# Set model to eval mode
model.eval()

# Initialize the Weight Transforms
preprocess = weights.transforms()

# Video content
content = av.datasets.curated("pexels/time-lapse-video-of-night-sky-857195.mp4")

# Run the complete pipeline
frames = run_pipeline(content)

# Get the singleton to find latency for specific and/or all functions
m = Metrics()

# Example 1 : Calculate latency for processing across all frames at once
m.calculate_metrics("decode_frames")

# Example 2: Calculate latency for frame by frame processing
m.calculate_metrics("preprocess_frame", frames)
m.calculate_metrics("inference", frames)

# Example 3: Calculate latency across all functions with latency_timer decorator
m.calculate_metrics()

# Example 4: If applicable (i.e number of frames is uniform across all functions), calculate total latency per frame
m.calculate_metrics(frames=frames)
