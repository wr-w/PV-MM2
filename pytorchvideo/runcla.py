import torch
import argparse
import json
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)
from typing import Dict
from pytorchvideo.models.hub.slowfast import slowfast_r50

from modelcl import ModelInspector

class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """
    def __init__(self,alpha):
        super().__init__()
        self.alpha=alpha

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

def parse_args():
    parser = argparse.ArgumentParser(description='Pytorchvideo demo')
    parser.add_argument('video', help='video file/url or rawframes directory')
    parser.add_argument('label', help='label file')
    parser.add_argument(
        '--name', type=str, default='slowfast', help='the name of the pre-trained model')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--batch_size', type=int, default=10, help='Batch size per GPU/CPU for test')
    parser.add_argument(
        '--batch_num', type=int, default=10, help='Batch number per GPU/CPU for test')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = slowfast_r50(True) 
    model = model.to(device)
    model=model.eval()

    kinetics_classnames=open(args.label).readlines()
    kinetics_classnames=[x.strip() for x in kinetics_classnames]

    side_size = 256
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    crop_size = 256
    num_frames = 32
    sampling_rate = 2
    frames_per_second = 30
    alpha = 4


    transform =  ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x/255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(
                    size=side_size
                ),
                CenterCropVideo(crop_size),
                PackPathway(alpha)
            ]
        ),
    )

    # The duration of the input clip is also specific to the model.
    clip_duration = (num_frames * sampling_rate)/frames_per_second


    # Select the duration of the clip to load by specifying the start and end duration
    # The start_sec should correspond to where the action occurs in the video
    start_sec = 0
    end_sec = start_sec + clip_duration

    # Initialize an EncodedVideo helper class
    video = EncodedVideo.from_path(args.video)

    # Load the desired clip
    video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

    # Apply a transform to normalize the video input
    video_data = transform(video_data)

    # Move the inputs to the desired device
    inputs = video_data["video"]
    inputs = [i.to(device)[None, ...] for i in inputs]

    benchmark=ModelInspector(inputs,args.batch_num,args.batch_size,args.label,model,args.name)
    benchmark.run_model()


if __name__ == '__main__':
    main()
