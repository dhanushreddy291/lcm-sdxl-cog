# Prediction interface for Cog
from cog import BasePredictor, Input, Path
import os
import torch
from typing import List
from diffusers import LCMScheduler, AutoPipelineForText2Image

MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_CACHE = "model-cache"
ADAPTER_ID = "latent-consistency/lcm-lora-sdxl"
ADAPTER_CACHE = "adapter-cache"

class Predictor(BasePredictor):
    def setup(self) -> None:
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            MODEL_NAME,
            cache_dir=MODEL_CACHE,
            torch_dtype=torch.float16,
            variant="fp16"     
        ).to("cuda")
        self.pipe.load_lora_weights(ADAPTER_ID, cache_dir=ADAPTER_CACHE)
        self.pipe.fuse_lora()

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="A beautiful brunette pilot girl, beautiful, moody lighting, best quality, full body portrait, real picture, intricate details, depth of field, in a cold snowstorm, , Fujifilm XT3, outdoors, bright day, Beautiful lighting, RAW photo, 8k uhd, film grain, unreal engine 5, ray travig"
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="3d, cgi, render, bad quality, normal quality",
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of inference steps",
            ge=1, le=10, default=4,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(3), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)

        common_args = {
            "prompt": [prompt] * num_outputs,
            "negative_prompt": [negative_prompt] * num_outputs,
            "guidance_scale": 0,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
        }

        output = self.pipe(**common_args)

        output_paths = []
        for i, image in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
        