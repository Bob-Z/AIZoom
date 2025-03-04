from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionInpaintPipeline
from diffusers import AutoPipelineForText2Image
from diffusers import AutoPipelineForInpainting
import torch
import config
import gc

def print_best_device():
    # Check if GPU is available and set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Best device available: ", device)

def get_image_cuda(pipeline=None):
    try:
        if pipeline is None:
            print("Trying CUDA device for initial image")
            pipeline = AutoPipelineForText2Image.from_pretrained(
                #    "CompVis/stable-diffusion-v1-4",
                #    "stable-diffusion-v1-5/stable-diffusion-v1-5",
                "stabilityai/stable-diffusion-xl-base-1.0",
                # "RunDiffusion/Juggernaut-XL-v9",
                #safety_checker=None,
                #requires_safety_checker=False,
                torch_dtype=torch.float16,
                variant='fp16'
            )

            pipeline.to('cuda')

        image = pipeline(
            height=config.data["height"],
            width=config.data["width"],
            prompt=config.data["prompt"],
            strength=config.data["strength"],
            guidance_scale=config.data["guidance_scale"],
            num_inference_steps=config.data["num_inference_steps"]
        ).images[0]

        return image
    except:
        if pipeline is not None:
            del pipeline
            gc.collect()
        print("Error using CUDA device for initial image")
        return None


def get_image_cpu_offload(pipeline=None):
    try:
        if pipeline is None:
            print("Trying CUDA device with CPU offload for initial image")
            pipeline = AutoPipelineForText2Image.from_pretrained(
                #    "CompVis/stable-diffusion-v1-4",
                #    "stable-diffusion-v1-5/stable-diffusion-v1-5",
                "stabilityai/stable-diffusion-xl-base-1.0",
                # "RunDiffusion/Juggernaut-XL-v9",
                #safety_checker=None,
                #requires_safety_checker=False,
                torch_dtype=torch.float16,
                variant='fp16'
            )

            pipeline.enable_model_cpu_offload()

        image = pipeline(
            height=config.data["height"],
            width=config.data["width"],
            prompt=config.data["prompt"],
            strength=config.data["strength"],
            guidance_scale=config.data["guidance_scale"],
            num_inference_steps=config.data["num_inference_steps"]
        ).images[0]

        return image
    except:
        if pipeline is not None:
            del pipeline
            gc.collect()
        print("Error using CUDA with CPU offload for initial image")
        return None


def get_image_cpu(pipeline=None):
    print("Trying CPU device for initial image")
    try:
        if pipeline is None:
            pipeline = AutoPipelineForText2Image.from_pretrained(
                #    "CompVis/stable-diffusion-v1-4",
                #    "stable-diffusion-v1-5/stable-diffusion-v1-5",
                "stabilityai/stable-diffusion-xl-base-1.0",
                # "RunDiffusion/Juggernaut-XL-v9",
                safety_checker=None,
                requires_safety_checker=False,
            )

            pipeline.to("cpu")

        image = pipeline(
            height=config.data["height"],
            width=config.data["width"],
            prompt=config.data["prompt"],
            strength=config.data["strength"],
            guidance_scale=config.data["guidance_scale"],
            num_inference_steps=config.data["num_inference_steps"]
        ).images[0]

        return image
    except:
        if pipeline is not None:
            del pipeline
            gc.collect()
        print("Error using CPU device for initial image")

        return None

def get_inpaint_image_cuda(image_pil, mask_pil, pipeline=None):
    try:
        if pipeline is None:
            print("Trying CUDA device for inpaint image")
            # Load the pre-trained Stable Diffusion Inpaint model for inpainting
            pipeline = AutoPipelineForInpainting.from_pretrained(
                # "runwayml/stable-diffusion-inpainting",
                # "stable-diffusion-v1-5/stable-diffusion-inpainting",
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                #safety_checker=None,
                #requires_safety_checker=False,
                torch_dtype=torch.float16,
                variant='fp16'
            )

            pipeline.to('cuda')

        image = pipeline(
            height=config.data["height"],
            width=config.data["width"],
            prompt=config.data["prompt"],
            image=image_pil,
            mask_image=mask_pil,
            strength=config.data["strength"],
            guidance_scale=config.data["guidance_scale"],
            num_inference_steps=config.data["num_inference_steps"]
        ).images[0]

        return image, pipeline
    except:
        if pipeline is not None:
            del pipeline
            gc.collect()
        print("Error using CUDA device for inpaint image")
        return None, None


def get_inpaint_image_cpu_offload(image_pil, mask_pil, pipeline=None,):
    try:
        if pipeline is None:
            print("Trying CUDA device with CPU offload for inpaint image")
            # Load the pre-trained Stable Diffusion Inpaint model for inpainting
            pipeline = AutoPipelineForInpainting.from_pretrained(
                # "runwayml/stable-diffusion-inpainting",
                # "stable-diffusion-v1-5/stable-diffusion-inpainting",
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                #safety_checker=None,
                #requires_safety_checker=False,
                torch_dtype=torch.float16,
                variant='fp16'
            )

            pipeline.enable_model_cpu_offload()

        image = pipeline(
            height=config.data["height"],
            width=config.data["width"],
            prompt=config.data["prompt"],
            image=image_pil,
            mask_image=mask_pil,
            strength=config.data["strength"],
            guidance_scale=config.data["guidance_scale"],
            num_inference_steps=config.data["num_inference_steps"]
        ).images[0]

        return image, pipeline
    except:
        if pipeline is not None:
            del pipeline
            gc.collect()
        print("Error using CUDA with CPU offload for inpaint image")
        return None, None


def get_inpaint_image_cpu(image_pil, mask_pil, pipeline=None,):
    print("Trying CPU device for inpaint image")
    try:
        if pipeline is None:
            # Load the pre-trained Stable Diffusion Inpaint model for inpainting
            pipeline = AutoPipelineForInpainting.from_pretrained(
                # "runwayml/stable-diffusion-inpainting",
                # "stable-diffusion-v1-5/stable-diffusion-inpainting",
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                safety_checker=None,
                requires_safety_checker=False,
                #torch_dtype=torch.float16,
                #variant='fp16'
            )

            pipeline.to('cpu')

        image = pipeline(
            height=config.data["height"],
            width=config.data["width"],
            prompt=config.data["prompt"],
            image=image_pil,
            mask_image=mask_pil,
            strength=config.data["strength"],
            guidance_scale=config.data["guidance_scale"],
            num_inference_steps=config.data["num_inference_steps"]
        ).images[0]

        return image, pipeline
    except:
        if pipeline is not None:
            del pipeline
            gc.collect()
        print("Error using CPU device for inpaint image")

        return None, None