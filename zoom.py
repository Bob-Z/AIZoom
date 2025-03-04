import gc

import cv2
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionInpaintPipeline
from diffusers import AutoPipelineForText2Image
from diffusers import AutoPipelineForInpainting
import config


def resize_image(image, scale_factor):
    height, width = image.shape[:2]
    new_size = (int(width * scale_factor), int(height * scale_factor))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)
    return resized_image


def center_image(image, target_size):
    height, width = image.shape[:2]
    target_height, target_width = target_size
    top = (target_height - height) // 2
    left = (target_width - width) // 2
    centered_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    centered_image[top:top + height, left:left + width] = image
    return centered_image


def create_inverted_centered_mask(image, mask_size_ratio):
    height, width = image.shape[:2]
    mask_height = int(height * mask_size_ratio)
    mask_width = int(width * mask_size_ratio)
    top = (height - mask_height) // 2
    left = (width - mask_width) // 2
    mask = np.ones((height, width), dtype=np.uint8) * 255
    mask[top:top + mask_height, left:left + mask_width] = 0
    return mask


def generate_missing_part(inpaint_pipeline, image_pil, mask_pil):
    # Generate the missing part using the inpaint pipeline
    result = inpaint_pipeline(
        height=config.data["height"],
        width=config.data["width"],
        prompt=config.data["prompt"],
        image=image_pil,
        mask_image=mask_pil,
        strength=config.data["strength"],
        guidance_scale=config.data["guidance_scale"],
        num_inference_steps=config.data["num_inference_steps"]
    ).images[0]

    # Convert the result back to a numpy array
    result_np = np.array(result)

    return result_np


def overlay_images(source_image, inpainted_image, mask):
    # Invert the mask
    inverted_mask = 255 - mask

    # Create a copy of the source image with an alpha channel
    source_image_rgba = cv2.cvtColor(source_image, cv2.COLOR_RGB2RGBA)

    # Apply the inverted mask to the alpha channel
    source_image_rgba[:, :, 3] = inverted_mask

    # Convert the inpainted image to RGBA
    inpainted_image_rgba = cv2.cvtColor(inpainted_image, cv2.COLOR_RGB2RGBA)

    # Overlay the source image onto the inpainted image
    combined = inpainted_image_rgba.copy()
    alpha_source = source_image_rgba[:, :, 3] / 255.0
    alpha_inpainted = 1 - alpha_source

    for c in range(0, 3):
        combined[:, :, c] = (
                alpha_source * source_image_rgba[:, :, c] + alpha_inpainted * inpainted_image_rgba[:, :, c])

    # Convert the result back to RGB
    combined_rgb = cv2.cvtColor(combined, cv2.COLOR_RGBA2RGB)

    return combined_rgb


def generate_noise_image(height, width):
    # Generate a random noise image
    noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return noise


def get_image_cuda(pipeline=None):
    print("\nTrying CUDA device\n")
    try:
        if pipeline is None:
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
        print("")
        print("Error using CUDA device")
        print("")
        return None


def get_image_cpu_offload(pipeline=None):
    print("\nTrying CUDA device with CPU offload\n")
    try:
        if pipeline is None:
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
        print("")
        print("Error using CUDA with CPU offload")
        print("")
        return None


def get_image_cpu(pipeline=None):
    print("\nTrying CPU device\n")
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
        print("")
        print("Error using CPU device")
        print("")

        return None


def get_image(pipeline=None, device="cuda", cpu_offload=False):
    image = get_image_cuda()
    if image is None:
        image = get_image_cpu_offload()
        if image is None:
            image = get_image_cpu()

    return image


def generate_initial_image():
    initial_image = get_image()
    if initial_image is None:
        raise "Error can't generate initial image"

    initial_image.save("initial_image.png", format='PNG')
    initial_image = np.array(initial_image)

    return initial_image


def generate_all_image(device, initial_image):
    print("Start generating inpaint images")
    # Load the pre-trained Stable Diffusion Inpaint model for inpainting
    inpaint_pipeline = AutoPipelineForInpainting.from_pretrained(
        # "runwayml/stable-diffusion-inpainting",
        # "stable-diffusion-v1-5/stable-diffusion-inpainting",
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        safety_checker=None,
        requires_safety_checker=False,
        torch_dtype=torch.float16,
        variant='fp16'
    )

    # inpaint_pipeline.to(device)
    inpaint_pipeline.enable_model_cpu_offload()

    current_image = initial_image

    # Create an inverted centered mask for the missing part
    mask = create_inverted_centered_mask(current_image, mask_size_ratio=config.data["resize_ratio"] - config.data[
        "mask_resize_margin"])
    mask_pil = Image.fromarray(mask)
    mask_pil.save("mask.png", format='PNG')

    for i in range(config.data["num_iterations"]):
        # Resize the image
        resized_image = resize_image(current_image, config.data["resize_ratio"])
        resized_image_pil = Image.fromarray(resized_image)
        resized_image_pil.save("resized_image.png", format='PNG')

        # Center the resized image within the original image dimensions
        centered_image = center_image(resized_image, current_image.shape[:2])
        centered_image_pil = Image.fromarray(centered_image)
        centered_image_pil.save("centered_image.png", format='PNG')

        # Generate a noise image
        noise_image = generate_noise_image(current_image.shape[0], current_image.shape[1])
        noise_image_pil = Image.fromarray(noise_image)
        noise_image_pil.save("noise_image.png", format='PNG')

        # Overlay the centered image onto the noise image
        to_inpaint_image = overlay_images(centered_image, noise_image, mask)
        to_inpaint_image_pil = Image.fromarray(to_inpaint_image)
        to_inpaint_image_pil.save("to_inpaint_image.png", format='PNG')

        # Generate the missing part to restore the image to its original size
        inpainted_image = generate_missing_part(inpaint_pipeline, to_inpaint_image_pil, mask_pil)
        inpainted_pil = Image.fromarray(inpainted_image)
        inpainted_pil.save("inpainted.png", format='PNG')

        # Overlay the source image onto the inpainted image
        final_image = overlay_images(centered_image, inpainted_image, mask)

        # Save the output image with an index
        output_image = Image.fromarray(final_image)
        output_image.save(f"{config.data['output_dir']}/output_image_{i}.png",
                          format='PNG')  # Explicitly specify the format

        # Use the output image as the input for the next iteration
        current_image = final_image


def main():
    # Check if GPU is available and set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Best device available: ", device)

    initial_image = generate_initial_image()

    generate_all_image(device, initial_image)


if __name__ == "__main__":
    config.read_command_line()
    main()
