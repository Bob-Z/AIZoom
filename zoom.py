import cv2
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionInpaintPipeline
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


def create_inverted_centered_mask(image, mask_size_ratio=0.95):
    height, width = image.shape[:2]
    mask_height = int(height * mask_size_ratio)
    mask_width = int(width * mask_size_ratio)
    top = (height - mask_height) // 2
    left = (width - mask_width) // 2
    mask = np.ones((height, width), dtype=np.uint8) * 255
    mask[top:top + mask_height, left:left + mask_width] = 0
    return mask


def generate_missing_part(image, inpaint_pipeline):
    # Create an inverted centered mask for the missing part
    mask = create_inverted_centered_mask(image, mask_size_ratio=0.95)

    # Convert image and mask to PIL format
    image_pil = Image.fromarray(image)
    mask_pil = Image.fromarray(mask)

    # Save the intermediate images for debugging
    image_pil.save("intermediate_image.png", format='PNG')
    mask_pil.save("intermediate_mask.png", format='PNG')

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

    return result_np, mask


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


def main():
    # Check if GPU is available and set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the Stable Diffusion model for generating the initial image
    generation_pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None,
                                                                  requires_safety_checker=False).to(device)

    # Generate the initial image using the prompt
    with torch.autocast("cuda"):
        initial_image = generation_pipeline(
            height=config.data["height"],
            width=config.data["width"],
            prompt=config.data["prompt"],
            strength=config.data["strength"],
            guidance_scale=config.data["guidance_scale"],
            num_inference_steps=config.data["num_inference_steps"]
        ).images[0]
    initial_image = np.array(initial_image)

    # Free up memory by deleting the generation pipeline
    del generation_pipeline
    torch.cuda.empty_cache()

    # Load the pre-trained Stable Diffusion Inpaint model for inpainting
    inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting",
                                                                      safety_checker=None,
                                                                      requires_safety_checker=False).to(device)

    current_image = initial_image

    for i in range(config.data["num_iterations"]):
        # Resize the image to 95% of its size
        resized_image = resize_image(current_image, 0.95)

        # Center the resized image within the original image dimensions
        centered_image = center_image(resized_image, current_image.shape[:2])

        # Generate the missing part to restore the image to its original size
        final_image, mask = generate_missing_part(centered_image, inpaint_pipeline)

        # Overlay the source image onto the inpainted image
        final_image = overlay_images(centered_image, final_image, mask)

        # Save the output image with an index
        output_image = Image.fromarray(final_image)
        output_image.save(f"{config.data['output_dir']}/output_image_{i}.png", format='PNG')  # Explicitly specify the format

        # Use the output image as the input for the next iteration
        current_image = np.array(output_image)


if __name__ == "__main__":
    config.read_command_line()
    main()
