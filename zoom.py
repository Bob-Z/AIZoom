import cv2
from PIL import Image
import config
import ai
import numpy as np


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


def get_inpaint_image(image_pil, mask_pil, pipeline=None):
    image, pipeline = ai.get_inpaint_image_cuda(image_pil, mask_pil, pipeline)
    if image is None:
        image, pipeline = ai.get_inpaint_image_cpu_offload(image_pil, mask_pil, pipeline)
        if image is None:
            image, pipeline = ai.get_inpaint_image_cpu(image_pil, mask_pil, pipeline)

    return image, pipeline


def generate_missing_part(image_pil, mask_pil, pipeline):
    # Generate the missing part using the inpaint pipeline
    image, pipeline = get_inpaint_image(image_pil, mask_pil, pipeline)
    # Convert the result back to a numpy array
    result_np = np.array(image)

    return result_np, pipeline


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


def get_image():
    image = ai.get_image_cuda()
    if image is None:
        image = ai.get_image_cpu_offload()
        if image is None:
            image = ai.get_image_cpu()

    return image


def generate_initial_image():
    initial_image = get_image()
    if initial_image is None:
        raise "Error can't generate initial image"

    initial_image.save("initial_image.png", format='PNG')
    initial_image = np.array(initial_image)

    return initial_image


def generate_all_image(initial_image):
    print("Start generating inpaint images")

    current_image = initial_image

    # Create an inverted centered mask for the missing part
    mask = create_inverted_centered_mask(current_image, mask_size_ratio=config.data["resize_ratio"] - config.data[
        "mask_resize_margin"])
    mask_pil = Image.fromarray(mask)
    mask_pil.save("mask.png", format='PNG')

    pipeline = None

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
        inpainted_image, pipeline = generate_missing_part(to_inpaint_image_pil, mask_pil, pipeline)
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
    ai.print_best_device()

    initial_image = generate_initial_image()

    generate_all_image(initial_image)


if __name__ == "__main__":
    config.read_command_line()
    main()
