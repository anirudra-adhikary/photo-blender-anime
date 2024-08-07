import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np

def blend_images(image_path1, image_path2, output_path):
    # Load the stable diffusion pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
    
    # Load images
    img1 = Image.open(image_path1).convert("RGB")
    img2 = Image.open(image_path2).convert("RGB")

    # Resize the second image to match the size of the first image
    img2 = img2.resize(img1.size)

    # Combine images side by side
    img1_np = np.array(img1)
    img2_np = np.array(img2)
    combined_image_np = np.concatenate((img1_np, img2_np), axis=1)
    
    # Ensure dimensions are divisible by 8
    combined_image_np = combined_image_np[:combined_image_np.shape[0]//8*8, :combined_image_np.shape[1]//8*8, :]

    # Convert back to PIL image
    combined_image = Image.fromarray(combined_image_np)

    # Save the combined image temporarily
    temp_combined_image_path = "temp_combined_image.jpg"
    combined_image.save(temp_combined_image_path)

    # Create a custom prompt for blending
    prompt = "A realistic photo of two people standing side-by-side, one on the left and one on the right, both looking at the camera."

    # Generate the blended image using stable diffusion
    generated_images = pipe(
        prompt=prompt,
        height=combined_image.height,
        width=combined_image.width
    ).images

    # Save the resulting image
    blended_image = generated_images[0]
    blended_image.save(output_path)

# Example usage
blend_images(r'd:\Code\Python\Persistent Ventures\photo-blender-anime\img1.png', 
             r'd:\Code\Python\Persistent Ventures\photo-blender-anime\img2.png', 
             'photo_blended.jpg')
