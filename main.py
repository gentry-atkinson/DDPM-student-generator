# Author: Gentry Atkinson
# Organization: St. Edwards University
# Date: 3 Jan, 2024

import torch
import os
from torchvision import transforms
from diffusers import StableDiffusionImageVariationPipeline
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image

if torch.cuda.is_available():
        device = "cuda"
else:
    device = "cpu"

if __name__ == '__main__':
    #Setup
    if not os.path.exists(os.path.join("DDPM student generator", "output_images")):
        os.mkdir(os.path.join("DDPM student generator", "output_images"))

    sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
        "lambdalabs/sd-image-variations-diffusers",
        revision="v2.0",
    )
    sd_pipe = sd_pipe.to(device)

    ref_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0", 
        torch_dtype=torch.float16, variant="fp16", 
        use_safetensors=True
    )
    ref_pipe = ref_pipe.to(device)
    prompt = "A university student enjoying life at St. Edwards University."

    im = Image.open(os.path.join("sted_train_photos", "train", "student", "sted_train_001.png"))
    tform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
            ),
    ])
    inp = tform(im).to(device)

    out = sd_pipe(im, guidance_scale=3)
    out = out.images[0]
    out = out.resize((500, 500))
    image = ref_pipe(prompt, image=out).images[0]
    image.save("result.png")