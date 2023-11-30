"""Script for running Simple Inpainting pipeline"""

import enum
from pathlib import Path

import PIL.Image
import torch
import typer

from pipelines import StableDiffusionSimpleInpaintingPipeline  # type: ignore
from image_masker import image_masker

app = typer.Typer(pretty_exceptions_show_locals=False)

class DTYPE(enum.Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"

@app.command()
def run_inpainting(
    image_path: Path = typer.Option(
        default=Path("images/watch.png"), help="Path to input image"
    ),
    save_path: Path = typer.Option(
        default=Path("images/inpainting_watch_test.png"), help="Saves output image to *save_path*"
    ),
    save_concat: bool = typer.Option(
        default=True, help="Saves concatenated image, mask and inpaint to *save_path*"
    ),
    disable_safety_checker: bool = typer.Option(
        default=True, help="Disables safety checker"
    ),
    model_id: str = typer.Option(
        default="redstonehero/Yiffymix_Diffusers", help="Model ID in HuggingFace Hub"
    ),
    device: str = typer.Option(default="cuda", help="Device to run inference on"),
    dtype: DTYPE = typer.Option(
        default=DTYPE.FLOAT16.value, help="Data type to use for inference"
    ),
    seed: int = typer.Option(default=0, help="Random seed"),
    prompt: str = typer.Option(default="Cute cat", help="Prompt to run inference on"),
    strength: float = typer.Option(
        default=0.7, help="How much to transform the reference `image`"
    ),
    num_inference_steps: int = typer.Option(
        default=150, help="Number of inference steps"
    ),
    guidance_scale: float = typer.Option(
        default=6,
        help="Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).",
    ),
    max_image_side: int = typer.Option(
        default=512, help="Maximum width or height of the output image"
    ),
) -> None:

    if dtype == DTYPE.FLOAT32:
        dtype_torch = torch.float32
    elif dtype == DTYPE.FLOAT16:
        dtype_torch = torch.float16
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

    pipeline = StableDiffusionSimpleInpaintingPipeline.from_pretrained(
        pretrained_model_name_or_path=model_id,
        #torch_dtype=dtype_torch,
    )
    pipeline = pipeline.to(device)

    if disable_safety_checker:
        pipeline.safety_checker = None

    # Load and process the image
    image = PIL.Image.open(image_path).convert("RGB")
    image.thumbnail((max_image_side, max_image_side))

    # Use image_masker to create a mask
    src = image_masker.resize_image(str(image_path))
    r = image_masker.select_roi(src)
    mask = image_masker.create_mask(src, r)
    mask = PIL.Image.fromarray(mask).convert("RGB")
    mask.thumbnail((max_image_side, max_image_side))

    for i in range(30):  # Iterate over 30 different seeds
        current_seed = seed + i  # Increment seed for each iteration
        image_result = pipeline(
            prompt=prompt,
            image=image,
            mask_image=mask,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.manual_seed(current_seed),
        ).images[0]

        if save_concat:
            # Concat original image, mask, and inpainted image
            concat = PIL.Image.new("RGB", (image.width * 3, image.height))
            concat.paste(image, (0, 0))
            concat.paste(mask, (image.width, 0))
            concat.paste(image_result, (image.width * 2, 0))

            # Save each concatenated image with a unique filename
            concat_save_path = save_path.parent / f"{save_path.stem}_seed_{current_seed}{save_path.suffix}"
            concat.save(concat_save_path)
        else:
            # Save each result image with a unique filename
            result_save_path = save_path.parent / f"{save_path.stem}_seed_{current_seed}{save_path.suffix}"
            image_result.save(result_save_path)

if __name__ == "__main__":
    app()
