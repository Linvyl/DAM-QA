"""InternVL Model Inference Module"""

import os
import torch
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 448

# Global model instances
_model: Optional[AutoModelForCausalLM] = None
_tokenizer: Optional[AutoTokenizer] = None

USER_PROMPT = (
    "Answer each question concisely in a single word or short phrase, without any lengthy descriptions or explanations.\n"
    "Rely only on information that is clearly visible in the provided image.\n"
    "If the answer cannot be determined from the image, respond with 'unanswerable'."
)

# Pre-built transform
_transform = T.Compose([
    T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


def find_closest_aspect_ratio(aspect_ratio: float, target_ratios: List[Tuple[int, int]], 
                             width: int, height: int, image_size: int) -> Tuple[int, int]:
    """Find the target aspect ratio closest to the original image's aspect ratio."""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_ar)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image: Image.Image, min_num: int = 1, max_num: int = 12, 
                      image_size: int = IMAGE_SIZE, use_thumbnail: bool = False) -> List[Image.Image]:
    """Preprocess image by dividing it into blocks based on aspect ratio."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Create target aspect ratios
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1]
    )
    
    target_ar = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_ar[0]
    target_height = image_size * target_ar[1]
    blocks = target_ar[0] * target_ar[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        processed_images.append(resized_img.crop(box))
    
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    
    return processed_images


def load_image(image_file: str, input_size: int = IMAGE_SIZE, max_num: int = 12) -> torch.Tensor:
    """Load and preprocess an image file into a tensor."""
    image = Image.open(image_file).convert('RGB')
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [_transform(img) for img in images]
    return torch.stack(pixel_values)


def get_model(model_dir: str = None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Initialize and return the InternVL model and tokenizer."""
    global _model, _tokenizer
    
    if model_dir is None:
        model_dir = os.getenv("INTERNVL_MODEL_PATH", "OpenGVLab/InternVL2.5-8B")
    
    if _model is None or _tokenizer is None:
        _model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True,
            device_map="auto",
        ).eval()
        
        _tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=True, use_fast=False
        )
    
    return _model, _tokenizer


def inference(questions: List[str], image_paths: List[str], config: Dict[str, Any]) -> List[str]:
    """Run inference on InternVL model."""
    model, tokenizer = get_model()
    device = next(model.parameters()).device

    # Prepare batch prompts
    batch_prompts = [
        f"{USER_PROMPT}<image>\nQuestion: {q.strip()}\nAnswer:" for q in questions
    ]

    # Load and process images
    batch_tensors = []
    num_patches_list = []
    
    for img_path in image_paths:
        tv = load_image(img_path).to(torch.bfloat16).to(device)
        batch_tensors.append(tv)
        num_patches_list.append(tv.size(0))

    pixel_values = torch.cat(batch_tensors, dim=0)
    
    # Generate responses
    with torch.no_grad():
        responses = model.batch_chat(
            tokenizer,
            pixel_values,
            num_patches_list=num_patches_list,
            questions=batch_prompts,
            generation_config={
                "max_new_tokens": config["max_new_tokens"],
                "pad_token_id": tokenizer.eos_token_id
            },
        )

    return [resp.splitlines()[0].strip() for resp in responses]
