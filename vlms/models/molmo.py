"""Molmo Model Inference Module"""

import os
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image

os.environ["TRANSFORMERS_NO_TF"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model instances
_model: Optional[Any] = None
_processor: Optional[Any] = None

USER_PROMPT = (
    "Answer each question concisely in a single word or short phrase, without any lengthy descriptions or explanations.\n"
    "Rely only on information that is clearly visible in the provided image.\n"
    "If the answer cannot be determined from the image, respond with 'unanswerable'."
)

# Patch numpy stack to fix dtype issues
orig_stack = np.stack


def patched_stack(arrays, axis=0, *args, **kwargs):
    """Patched numpy stack that removes dtype argument."""
    kwargs.pop("dtype", None)
    return orig_stack(arrays, axis=axis, *args, **kwargs)


np.stack = patched_stack


def get_model(model_dir: str = "allenai/Molmo-7B-D-0924") -> Tuple[Any, Any]:
    """Initialize and return the Molmo model and processor."""
    global _model, _processor
    
    if _model is None or _processor is None:
        _processor = AutoProcessor.from_pretrained(
            model_dir, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16, 
            device_map="auto", 
            use_fast=True
        )
        
        _model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto"
        ).eval()
    
    return _model, _processor


def inference(questions: List[str], image_paths: List[str], config: Dict[str, Any]) -> List[str]:
    """Run inference on Molmo model."""
    model, processor = get_model()
    
    # Process single question/image pair
    question, image_path = questions[0], image_paths[0]
    prompt = f"{USER_PROMPT}Question: {question.strip()}\nAnswer:"
    
    # Load and process image
    image = Image.open(image_path).convert("RGB")
    inputs = processor.process(images=[image], text=prompt)
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
    inputs["images"] = inputs["images"].to(torch.bfloat16)
    
    # Generate response
    with torch.no_grad(), torch.autocast(
        device_type="cuda", 
        enabled=torch.cuda.is_available(), 
        dtype=torch.bfloat16
    ):
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(
                max_new_tokens=config.max_new_tokens,
                stop_strings="<|endoftext|>"
            ),
            tokenizer=processor.tokenizer
        )
    
    # Decode response
    generated_tokens = output[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return [generated_text.splitlines()[0].strip()]
