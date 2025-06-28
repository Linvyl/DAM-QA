"""Phi Model Inference Module"""

import os
import torch
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model instances
_model = None
_processor = None

USER_PROMPT_TEMPLATE = (
    "Answer each question concisely in a single word or short phrase, without any lengthy descriptions or explanations.\n"
    "Rely only on information that is clearly visible in the provided image.\n"
    "If the answer cannot be determined from the image, respond with 'unanswerable'.\n"
)


def get_model(model_dir=None):
    """Load Phi model and processor."""
    global _model, _processor
    
    if model_dir is None:
        model_dir = os.getenv("PHI_MODEL_PATH", "microsoft/Phi-3.5-vision-instruct")
    
    if _model is None or _processor is None:
        _processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        
        _model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2",
        ).eval().to(device)
        
    return _model, _processor


def inference(questions: List[str], image_paths: List[str], config: Dict[str, Any]) -> List[str]:
    """Main inference function for Phi model."""
    model, processor = get_model()
    question, image_path = questions[0], image_paths[0]
    
    # Prepare chat messages
    chat = [
        {"role": "system", "content": USER_PROMPT_TEMPLATE},
        {"role": "user", "content": f"<|image_1|>\nQuestion: {question.strip()}\nAnswer:"}
    ]
    
    prompt = processor.tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    
    # Remove trailing token if present
    if prompt.endswith('<|endoftext|>'):
        prompt = prompt.rstrip('<|endoftext|>')
    
    # Load and process image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    
    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=config["max_new_tokens"])
    
    # Decode response
    generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return [response.splitlines()[0].strip()]
