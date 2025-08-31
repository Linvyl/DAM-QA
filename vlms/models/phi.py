"""Phi Model Inference Module"""

import torch
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model instances
_model: Optional[Any] = None
_processor: Optional[Any] = None

USER_PROMPT = (
    "Answer each question concisely in a single word or short phrase, without any lengthy descriptions or explanations.\n"
    "Rely only on information that is clearly visible in the provided image.\n"
    "If the answer cannot be determined from the image, respond with 'unanswerable'."
)


def get_model(model_dir: str = "/mnt/dataset1/pretrained_fm/microsoft_Phi-4-multimodal-instruct") -> Tuple[Any, Any]:
    """Initialize and return the Phi model and processor."""
    global _model, _processor
    
    if _model is None or _processor is None:
        _processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        
        _model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2",
        ).eval().to(device)
    
    return _model, _processor


def create_chat_message(question: str) -> List[Dict[str, str]]:
    """Create chat message format for Phi model."""
    return [
        {"role": "system", "content": USER_PROMPT},
        {"role": "user", "content": f"<|image_1|>\nQuestion: {question.strip()}\nAnswer:"}
    ]


def inference(questions: List[str], image_paths: List[str], config: Dict[str, Any]) -> List[str]:
    """Run inference on Phi model."""
    model, processor = get_model()
    
    # Process single question/image pair (Phi doesn't support batch)
    question, image_path = questions[0], image_paths[0]
    
    # Prepare chat and prompt
    chat = create_chat_message(question)
    prompt = processor.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    
    # Remove trailing token if present
    if prompt.endswith('<|endoftext|>'):
        prompt = prompt.rstrip('<|endoftext|>')
    
    # Load and process image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    
    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=config.max_new_tokens)
    
    # Decode response
    generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return [response.splitlines()[0].strip()]
