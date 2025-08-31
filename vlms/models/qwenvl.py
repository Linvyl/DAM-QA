"""QwenVL Model Inference Module"""

import torch
from typing import List, Dict, Any, Tuple, Optional
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model instances
_model: Optional[Qwen2_5_VLForConditionalGeneration] = None
_processor: Optional[AutoProcessor] = None

USER_PROMPT = (
    "Answer each question concisely in a single word or short phrase, without any lengthy descriptions or explanations.\n"
    "Rely only on information that is clearly visible in the provided image.\n"
    "If the answer cannot be determined from the image, respond with 'unanswerable'."
)


def get_model(model_dir: str = "Qwen/Qwen2.5-VL-7B-Instruct") -> Tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    """Initialize and return the QwenVL model and processor."""
    global _model, _processor
    
    if _model is None or _processor is None:
        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16
        ).eval().to(device)
        
        _processor = AutoProcessor.from_pretrained(
            model_dir,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
            use_fast=True
        )
        _processor.tokenizer.padding_side = "left"
    
    return _model, _processor


def create_message_batch(questions: List[str], image_paths: List[str]) -> List[List[Dict[str, Any]]]:
    """Create message batch for model input."""
    return [
        [
            {"role": "system", "content": [{"type": "text", "text": USER_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{img_path}"},
                    {"type": "text", "text": f"Question: {question.strip()}\nAnswer:"}
                ]
            }
        ]
        for question, img_path in zip(questions, image_paths)
    ]


def inference(questions: List[str], image_paths: List[str], config: Dict[str, Any]) -> List[str]:
    """Run inference on QwenVL model."""
    model, processor = get_model()
    
    messages_batch = create_message_batch(questions, image_paths)
    
    # Apply chat template
    texts = [
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_batch
    ]
    
    # Process vision inputs
    image_inputs, video_inputs = process_vision_info(messages_batch)
    
    # Prepare model inputs
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)
    
    # Generate responses
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=config.max_new_tokens)
    
    # Decode and clean responses
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    decoded = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return [text.splitlines()[0].strip() for text in decoded]
