import os
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_processor = None

USER_PROMPT_TEMPLATE = (
    "Answer each question concisely in a single word or short phrase, without any lengthy descriptions or explanations.\n"
    "Rely only on information that is clearly visible in the provided image.\n"
    "If the answer cannot be determined from the image, respond with 'unanswerable'.\n"
)


def get_model(model_dir="DAMO-NLP-SG/VideoLLaMA3-7B-Image"):
    global _model, _processor
    if _model is None or _processor is None:
        _processor = AutoProcessor.from_pretrained(
            model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
        )
        _model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
            device_map="auto"
        ).eval()
    return _model, _processor


def inference(questions, image_paths, config):
    question, image_path = questions[0], image_paths[0]
    model, processor = get_model()

    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": USER_PROMPT_TEMPLATE}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": {"image_path": image_path}},
                {"type": "text", "text": f"Question: {question.strip()}\nAnswer:"}
            ]
        }
    ]

    inputs = processor(conversation=conversation, return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor)
              else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
        output_ids = model.generate(
            **inputs, max_new_tokens=config.max_new_tokens)

    answer = processor.batch_decode(
        output_ids, skip_special_tokens=True)[0].strip()

    return [answer]
