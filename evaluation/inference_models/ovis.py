import os
import torch
from transformers import AutoModelForCausalLM
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None
_text_tokenizer = None
_visual_tokenizer = None

USER_PROMPT_TEMPLATE = (
    "Answer each question concisely in a single word or short phrase, without any lengthy descriptions or explanations.\n"
    "Rely only on information that is clearly visible in the provided image.\n"
    "If the answer cannot be determined from the image, respond with 'unanswerable'.\n"
)


def get_model(model_dir=None):
    global _model, _text_tokenizer, _visual_tokenizer
    
    if model_dir is None:
        model_dir = os.getenv("OVIS_MODEL_PATH", "AIDC-AI/Ovis1.6-Gemma2-9B")
    
    if _model is None or _text_tokenizer is None or _visual_tokenizer is None:
        _model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            multimodal_max_length=32768,
            trust_remote_code=True).eval().to(device)
        _text_tokenizer = _model.get_text_tokenizer()
        _visual_tokenizer = _model.get_visual_tokenizer()

    return _model, _text_tokenizer, _visual_tokenizer


def inference(questions, image_paths, config):
    model, text_tokenizer, visual_tokenizer = get_model()
    batch_inputs = [
        (img_path, f"Question: {question.strip()}\nAnswer:") for img_path, question in zip(image_paths, questions)
    ]
    batch_input_ids = []
    batch_attention_mask = []
    batch_pixel_values = []

    for image_path, text in batch_inputs:
        image = Image.open(image_path)
        query = f"{USER_PROMPT_TEMPLATE}<image>\n{text}"
        prompt, input_ids, pixel_values = model.preprocess_inputs(
            query, [image], max_partition=9)
        attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
        batch_input_ids.append(input_ids.to(device=model.device))
        batch_attention_mask.append(attention_mask.to(device=model.device))
        batch_pixel_values.append(pixel_values.to(
            dtype=visual_tokenizer.dtype, device=visual_tokenizer.device))

    batch_input_ids = torch.nn.utils.rnn.pad_sequence([i.flip(dims=[0]) for i in batch_input_ids], batch_first=True,
                                                      padding_value=0.0).flip(dims=[1])
    batch_input_ids = batch_input_ids[:, -model.config.multimodal_max_length:]
    batch_attention_mask = torch.nn.utils.rnn.pad_sequence([i.flip(dims=[0]) for i in batch_attention_mask],
                                                           batch_first=True, padding_value=False).flip(dims=[1])
    batch_attention_mask = batch_attention_mask[:, -
                                                model.config.multimodal_max_length:]

    with torch.inference_mode():
        gen_kwargs = dict(
            max_new_tokens=config["max_new_tokens"],
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
            repetition_penalty=None,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=text_tokenizer.pad_token_id,
            use_cache=True
        )
        output_ids = model.generate(batch_input_ids, pixel_values=batch_pixel_values, attention_mask=batch_attention_mask,
                                    **gen_kwargs)
    outputs = [
        text_tokenizer.decode(ids, skip_special_tokens=True)
        for ids in output_ids
    ]

    answers = [out.splitlines()[0].strip() for out in outputs]

    return answers
