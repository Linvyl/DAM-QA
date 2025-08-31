import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None
_tokenizer = None

USER_PROMPT_TEMPLATE = (
    "Answer each question concisely in a single word or short phrase, without any lengthy descriptions or explanations.\n"
    "Rely only on information that is clearly visible in the provided image.\n"
    "If the answer cannot be determined from the image, respond with 'unanswerable'.\n"
)


def get_model(model_dir="openbmb/MiniCPM-o-2_6"):
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        _model = AutoModel.from_pretrained(
            model_dir,
            trust_remote_code=True,
            attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16,
            init_vision=True,
            init_audio=False,
            init_tts=False,
            low_cpu_mem_usage=True,
            device_map="auto"
        ).eval()
        _tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=True)
        _model.init_tts()
    return _model, _tokenizer


def inference(questions, image_paths, config):
    model, tokenizer = get_model()
    question, image_path = questions[0], image_paths[0]
    image = Image.open(image_path).convert('RGB')
    msgs = [
        {'role': 'system', 'content': USER_PROMPT_TEMPLATE},
        {'role': 'user', 'content': [
            image, f"Question: {question.strip()}\nAnswer:"]
         }
    ]

    with torch.no_grad():
        res = model.chat(
            image=image,
            msgs=msgs,
            tokenizer=tokenizer
        )

    answer = res.splitlines()[0].strip()

    return [answer]
