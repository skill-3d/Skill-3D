import os
from openai import OpenAI
from typing import List, Optional

try:
    from .qwen import encode_image, create_message_with_image
except ImportError:
    from qwen import encode_image, create_message_with_image


def _get_env_or_default(key: str, default: str) -> str:
    value = os.getenv(key)
    return value if value else default


def _env_flag(key: str, default: bool = False) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _vllm_extra_body() -> dict:
    # Skill-3D protocol tags are tokenizer special tokens after SFT. vLLM
    # skips special tokens by default, which strips tags from decoded text.
    return {"skip_special_tokens": _env_flag("SPAGENT_VLLM_SKIP_SPECIAL_TOKENS", False)}


# Initialize Qwen client with vLLM configuration
client = OpenAI(
    api_key=_get_env_or_default("OPENAI_API_KEY", "dummy"),
    base_url=_get_env_or_default("OPENAI_BASE_URL", "http://10.8.131.51:30058/v1"),
)


def qwen_single_image_inference(
    image_path: str,
    prompt: str,
    model: str = "qwen-vl-2B",
    temperature: float = 0.1,
    max_tokens: Optional[int] = None,
) -> str:
    """Qwen single-image inference through an OpenAI-compatible vLLM server."""
    try:
        message = create_message_with_image(prompt, image_path)
        response = client.chat.completions.create(
            model=model,
            messages=[message],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=_vllm_extra_body(),
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"推理过程中出错: {e}")
        print(f"错误类型: {type(e).__name__}")
        return f"推理失败: {str(e)}"


def qwen_multiple_images_inference(
    image_paths: List[str],
    prompt: str,
    model: str = "qwen-vl-2B",
    temperature: float = 0.1,
    max_tokens: Optional[int] = None,
) -> str:
    """Qwen multi-image inference through an OpenAI-compatible vLLM server."""
    message = {"role": "user", "content": [{"type": "text", "text": prompt}]}
    for image_path in image_paths:
        base64_image = encode_image(image_path)
        message["content"].append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        )

    response = client.chat.completions.create(
        model=model,
        messages=[message],
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=_vllm_extra_body(),
    )
    return response.choices[0].message.content


def qwen_text_only_inference(
    prompt: str,
    model: str = "qwen-vl-2B",
    temperature: float = 0.1,
    max_tokens: Optional[int] = None,
) -> str:
    """Qwen text-only inference through an OpenAI-compatible vLLM server."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=_vllm_extra_body(),
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    result = qwen_single_image_inference(image_path="assets/example.png", prompt="What is in the image?")
    print(result)
