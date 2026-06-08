from openai import OpenAI
import base64
from pathlib import Path
from typing import List, Optional, Union
import os
import time
import random

try:
    from openai import (
        APIConnectionError,
        APITimeoutError,
        APIStatusError,
        RateLimitError,
        InternalServerError,
    )
except Exception:  # pragma: no cover - keeps compatibility across openai versions
    APIConnectionError = APITimeoutError = APIStatusError = RateLimitError = InternalServerError = ()  # type: ignore

def _get_env_or_default(key: str, default: str) -> str:
    value = os.getenv(key)
    return value if value else default


# client = OpenAI()
client = OpenAI(
    api_key=_get_env_or_default("OPENAI_API_KEY", "sk-CwYAuaV96a3Tu9pEQH3SLkTBypGJocjM66hOiJGrLJyVCqLL"),
    base_url=_get_env_or_default("OPENAI_BASE_URL", "https://hk.xty.app/v1"),
    timeout=float(_get_env_or_default("OPENAI_TIMEOUT", "60.0")),
    max_retries=int(_get_env_or_default("OPENAI_MAX_RETRIES", "2")),
)

def encode_image(image_path: str) -> str:
    """将图像文件编码为base64字符串
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        base64编码的图像字符串
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def _extract_response_content(response) -> str:
    if isinstance(response, str):
        return response
    if hasattr(response, "choices"):
        return response.choices[0].message.content
    if isinstance(response, dict):
        if "choices" in response and response["choices"]:
            return response["choices"][0]["message"]["content"]
        if "text" in response:
            return response["text"]
    raise TypeError(f"Unexpected response type: {type(response).__name__}")

def _get_env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except Exception:
        return default


def _get_env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except Exception:
        return default


def _retry_after_seconds(exc: Exception) -> Optional[float]:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", {}) or {}
    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if retry_after is None:
        return None
    try:
        return max(0.0, float(retry_after))
    except Exception:
        return None


def _is_retryable_openai_error(exc: Exception) -> bool:
    retryable_classes = tuple(
        cls
        for cls in (
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
            InternalServerError,
        )
        if isinstance(cls, type)
    )
    if retryable_classes and isinstance(exc, retryable_classes):
        return True

    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
    try:
        status_code = int(status_code) if status_code is not None else None
    except Exception:
        status_code = None
    if status_code in {408, 409, 429, 500, 502, 503, 504, 520, 522, 524}:
        return True
    if status_code in {400, 401, 403, 404, 422}:
        return False

    msg = str(exc).lower()
    retryable_markers = [
        "connection error",
        "connection reset",
        "connection aborted",
        "timed out",
        "timeout",
        "temporarily unavailable",
        "bad gateway",
        "gateway timeout",
        "service unavailable",
        "rate limit",
        "too many requests",
        "502",
        "503",
        "504",
    ]
    return any(marker in msg for marker in retryable_markers)


def _call_with_openai_recovery(fn, *args, **kwargs):
    """Call OpenAI and wait through transient provider/network outages.

    Environment knobs:
    - OPENAI_RECOVERY_MAX_ATTEMPTS: 0 means retry forever, default 0.
    - OPENAI_RECOVERY_INITIAL_WAIT: first wait in seconds, default 30.
    - OPENAI_RECOVERY_MAX_WAIT: cap per wait in seconds, default 600.
    - OPENAI_RECOVERY_BACKOFF: exponential multiplier, default 2.0.
    - OPENAI_RECOVERY_JITTER: jitter ratio, default 0.1.
    """
    max_attempts = _get_env_int("OPENAI_RECOVERY_MAX_ATTEMPTS", 0)
    initial_wait = max(1.0, _get_env_float("OPENAI_RECOVERY_INITIAL_WAIT", 30.0))
    max_wait = max(initial_wait, _get_env_float("OPENAI_RECOVERY_MAX_WAIT", 600.0))
    backoff = max(1.0, _get_env_float("OPENAI_RECOVERY_BACKOFF", 2.0))
    jitter_ratio = max(0.0, _get_env_float("OPENAI_RECOVERY_JITTER", 0.1))

    attempt = 0
    wait_s = initial_wait
    while True:
        attempt += 1
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if not _is_retryable_openai_error(exc):
                raise
            if max_attempts > 0 and attempt >= max_attempts:
                print(
                    f"[OpenAI恢复重试] 已达到最大尝试次数 {max_attempts}，最后错误: "
                    f"{type(exc).__name__}: {exc}"
                )
                raise

            retry_after = _retry_after_seconds(exc)
            sleep_s = retry_after if retry_after is not None else wait_s
            if jitter_ratio > 0 and retry_after is None:
                sleep_s += random.uniform(0.0, sleep_s * jitter_ratio)
            print(
                "[OpenAI恢复重试] 遇到可恢复的 OpenAI/API 连接错误，"
                f"attempt={attempt}, error={type(exc).__name__}: {exc}. "
                f"等待 {sleep_s:.1f}s 后继续重试；这不会计为系统/skill失败。"
            )
            time.sleep(sleep_s)
            wait_s = min(max_wait, wait_s * backoff)

def create_message_with_image(text: str, image_path: Optional[str] = None) -> dict:
    """创建包含文本和图像的消息
    
    Args:
        text: 文本内容
        image_path: 图像文件路径（可选）
        
    Returns:
        包含文本和图像的消息字典
    """
    message = {
        "role": "user",
        "content": []
    }
    
    # 添加文本内容
    if text:
        message["content"].append({
            "type": "text",
            "text": text
        })
    
    # 添加图像内容
    if image_path:
        base64_image = encode_image(image_path)
        message["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })
    
    return message

def gpt_single_image_inference(
    image_path: str, 
    prompt: str, 
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: Optional[int] = 4096  # 设置合理的默认值
) -> str:
    """GPT单图像推理函数
    
    Args:
        image_path: 图像文件路径
        prompt: 提示文本
        model: 使用的模型名称
        temperature: 温度参数，控制输出的随机性
        max_tokens: 最大输出token数
        
    Returns:
        GPT的回复文本
    """
    try:
        print(f"[推理开始] 模型: {model}, 图像: {image_path}")
        print(f"[参数] max_tokens: {max_tokens}, temperature: {temperature}")
        
        message = create_message_with_image(prompt, image_path)
        
        response = _call_with_openai_recovery(
            client.chat.completions.create,
            model=model,
            messages=[message],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        result = _extract_response_content(response)
        print(f"[推理成功] 返回长度: {len(result)}")
        return result
        
    except Exception as e:
        error_msg = f"推理失败: {type(e).__name__}: {str(e)}"
        print(f"[推理失败] {error_msg}")
        import traceback
        traceback.print_exc()
        raise

def gpt_multiple_images_inference(
    image_paths: List[str], 
    prompt: str, 
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: Optional[int] = 4096  # 设置合理的默认值
) -> str:
    """GPT多图像推理函数
    
    Args:
        image_paths: 图像文件路径列表
        prompt: 提示文本
        model: 使用的模型名称
        temperature: 温度参数，控制输出的随机性
        max_tokens: 最大输出token数
        
    Returns:
        GPT的回复文本
    """
    max_tokens = 4096
    try:
        print(f"[推理开始] 模型: {model}, 图像数量: {len(image_paths)}")
        print(f"[参数] max_tokens: {max_tokens}, temperature: {temperature}")
        
        message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
        
        # 添加所有图像（过滤无效路径）
        valid_images = 0
        for image_path in image_paths:
            if image_path is not None and os.path.exists(image_path):
                try:
                    base64_image = encode_image(image_path)
                    message["content"].append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })
                    valid_images += 1
                except Exception as e:
                    print(f"Warning: Failed to encode image {image_path}: {e}")
                    continue
        
        print(f"[图像处理] 成功编码 {valid_images}/{len(image_paths)} 张图像")
        
        response = _call_with_openai_recovery(
            client.chat.completions.create,
            model=model,
            messages=[message],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        result = _extract_response_content(response)
        print(f"[推理成功] 返回长度: {len(result)}")
        return result
        
    except Exception as e:
        error_msg = f"推理失败: {type(e).__name__}: {str(e)}"
        print(f"[推理失败] {error_msg}")
        import traceback
        traceback.print_exc()
        raise

def gpt_text_only_inference(
    prompt: str, 
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: Optional[int] = 4096  # 设置合理的默认值
) -> str:
    """GPT纯文本推理函数
    
    Args:
        prompt: 提示文本
        model: 使用的模型名称
        temperature: 温度参数，控制输出的随机性
        max_tokens: 最大输出token数
        
    Returns:
        GPT的回复文本
    """
    try:
        print(f"[推理开始] 模型: {model}, 纯文本推理")
        print(f"[参数] max_tokens: {max_tokens}, temperature: {temperature}")
        
        response = _call_with_openai_recovery(
            client.chat.completions.create,
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        result = _extract_response_content(response)
        print(f"[推理成功] 返回长度: {len(result)}")
        return result
        
    except Exception as e:
        error_msg = f"推理失败: {type(e).__name__}: {str(e)}"
        print(f"[推理失败] {error_msg}")
        import traceback
        traceback.print_exc()
        raise

# 使用示例
if __name__ == "__main__":
    # 单图像推理示例
    result = gpt_single_image_inference(
        image_path="assets/image.png",
        prompt="请描述这张图片中的内容"
    )
    print("单图像推理结果:", result)
    
    # 多图像推理示例
    result = gpt_multiple_images_inference(
        image_paths=["assets/image1.png", "assets/image2.png"],
        prompt="请比较这两张图片的差异"
    )
    print("多图像推理结果:", result)
    
    # 纯文本推理示例
    result = gpt_text_only_inference(
        prompt="Write a one-sentence bedtime story about a unicorn."
    )
    print("纯文本推理结果:", result)
