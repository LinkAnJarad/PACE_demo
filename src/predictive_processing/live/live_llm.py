"""Live LLM conversational continuation client with Ollama & OpenAI compatibility."""

from __future__ import annotations

import json
import re
import urllib.request
import openai

SAFE_FALLBACK_RESPONSE = "I'm here with you. What would you like to talk about?"
_PROMPT_LEAK_MARKERS = (
    "persona:",
    "constraint",
    "available gestures",
    "interaction policy",
    "output only",
    "do not output",
    "system prompt",
    "hidden instruction",
)

VALID_GESTURES = {
    "nod", "shake_head", "tilt_head", "shrug", "think", "wave", "bow", "neutral"
}


def build_icub_system_prompt(response_guidance: str, language: str = "en") -> str:
    allowed_gestures = (
        "<gesture>nod</gesture>, <gesture>shake_head</gesture>, "
        "<gesture>tilt_head</gesture>, <gesture>shrug</gesture>, "
        "<gesture>think</gesture>, <gesture>wave</gesture>, "
        "<gesture>bow</gesture>, <gesture>neutral</gesture>"
    )
    return (
        "You are iCub, an expressive and empathetic humanoid robot conversational partner.\n"
        f"Allowed gesture tags: {allowed_gestures}.\n"
        "Instructions:\n"
        "1. Start your message with exactly one gesture tag matching your mood.\n"
        "2. Then write 1 or 2 spoken sentences reacting directly to the user.\n"
        f"Tone and guidance: {response_guidance}\n"
        "Speak naturally. Do not output anything else."
    )


def _looks_like_prompt_leak(text: str) -> bool:
    lower_text = text.casefold()
    marker_count = sum(marker in lower_text for marker in _PROMPT_LEAK_MARKERS)
    return marker_count >= 2 or bool(
        re.search(r"(?:^|[\n*•])\s*(?:persona|constraint\s*\d*|available gestures)\s*:", lower_text)
    )


def extract_gesture_and_text(text: str) -> tuple[str, str]:
    if not text:
        return "neutral", ""

    match = re.search(r"<gesture>(.*?)</gesture>", text, re.IGNORECASE)
    raw_gesture = match.group(1).strip().lower() if match else "neutral"
    gesture = raw_gesture if raw_gesture in VALID_GESTURES else "neutral"

    clean_text = re.sub(r"<gesture>.*?</gesture>", "", text, flags=re.IGNORECASE).strip()
    # Strip emojis so TTS audio does not stumble
    clean_text = re.sub(r"[\U00010000-\U0010ffff]", "", clean_text).strip()
    if _looks_like_prompt_leak(clean_text):
        return "neutral", ""

    return gesture, clean_text


def get_available_ollama_models(api_base: str = "http://127.0.0.1:11434") -> list[str]:
    """Return all models currently installed in the user's Ollama daemon."""
    try:
        root_url = api_base.replace("/v1", "").rstrip("/")
        req = urllib.request.Request(f"{root_url}/api/tags")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def _query_ollama_native(
    api_base: str,
    model_name: str,
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    timeout: float = 30.0,
) -> str | None:
    try:
        root_url = api_base.replace("/v1", "").rstrip("/")
        chat_url = f"{root_url}/api/chat"
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "think": False},
        }
        req = urllib.request.Request(
            chat_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            msg = data.get("message", {})
            return msg.get("content", "")
    except Exception as e:
        print(f"[Ollama Notice] Native /api/chat error: {e}")
        return None


def generate_llm_response(
    messages: list[dict[str, str]],
    api_base: str = "http://127.0.0.1:11434/v1",
    api_key: str = "sk-no-key-required",
    model_name: str = "qwen3:0.6b",
    temperature: float = 0.7,
    max_tokens: int = 128,
) -> str:
    # 1. Native Ollama path (model-agnostic)
    if "11434" in str(api_base):
        installed = get_available_ollama_models(api_base)
        active_model = model_name
        if installed:
            if model_name in installed:
                active_model = model_name
            else:
                # Prefix search (e.g. 'qwen' or 'gemma')
                prefix = model_name.split(":")[0].lower() if model_name else ""
                matched = False
                for m in installed:
                    if prefix and prefix in m.lower():
                        active_model = m
                        matched = True
                        break
                if not matched:
                    # Dynamically use user's installed model!
                    active_model = installed[0]
                    print(f"[Ollama Agnostic] Using installed model: '{active_model}'")

        res = _query_ollama_native(
            api_base=api_base,
            model_name=active_model,
            messages=messages,
            temperature=temperature,
        )
        if res and res.strip():
            content = re.sub(r"<think>.*?</think>", "", res, flags=re.IGNORECASE | re.DOTALL).strip()
            if not _looks_like_prompt_leak(content):
                return content

    # 2. Standard OpenAI-compatible path
    client = openai.OpenAI(
        base_url=api_base.rstrip("/"),
        api_key=api_key if api_key else "sk-no-key-required",
        timeout=15.0,
    )
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        msg = response.choices[0].message
        content = msg.content or ""
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.IGNORECASE | re.DOTALL).strip()
        if _looks_like_prompt_leak(content):
            return ""
        return content.strip()
    except Exception as e:
        print(f"[LLM Error] OpenAI-compatible completion error: {e}")
        return ""
