"""
Unified LLM Client
===================
Drop-in replacement for Anthropic API calls.
Routes to local Ollama (Qwen 3 4B) by default, with Anthropic fallback.

Usage:
    from technical_analysis.bot.llm_client import llm_chat

    # Returns parsed content string (same as msg.content[0].text)
    result = llm_chat(
        system="You are a market analyst.",
        user="Analyze this post...",
        max_tokens=800,
        temperature=0.3,
        json_mode=True,       # forces valid JSON output (Ollama format mode)
        json_array=False,     # set True if expecting a JSON array
        backend="ollama",     # "ollama" (default) or "anthropic"
        model="qwen3:4b",    # ollama model name, or "haiku"/"sonnet" for anthropic
    )
"""

import json
import os
import time
from typing import Optional


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_DEFAULT_MODEL = "qwen3:4b"


def _ollama_chat(
    system: str,
    user: str,
    max_tokens: int = 800,
    temperature: float = 0.3,
    json_mode: bool = False,
    model: str = OLLAMA_DEFAULT_MODEL,
    timeout: int = 120,
) -> str:
    """Call local Ollama model. Returns response text."""
    import requests as _req

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,  # disable Qwen3 thinking mode — we want direct output
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if json_mode:
        payload["format"] = "json"

    resp = _req.post(OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    content = data.get("message", {}).get("content", "")

    # If thinking mode leaked through despite think=False, fall back to thinking field
    if not content.strip() and data.get("message", {}).get("thinking"):
        content = data["message"]["thinking"]

    return content.strip()


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------

def _anthropic_chat(
    system: str,
    user: str,
    max_tokens: int = 800,
    temperature: float = 0.7,
    model: str = "haiku",
) -> str:
    """Call Anthropic API. Returns response text."""
    import anthropic

    model_id = {
        "haiku": "claude-haiku-4-5",
        "sonnet": "claude-sonnet-4-5-20241022",
        "haiku3": "claude-3-haiku-20240307",
    }.get(model, model)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    if api_key.startswith("sk-ant-oat"):
        client = anthropic.Anthropic(auth_token=api_key)
    else:
        client = anthropic.Anthropic(api_key=api_key)

    kwargs = {
        "model": model_id,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": user}],
    }
    if system:
        kwargs["system"] = system
    if temperature is not None:
        kwargs["temperature"] = temperature

    for attempt in range(3):
        try:
            response = client.messages.create(**kwargs)
            return response.content[0].text.strip()
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def llm_chat(
    system: str = "",
    user: str = "",
    max_tokens: int = 800,
    temperature: float = 0.3,
    json_mode: bool = False,
    json_array: bool = False,
    backend: str = "ollama",
    model: Optional[str] = None,
    timeout: int = 120,
) -> str:
    """
    Unified LLM call. Routes to Ollama (local) or Anthropic (API).

    Returns the raw response text. Caller is responsible for JSON parsing.

    Args:
        system: System prompt
        user: User message
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
        json_mode: If True, force JSON output (Ollama format mode)
        json_array: If True and json_mode, wrap prompt to request array
        backend: "ollama" or "anthropic"
        model: Model name (default: qwen3:4b for ollama, haiku for anthropic)
        timeout: Request timeout in seconds (ollama only)
    """
    if backend == "ollama":
        effective_model = model or OLLAMA_DEFAULT_MODEL
        try:
            return _ollama_chat(
                system=system,
                user=user,
                max_tokens=max_tokens,
                temperature=temperature,
                json_mode=json_mode,
                model=effective_model,
                timeout=timeout,
            )
        except Exception as e:
            # If Ollama is down, fall back to Anthropic
            print(f"  [llm] Ollama failed ({e}), falling back to Anthropic")
            return _anthropic_chat(
                system=system,
                user=user,
                max_tokens=max_tokens,
                temperature=temperature,
                model="haiku",
            )
    else:
        effective_model = model or "haiku"
        return _anthropic_chat(
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
            model=effective_model,
        )


def llm_chat_json(
    system: str = "",
    user: str = "",
    max_tokens: int = 800,
    temperature: float = 0.3,
    backend: str = "ollama",
    model: Optional[str] = None,
) -> dict:
    """Convenience: call llm_chat with json_mode=True and parse the result."""
    import re
    raw = llm_chat(
        system=system, user=user, max_tokens=max_tokens,
        temperature=temperature, json_mode=True,
        backend=backend, model=model,
    )
    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Try to extract JSON object
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        return json.loads(match.group())
    raise ValueError(f"Could not parse JSON from LLM response: {raw[:200]}")


def llm_chat_json_array(
    system: str = "",
    user: str = "",
    max_tokens: int = 800,
    temperature: float = 0.3,
    backend: str = "ollama",
    model: Optional[str] = None,
) -> list:
    """Convenience: call llm_chat and parse a JSON array from the result."""
    import re
    # For Ollama json_mode, we can't force array output directly,
    # so we wrap it: ask for {"items": [...]} and unwrap
    wrapped_user = user + "\n\nReturn your answer as a JSON object with key \"items\" containing an array of results."
    raw = llm_chat(
        system=system, user=wrapped_user, max_tokens=max_tokens,
        temperature=temperature, json_mode=True,
        backend=backend, model=model,
    )
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and "items" in parsed:
            return parsed["items"]
        # Try first list-valued key
        for v in parsed.values():
            if isinstance(v, list):
                return v
        return [parsed]
    except json.JSONDecodeError:
        pass
    # Fallback: extract array
    match = re.search(r"\[[\s\S]*\]", raw)
    if match:
        return json.loads(match.group())
    raise ValueError(f"Could not parse JSON array from LLM response: {raw[:200]}")
