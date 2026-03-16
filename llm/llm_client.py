# llm/llm_client.py

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


# ============================================================
# Config Dataclass
# ============================================================

@dataclass
class LLMConfig:
    provider: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 1200
    timeout: int = 120
    api_key: Optional[str] = None

    # Optional provider-specific settings
    base_url: Optional[str] = None
    api_version: Optional[str] = None


# ============================================================
# Base Client
# ============================================================

class BaseLLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        raise NotImplementedError("Subclasses must implement generate().")

    def invoke(self, prompt: str) -> str:
        return self.generate(prompt=prompt)

    @staticmethod
    def _extract_text_from_content(content: Any) -> str:
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    elif "text" in item:
                        parts.append(str(item["text"]))
                else:
                    parts.append(str(item))
            return "\n".join(p for p in parts if p).strip()

        return str(content)


# ============================================================
# Groq Client
# Endpoint style similar to OpenAI Chat Completions
# ============================================================

# ============================================================
# Groq Client
# ============================================================

class GroqLLMClient(BaseLLMClient):

    def __init__(self, config: LLMConfig):
        super().__init__(config)

        # Load API key
        self.api_key = (config.api_key or os.getenv("GROQ_API_KEY", "")).strip()

        if not self.api_key:
            raise ValueError(
                "❌ GROQ_API_KEY not found.\n"
                "Create a .env file in project root:\n"
                "GROQ_API_KEY=gsk_xxxxxxxxx"
            )

        # Correct Groq endpoint
        self.base_url = config.base_url or "https://api.groq.com/openai/v1/chat/completions"

        print("✅ Groq client initialized")
        print(f"🔑 Using key: {self.api_key[:10]}********")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> str:

        # Build message list
        payload_messages = []

        if messages:
            payload_messages = messages
        else:
            if system_prompt:
                payload_messages.append(
                    {"role": "system", "content": system_prompt}
                )

            payload_messages.append(
                {"role": "user", "content": prompt}
            )

        payload = {
            "model": self.config.model,
            "messages": payload_messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout,
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"Groq API Error {response.status_code}:\n{response.text}"
                )

            data = response.json()

            return (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Groq request failed: {e}")


# ============================================================
# Hugging Face Inference Client
# Works with text generation / chat-compatible hosted endpoints
# ============================================================

class HuggingFaceLLMClient(BaseLLMClient):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.api_key = config.api_key or os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
        if not self.api_key:
            raise ValueError("Missing HUGGINGFACE_API_KEY or HF_TOKEN for Hugging Face provider.")

        # Two modes:
        # 1. base_url explicitly given -> use that
        # 2. otherwise infer standard inference endpoint from model name
        if config.base_url:
            self.base_url = config.base_url
        else:
            self.base_url = f"https://api-inference.huggingface.co/models/{config.model}"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        # For maximum compatibility, convert chat messages to a single prompt
        if messages:
            final_prompt = self._messages_to_prompt(messages)
        else:
            if system_prompt:
                final_prompt = f"System:\n{system_prompt}\n\nUser:\n{prompt}\n\nAssistant:"
            else:
                final_prompt = prompt

        payload = {
            "inputs": final_prompt,
            "parameters": {
                "temperature": self.config.temperature,
                "max_new_tokens": self.config.max_tokens,
                "return_full_text": False,
            },
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=self.config.timeout,
        )
        response.raise_for_status()

        data = response.json()

        # Common hosted inference response patterns
        if isinstance(data, list) and len(data) > 0:
            first = data[0]
            if isinstance(first, dict):
                if "generated_text" in first:
                    return str(first["generated_text"]).strip()
                if "summary_text" in first:
                    return str(first["summary_text"]).strip()

        if isinstance(data, dict):
            if "generated_text" in data:
                return str(data["generated_text"]).strip()
            if "error" in data:
                raise ValueError(f"Hugging Face inference error: {data['error']}")

        return str(data).strip()

    @staticmethod
    def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
        parts = []
        for m in messages:
            role = m.get("role", "user").capitalize()
            content = m.get("content", "")
            parts.append(f"{role}:\n{content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)


# ============================================================
# Google Gemini Client
# Uses Generative Language API
# ============================================================

class GoogleGeminiLLMClient(BaseLLMClient):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.api_key = config.api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing GOOGLE_API_KEY or GEMINI_API_KEY for Google provider.")

        model_name = config.model
        self.base_url = config.base_url or (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        contents = []

        if messages:
            for msg in messages:
                role = "user" if msg.get("role") in ["user", "system"] else "model"
                contents.append(
                    {
                        "role": role,
                        "parts": [{"text": msg.get("content", "")}],
                    }
                )
        else:
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"System Instructions:\n{system_prompt}\n\nUser Request:\n{prompt}"
            contents = [{"role": "user", "parts": [{"text": full_prompt}]}]

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_tokens,
            },
        }

        headers = {
            "Content-Type": "application/json",
        }

        response = requests.post(
            f"{self.base_url}?key={self.api_key}",
            headers=headers,
            json=payload,
            timeout=self.config.timeout,
        )
        response.raise_for_status()

        data = response.json()

        try:
            candidates = data.get("candidates", [])
            if not candidates:
                raise ValueError(f"No candidates returned by Gemini: {json.dumps(data, indent=2)}")

            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            texts = [part.get("text", "") for part in parts if "text" in part]
            return "\n".join(texts).strip()
        except Exception as exc:
            raise ValueError(f"Failed to parse Gemini response: {exc}\nRaw response: {json.dumps(data, indent=2)}")


# ============================================================
# Factory
# ============================================================

def get_llm_client(config: LLMConfig) -> BaseLLMClient:
    provider = config.provider.lower().strip()

    if provider == "groq":
        return GroqLLMClient(config)
    elif provider in ["huggingface", "hf"]:
        return HuggingFaceLLMClient(config)
    elif provider in ["google", "gemini"]:
        return GoogleGeminiLLMClient(config)
    else:
        raise ValueError(
            f"Unsupported LLM provider: {config.provider}. "
            f"Supported providers: groq, huggingface, google"
        )


# ============================================================
# Helper for project settings integration
# ============================================================

def load_llm_config_from_settings(settings: Any) -> LLMConfig:
    """
    Expects something like this in config/settings.py:

    LLM_PROVIDER = "groq"
    LLM_MODEL = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE = 0.2
    LLM_MAX_TOKENS = 1200
    LLM_TIMEOUT = 120

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    """

    provider = getattr(settings, "LLM_PROVIDER", "groq")
    model = getattr(settings, "LLM_MODEL", "")

    if not model:
        raise ValueError("LLM_MODEL is missing in settings.")

    provider_lower = provider.lower().strip()

    api_key = None
    base_url = getattr(settings, "LLM_BASE_URL", None)
    api_version = getattr(settings, "LLM_API_VERSION", None)

    if provider_lower == "groq":
        api_key = getattr(settings, "GROQ_API_KEY", None) or os.getenv("GROQ_API_KEY")
    elif provider_lower in ["huggingface", "hf"]:
        api_key = getattr(settings, "HUGGINGFACE_API_KEY", None) or os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
    elif provider_lower in ["google", "gemini"]:
        api_key = getattr(settings, "GOOGLE_API_KEY", None) or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    return LLMConfig(
        provider=provider,
        model=model,
        temperature=float(getattr(settings, "LLM_TEMPERATURE", 0.2)),
        max_tokens=int(getattr(settings, "LLM_MAX_TOKENS", 1200)),
        timeout=int(getattr(settings, "LLM_TIMEOUT", 120)),
        api_key=api_key,
        base_url=base_url,
        api_version=api_version,
    )