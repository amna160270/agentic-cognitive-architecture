"""core/local_llm.py
Thin wrapper around a locally running Ollama model (Task 1).
"""
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import ollama

# Model is told to stop generating when it writes these strings.
DEFAULT_STOP = ["Observation:", "\n\nTask Complete:"]


@dataclass
class LLMResult:
    """What one model call returns (we need the stats for the benchmark)."""
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_sec: float
    tokens_per_sec: float


class LocalLLMClient:
    def __init__(
        self,
        model: str = "qwen2.5-coder:1.5b",
        temperature: float = 0.2,
        top_p: float = 0.9,
        num_ctx: int = 4096,
        stop: Optional[List[str]] = None,
        host: str = "http://localhost:11434",
    ):
        if not 0.0 <= temperature <= 1.0:
            raise ValueError("temperature must be in [0.0, 1.0]")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0.0, 1.0]")
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.num_ctx = num_ctx
        self.stop = stop if stop is not None else DEFAULT_STOP
        self.client = ollama.Client(host=host)

    def chat(self, messages: List[Dict[str, str]]) -> LLMResult:
        """Send a list of {role, content} messages, get the model's reply."""
        options = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_ctx": self.num_ctx,
            "stop": self.stop,
        }
        start = time.perf_counter()
        response = self.client.chat(
            model=self.model, messages=messages, options=options
        )
        latency = time.perf_counter() - start

        text = response["message"]["content"]
        prompt_tokens = response.get("prompt_eval_count", 0) or 0
        completion_tokens = response.get("eval_count", 0) or 0
        eval_ns = response.get("eval_duration", 0) or 0
        tps = completion_tokens / (eval_ns / 1e9) if eval_ns > 0 else (
            completion_tokens / latency if latency > 0 else 0.0
        )
        return LLMResult(text, prompt_tokens, completion_tokens, latency, tps)

    def generate(self, prompt: str, system: Optional[str] = None) -> LLMResult:
        """Convenience method for a single prompt."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages)


if __name__ == "__main__":
    llm = LocalLLMClient()
    r = llm.generate("Say hello in one line.")
    print("Reply:", r.text)
    print(f"Tokens: {r.completion_tokens} | {r.tokens_per_sec:.1f} tok/s | {r.latency_sec:.1f}s")