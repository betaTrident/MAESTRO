import os
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Optional

import ulid
from dotenv import load_dotenv

try:
    from langfuse import Langfuse, observe, propagate_attributes
except Exception:
    Langfuse = None

    def observe(*args, **kwargs):
        # Support both @observe and @observe(...)
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def _decorate(func):
            return func

        return _decorate

    def propagate_attributes(**kwargs):
        return nullcontext()


load_dotenv()

# Use a single model everywhere (default + trace paths).
MODEL_ID = "gpt-4o-mini"
TRACE_MODEL_ID = MODEL_ID
TRACE_MODEL_CANDIDATES = [MODEL_ID]
MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0"))
MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", "128"))
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_TIMEOUT_SECONDS = float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "12"))
STRICT_OUTPUT_SYSTEM_PROMPT = os.getenv(
    "STRICT_OUTPUT_SYSTEM_PROMPT",
    (
        "Return only the exact requested output: either one code letter or a"
        " binary mask containing only 0 and 1 characters. "
        "Never include words, punctuation, backticks, or prefixes like 0b."
    ),
).strip()


@dataclass
class _ModelResponse:
    content: str
    model: Optional[str] = None
    usage_details: Optional[Dict[str, int]] = None
    cost_details: Optional[Dict[str, float]] = None


_session_metrics: Dict[str, Dict[str, float]] = {}
_session_metrics_lock = threading.Lock()


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except Exception:
        return None


def _normalize_content(content):
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def _normalize_messages(messages):
    if isinstance(messages, str):
        text = messages.strip()
        normalized = [{"role": "user", "content": text}] if text else []
        return normalized, text

    if isinstance(messages, list):
        normalized = []
        for message in messages:
            if isinstance(message, dict):
                role = str(message.get("role", "user")).lower()
                content = message.get("content", "")
            else:
                role = str(getattr(message, "role", "user")).lower()
                content = getattr(message, "content", str(message))

            content_text = _normalize_content(content).strip()
            if not content_text:
                continue

            if role not in {"system", "user", "assistant", "tool"}:
                role = "user"
            normalized.append({"role": role, "content": content_text})

        preview = "\n".join(message["content"] for message in normalized).strip()
        return normalized, preview

    text = str(messages).strip()
    normalized = [{"role": "user", "content": text}] if text else []
    return normalized, text


def _usage_to_dict(usage_obj: Any) -> Dict[str, Any]:
    if usage_obj is None:
        return {}

    if isinstance(usage_obj, dict):
        raw = dict(usage_obj)
    elif hasattr(usage_obj, "model_dump"):
        raw = usage_obj.model_dump(exclude_none=True)
    elif hasattr(usage_obj, "__dict__"):
        raw = {k: v for k, v in usage_obj.__dict__.items() if v is not None}
    else:
        raw = {}

    usage = dict(raw)

    # Map provider-specific keys to Langfuse's generic usage buckets.
    if "input" not in usage:
        if "prompt_tokens" in usage and isinstance(usage["prompt_tokens"], int):
            usage["input"] = usage["prompt_tokens"]
        elif "input_tokens" in usage and isinstance(usage["input_tokens"], int):
            usage["input"] = usage["input_tokens"]

    if "output" not in usage:
        if "completion_tokens" in usage and isinstance(usage["completion_tokens"], int):
            usage["output"] = usage["completion_tokens"]
        elif "output_tokens" in usage and isinstance(usage["output_tokens"], int):
            usage["output"] = usage["output_tokens"]

    if "total" not in usage and "total_tokens" in usage and isinstance(usage["total_tokens"], int):
        usage["total"] = usage["total_tokens"]

    return usage


def _extract_cost_details(response_obj: Any, usage_details: Dict[str, Any]) -> Optional[Dict[str, float]]:
    cost_candidates = []

    usage_cost = _to_float(usage_details.get("cost"))
    if usage_cost is not None:
        cost_candidates.append(usage_cost)

    usage_raw = getattr(response_obj, "usage", None)
    if usage_raw is not None:
        if isinstance(usage_raw, dict):
            nested_cost = _to_float(usage_raw.get("cost"))
            if nested_cost is not None:
                cost_candidates.append(nested_cost)
        else:
            nested_cost = _to_float(getattr(usage_raw, "cost", None))
            if nested_cost is not None:
                cost_candidates.append(nested_cost)

    direct_cost = _to_float(getattr(response_obj, "cost", None))
    if direct_cost is not None:
        cost_candidates.append(direct_cost)

    if not cost_candidates:
        return None

    return {"total": max(cost_candidates)}


def _record_session_metrics(session_id: str, response: _ModelResponse, elapsed_seconds: float):
    if not session_id:
        return

    usage = response.usage_details or {}
    input_tokens = int(usage.get("input", 0) or 0)
    output_tokens = int(usage.get("output", 0) or 0)
    total_tokens = int(usage.get("total", input_tokens + output_tokens) or 0)

    call_cost = 0.0
    if response.cost_details:
        call_cost = float(response.cost_details.get("total", 0.0) or 0.0)

    with _session_metrics_lock:
        stats = _session_metrics.setdefault(
            session_id,
            {
                "latency": 0.0,
                "total_cost": 0.0,
                "llm_calls": 0.0,
                "input_tokens": 0.0,
                "output_tokens": 0.0,
                "total_tokens": 0.0,
            },
        )
        stats["latency"] += max(0.0, float(elapsed_seconds or 0.0))
        stats["total_cost"] += max(0.0, call_cost)
        stats["llm_calls"] += 1.0
        stats["input_tokens"] += float(input_tokens)
        stats["output_tokens"] += float(output_tokens)
        stats["total_tokens"] += float(total_tokens)


class OpenRouterChatModel:
    """Tiny bind/invoke-compatible wrapper for OpenRouter chat completions."""

    def __init__(self, client, model_id, temperature=0.0, max_tokens=128, extra_body=None):
        self.client = client
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = extra_body

    def bind(self, **kwargs):
        return OpenRouterChatModel(
            client=self.client,
            model_id=kwargs.get("model", self.model_id),
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            extra_body=kwargs.get("extra_body", self.extra_body),
        )

    def invoke(self, messages, config=None):
        normalized_messages, prompt_preview = _normalize_messages(messages)
        if not prompt_preview:
            return _ModelResponse(content="")

        payload = {
            "model": self.model_id,
            "messages": normalized_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.extra_body:
            payload["extra_body"] = self.extra_body

        response = self.client.chat.completions.create(**payload)
        content = response.choices[0].message.content
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )

        usage_details = _usage_to_dict(getattr(response, "usage", None))
        cost_details = _extract_cost_details(response, usage_details)

        return _ModelResponse(
            content=str(content or "").strip(),
            model=getattr(response, "model", self.model_id),
            usage_details=usage_details or None,
            cost_details=cost_details,
        )


def _build_provider_extra_body(order_env_name):
    provider_order_raw = os.getenv(order_env_name, "").strip()
    if not provider_order_raw:
        return None

    provider_order = [p.strip() for p in provider_order_raw.split(",") if p.strip()]
    if not provider_order:
        return None

    return {
        "provider": {
            "order": provider_order,
            "allow_fallbacks": False,
        }
    }


def _build_openrouter_client():
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return None

    from openai import OpenAI

    return OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        timeout=OPENROUTER_TIMEOUT_SECONDS,
    )


def _build_model(model_id=None, provider_order_env="OPENROUTER_PROVIDER_ORDER"):
    if openrouter_client is None:
        return None

    return OpenRouterChatModel(
        client=openrouter_client,
        model_id=model_id or MODEL_ID,
        temperature=MODEL_TEMPERATURE,
        max_tokens=MODEL_MAX_TOKENS,
        extra_body=_build_provider_extra_body(provider_order_env),
    )


def _build_langfuse_client():
    if Langfuse is None:
        return None

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    if not public_key or not secret_key:
        return None

    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse"),
    )


model = None
trace_model = None
openrouter_client = _build_openrouter_client()
langfuse_client = _build_langfuse_client()


def get_model():
    """Lazily initialize the default model only when needed."""
    global model
    if model is None:
        model = _build_model()
    return model


def get_trace_model():
    """Use the same single model for trace calls."""
    global trace_model
    if trace_model is None:
        trace_model = get_model()
    return trace_model


def generate_session_id():
    """Generate a unique session ID using TEAM_NAME and ULID."""
    return f"{os.getenv('TEAM_NAME', 'tutorial')}-{ulid.new().str}"


def invoke_model(llm_model, prompt):
    """Invoke a model once and return plain text response."""
    prompt_text = (prompt or "").strip()
    if not prompt_text:
        return _ModelResponse(content="")

    messages = [{"role": "user", "content": prompt_text}]
    if STRICT_OUTPUT_SYSTEM_PROMPT:
        messages.insert(0, {"role": "system", "content": STRICT_OUTPUT_SYSTEM_PROMPT})

    response = llm_model.invoke(
        messages,
        config={},
    )
    if isinstance(response, _ModelResponse):
        return response
    return _ModelResponse(content=str(getattr(response, "content", response)).strip())


@observe(as_type="generation")
def _run_llm_network_call(session_id, llm_model, prompt):
    """Run one traced network LLM call."""
    with propagate_attributes(session_id=session_id):
        start = time.perf_counter()
        response = invoke_model(llm_model, prompt)
        elapsed = time.perf_counter() - start
        _record_session_metrics(session_id, response, elapsed)

        if langfuse_client and response.usage_details:
            try:
                langfuse_client.update_current_generation(
                    model=response.model,
                    usage_details=response.usage_details,
                    cost_details=response.cost_details,
                )
            except Exception:
                pass
        return response.content


def run_llm_call(session_id, llm_model, prompt):
    """Run one model call with live network invocation."""
    if llm_model is None:
        llm_model = get_model()

    if not llm_model:
        raise RuntimeError("LLM model is not configured.")

    prompt = (prompt or "").strip()
    if not prompt:
        return ""

    response = _run_llm_network_call(
        session_id=session_id,
        llm_model=llm_model,
        prompt=prompt,
    )
    return response


def get_session_summary(session_id):
    """Return local per-session metrics aggregated during LLM invocations."""
    if not session_id:
        return None

    with _session_metrics_lock:
        stats = _session_metrics.get(session_id)
        if not stats:
            return None
        return {
            "session_id": session_id,
            "latency": float(stats.get("latency", 0.0)),
            "total_cost": float(stats.get("total_cost", 0.0)),
            "llm_calls": int(stats.get("llm_calls", 0.0)),
            "input_tokens": int(stats.get("input_tokens", 0.0)),
            "output_tokens": int(stats.get("output_tokens", 0.0)),
            "total_tokens": int(stats.get("total_tokens", 0.0)),
        }

def get_trace_info(session_id):
    """Retrieve trace information from Langfuse using the session ID."""
    if not langfuse_client:
        return None

    max_wait_seconds = int(os.getenv("TRACE_POLL_SECONDS", "10"))
    poll_interval_seconds = 2
    tries = max(1, max_wait_seconds // poll_interval_seconds)

    for _ in range(tries):
        try:
            traces = langfuse_client.api.trace.list(session_id=session_id)
            if traces and traces.data:
                return traces.data[0]

            recent = langfuse_client.api.trace.list(limit=20)
            if recent and recent.data:
                for trace in recent.data:
                    if getattr(trace, "session_id", None) == session_id:
                        return trace
        except Exception:
            pass
        time.sleep(poll_interval_seconds)
    return None


def print_results(info, session_id=None):
    """Print trace details from Langfuse or local session metrics."""
    if not info and session_id:
        info = get_session_summary(session_id)

    if not info:
        print("No trace found for this session ID.")
        return

    if isinstance(info, dict):
        print("\n--- Trace Info ---")
        print(f"Session ID: {info.get('session_id', session_id or 'N/A')}")
        print(f"Latency: {float(info.get('latency', 0.0)):.3f}s")
        print(f"Total Cost: ${float(info.get('total_cost', 0.0)):.6f}")
        if "llm_calls" in info:
            print(f"LLM Calls: {int(info.get('llm_calls', 0))}")
        if "total_tokens" in info:
            print(
                "Tokens: "
                f"in={int(info.get('input_tokens', 0))}, "
                f"out={int(info.get('output_tokens', 0))}, "
                f"total={int(info.get('total_tokens', 0))}"
            )
        return

    print("\n--- Trace Info ---")
    print(f"Trace ID: {info.id}")
    print(f"Session ID: {info.session_id}")
    print(f"Latency: {getattr(info, 'latency', 'N/A')}s")
    print(f"Total Cost: ${getattr(info, 'total_cost', 0.0):.6f}")
    base_url = os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse").rstrip("/")
    if hasattr(info, "html_path") and info.html_path:
        print(f"Link: {base_url}{info.html_path}")