import json
import os

from transformers import AutoConfig, AutoTokenizer


def get_model_type(model_path: str) -> str | None:
        """Best-effort model_type lookup from a local HF snapshot/config dir."""
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            return None
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f).get("model_type")
        except (OSError, json.JSONDecodeError):
            return None


def needs_remote_code(model_path: str) -> bool:
        return get_model_type(model_path) == "kimi_linear"


def load_model_config(model_path: str):
        return AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=needs_remote_code(model_path),
        )


def load_tokenizer(model_path: str):
        kwargs = {"trust_remote_code": needs_remote_code(model_path)}
        if not needs_remote_code(model_path):
            kwargs["use_fast"] = True
        return AutoTokenizer.from_pretrained(model_path, **kwargs)


def infer_model_family(model_path: str) -> str:
        """Infer the high-level model family from config or local path."""
        model_type = get_model_type(model_path)
        if model_type == "llama":
            return "llama"
        if model_type == "qwen3":
            return "qwen"
        if model_type == "kimi_linear":
            return "kimi"

        model_path_lower = model_path.lower()
        if "llama" in model_path_lower:
            return "llama"
        elif "qwen" in model_path_lower:
            return "qwen"
        elif "kimi" in model_path_lower:
            return "kimi"
        else:
            return "unknown"


def decode_tokens(token_ids: list[int], tokenizer: AutoTokenizer) -> list[str]:
    decoded = []
    for token in token_ids:
        try:
            text = tokenizer.decode([token], skip_special_tokens=False)
            decoded.append(text)
        except Exception:
            decoded.append(f"<token_id:{token}>")
    return decoded
