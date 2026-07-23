"""Pinned tokenizer loaders for the validation experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Protocol

import tiktoken
from transformers import AutoTokenizer


class EncodeFn(Protocol):
    def __call__(self, text: str) -> List[int]: ...


class SurfaceFn(Protocol):
    def __call__(self, token_id: int) -> str: ...


@dataclass(frozen=True)
class TokenizerSpec:
    id: str
    name: str
    source: str
    is_frontier: bool
    encode: EncodeFn
    surface: SurfaceFn
    # Frontier BPE tokenizers are space-sensitive: leading space changes tokenization.
    leading_space_for_words: bool = True


# Pinned model / encoding identifiers
O200K_ENCODING = "o200k_base"
GLM_REPO = "zai-org/GLM-5.2"
LLAMA_REPO = "meta-llama/Meta-Llama-3.1-8B"
QWEN_REPO = "Qwen/Qwen2.5-7B"
NLLB_REPO = "facebook/nllb-200-distilled-600M"
MT5_REPO = "google/mt5-base"
MBERT_REPO = "bert-base-multilingual-cased"
SUPERBPE_REPO = "UW/OLMo2-8B-SuperBPE-t180k"


def _tiktoken_surface(enc: tiktoken.Encoding) -> SurfaceFn:
    def surface(token_id: int) -> str:
        raw = enc.decode_single_token_bytes(token_id)
        return raw.decode("utf-8", errors="replace")

    return surface


def _hf_surface(tok) -> SurfaceFn:
    def surface(token_id: int) -> str:
        # Decode a single id in isolation for surface-length checks (STFR).
        return tok.decode([token_id], skip_special_tokens=True)

    return surface


def _hf_spec(
    tid: str,
    name: str,
    repo_id: str,
    is_frontier: bool,
    trust_remote_code: bool = False,
    leading_space_for_words: bool = True,
) -> TokenizerSpec:
    tok = AutoTokenizer.from_pretrained(
        repo_id,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )

    def encode(text: str) -> List[int]:
        return tok.encode(text, add_special_tokens=False)

    return TokenizerSpec(
        id=tid,
        name=name,
        source=repo_id,
        is_frontier=is_frontier,
        encode=encode,
        surface=_hf_surface(tok),
        leading_space_for_words=leading_space_for_words,
    )


def _tiktoken_spec(tid: str, name: str, encoding_name: str, is_frontier: bool) -> TokenizerSpec:
    enc = tiktoken.get_encoding(encoding_name)
    return TokenizerSpec(
        id=tid,
        name=name,
        source=encoding_name,
        is_frontier=is_frontier,
        encode=lambda text: enc.encode(text),
        surface=_tiktoken_surface(enc),
        leading_space_for_words=True,
    )


def _o200k_grapheme_spec() -> TokenizerSpec:
    from .grapheme_wrap import make_o200k_grapheme_encode

    enc = tiktoken.get_encoding(O200K_ENCODING)
    return TokenizerSpec(
        id="o200k_grapheme",
        name="o200k + grapheme-healed pretok",
        source=f"{O200K_ENCODING}+UAX29_heal",
        is_frontier=False,
        encode=make_o200k_grapheme_encode(),
        surface=_tiktoken_surface(enc),
        leading_space_for_words=True,
    )


def load_tokenizers(
    include: List[str] | None = None,
) -> Dict[str, TokenizerSpec]:
    """Load all (or selected) tokenizers. Keys are short ids from the plan."""
    builders: Dict[str, Callable[[], TokenizerSpec]] = {
        "o200k": lambda: _tiktoken_spec(
            "o200k", "OpenAI o200k_base", O200K_ENCODING, True
        ),
        "o200k_grapheme": _o200k_grapheme_spec,
        "glm": lambda: _hf_spec(
            "glm", "GLM-5.2", GLM_REPO, True, trust_remote_code=True
        ),
        "llama": lambda: _hf_spec("llama", "Llama-3.1-8B", LLAMA_REPO, True),
        "qwen": lambda: _hf_spec("qwen", "Qwen2.5-7B", QWEN_REPO, True),
        "multi": lambda: _hf_spec(
            "multi",
            "NLLB-200",
            NLLB_REPO,
            False,
            leading_space_for_words=False,
        ),
        "unigram": lambda: _hf_spec(
            "unigram",
            "mT5 (Unigram)",
            MT5_REPO,
            False,
            leading_space_for_words=False,
        ),
        "wordpiece": lambda: _hf_spec(
            "wordpiece",
            "mBERT (WordPiece)",
            MBERT_REPO,
            False,
            leading_space_for_words=False,
        ),
        "superbpe": lambda: _hf_spec(
            "superbpe",
            "SuperBPE t180k",
            SUPERBPE_REPO,
            False,
        ),
    }

    wanted = include or list(builders.keys())
    loaded: Dict[str, TokenizerSpec] = {}
    errors: Dict[str, str] = {}

    for tid in wanted:
        if tid not in builders:
            raise KeyError(f"Unknown tokenizer id: {tid}")
        try:
            loaded[tid] = builders[tid]()
            print(f"[ok] loaded tokenizer: {tid} ({loaded[tid].source})")
        except Exception as exc:  # noqa: BLE001
            errors[tid] = str(exc)
            print(f"[fail] tokenizer {tid}: {exc}")

    if not loaded:
        raise RuntimeError(f"No tokenizers loaded. Errors: {errors}")
    if errors:
        print(f"[warn] some tokenizers failed: {errors}")
    return loaded
