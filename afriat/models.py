"""Model loading, decoder-layer hooks, counterbalanced A/B log-odds, and residual steering."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Sequence
import numpy as np

MODEL_PRESETS: dict[str, str] = {
    # Fast high-throughput 8B-14B scouts
    "gemma4-12b": "google/gemma-4-12B-it",
    "gemma-4-12b-it": "google/gemma-4-12B-it",
    "phi-4-14b": "microsoft/phi-4",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-14b": "Qwen/Qwen3-14B",
    # Confirmatory 24B-32B dense flagships
    "gemma4-31b": "google/gemma-4-31B-it",
    "gemma-4-31b-it": "google/gemma-4-31B-it",
    "mistral-small-24b-it": "mistralai/Mistral-Small-24B-Instruct-2501",
    "qwen3.8-27b": "Qwen/Qwen3.8-27B",
    "qwen3-32b": "Qwen/Qwen3-32B",
}


def resolve_model_name(name_or_alias: str) -> str:
    return MODEL_PRESETS.get(name_or_alias.lower(), name_or_alias)


def resolve_decoder_layers(model: Any) -> Sequence[Any]:
    """Locate the text decoder block list across standard and multimodal-unified wrappers."""
    cfg = getattr(model, "config", None)
    text_cfg = getattr(cfg, "text_config", None) or cfg
    expected_n = getattr(text_cfg, "num_hidden_layers", None)

    candidate_paths = (
        ("model", "language_model", "layers"),
        ("model", "language_model", "model", "layers"),
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("layers",),
    )
    for path in candidate_paths:
        cur = model
        ok = True
        for attr in path:
            if hasattr(cur, attr):
                cur = getattr(cur, attr)
            else:
                ok = False
                break
        if ok and hasattr(cur, "__len__") and len(cur) >= 2:
            if expected_n is None or len(cur) == int(expected_n):
                return cur

    if hasattr(model, "named_modules"):
        for name, mod in model.named_modules():
            if (
                name.endswith("layers")
                and "vision" not in name
                and "audio" not in name
                and hasattr(mod, "__len__")
                and len(mod) >= 2
                and (expected_n is None or len(mod) == int(expected_n))
            ):
                return mod

    raise AttributeError(
        f"Could not locate decoder block list on model class {type(model).__name__}. "
        "Inspect model.named_modules() and add its path to resolve_decoder_layers()."
    )


def _single_token_candidates(tokenizer: Any, letter: str) -> list[int]:
    """Collect distinct single-token encodings for e.g. 'A' and ' A' without BOS."""
    ids: list[int] = []
    for surface in (letter, f" {letter}"):
        toks = tokenizer.encode(surface, add_special_tokens=False)
        if len(toks) == 1 and toks[0] not in ids:
            ids.append(int(toks[0]))
    if not ids:
        toks = tokenizer.encode(letter, add_special_tokens=False)
        if not toks:
            raise ValueError(f"Tokenizer returned empty encoding for {letter!r}")
        ids.append(int(toks[-1]))
    return ids


def format_chat_prompt(tokenizer: Any, user_content: str, assistant_prefill: str = "") -> str:
    """Format single-turn prompt with thinking disabled when supported."""
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": user_content}]
        rendered: str | None = None
        for extra_kwargs in (
            {"enable_thinking": False},
            {"thinking": False},
            {},
        ):
            try:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **extra_kwargs,
                )
                break
            except TypeError:
                continue
        if rendered is not None:
            return rendered + assistant_prefill
    return f"User: {user_content}\nAssistant: {assistant_prefill}"


def build_choice_user_prompt(scenario: str, option_a: str, option_b: str) -> str:
    return (
        f"{scenario}\n\n"
        f"Options:\n"
        f"A) {option_a}\n"
        f"B) {option_b}\n\n"
        f"Reply with only the single letter A or B."
    )


def build_scenario_reading_prompt(scenario: str) -> str:
    return (
        "Read the following situation carefully and consider your position in it:\n\n"
        f"{scenario}"
    )


@dataclass
class ChoiceScore:
    target_log_odds: float   # 0.5 * [(logP(A|A=target) - logP(B|A=target)) + (logP(B|B=target) - logP(A|B=target))]
    target_prob: float       # sigmoid(target_log_odds)
    order_ab_log_odds: float
    order_ba_log_odds: float
    ab_mass: float           # mean probability mass falling on {A, B} tokens across both orderings


class TransformerHarness:
    """Thin, transparent wrapper around a HuggingFace causal LM for residual probing & steering."""

    def __init__(self, model: Any, tokenizer: Any, model_id: str, device: str | None = None):
        import torch

        self.torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.layers = resolve_decoder_layers(model)
        self.n_layers = len(self.layers)

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "left"

        if device is not None:
            self.input_device = torch.device(device)
        else:
            first_param = next(model.parameters())
            self.input_device = first_param.device

        self.a_token_ids = _single_token_candidates(tokenizer, "A")
        self.b_token_ids = _single_token_candidates(tokenizer, "B")

    def _tokenize_rendered_batch(self, prompts: list[str]) -> dict[str, Any]:
        """Tokenize already chat-templated prompts without adding duplicate <bos>."""
        has_chat_tmpl = bool(getattr(self.tokenizer, "chat_template", None))
        enc = self.tokenizer(
            prompts,
            add_special_tokens=not has_chat_tmpl,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        keep = ("input_ids", "attention_mask")
        return {k: v.to(self.input_device) for k, v in enc.items() if k in keep}

    def _forward_last_logits(self, enc: dict[str, Any]) -> Any:
        """Run forward pass computing only final-token logits when supported."""
        try:
            return self.model(**enc, use_cache=False, logits_to_keep=1)
        except TypeError:
            try:
                return self.model(**enc, use_cache=False)
            except TypeError:
                return self.model(**enc)

    @classmethod
    def from_pretrained(
        cls,
        name_or_alias: str,
        dtype: str = "bfloat16",
        device_map: str = "cuda",
        trust_remote_code: bool = False,
        model_id_label: str | None = None,
    ) -> "TransformerHarness":
        import torch
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        model_id = resolve_model_name(name_or_alias)
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "auto": "auto",
        }[dtype.lower()]

        if device_map == "cuda" and not torch.cuda.is_available():
            device_map = "auto"

        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)

        loader_classes: list[Any] = []
        for arch in (getattr(cfg, "architectures", None) or []):
            arch_cls = getattr(transformers, arch, None)
            if arch_cls is not None and arch_cls not in loader_classes:
                loader_classes.append(arch_cls)
        if AutoModelForCausalLM not in loader_classes:
            loader_classes.append(AutoModelForCausalLM)
        for fallback_name in ("AutoModelForImageTextToText", "AutoModelForMultimodalLM"):
            if hasattr(transformers, fallback_name):
                fb = getattr(transformers, fallback_name)
                if fb not in loader_classes:
                    loader_classes.append(fb)

        model = None
        load_errors: list[str] = []
        for loader_cls in loader_classes:
            for dtype_kw in ({"dtype": torch_dtype}, {"torch_dtype": torch_dtype}):
                try:
                    model = loader_cls.from_pretrained(
                        model_id,
                        device_map=device_map,
                        trust_remote_code=trust_remote_code,
                        **dtype_kw,
                    )
                    break
                except TypeError:
                    continue
                except Exception as exc:  # pragma: no cover - hardware/hub path
                    load_errors.append(f"{loader_cls.__name__}: {exc}")
                    break
            if model is not None:
                break

        if model is None:
            raise RuntimeError(f"Failed to load {model_id}: {'; '.join(load_errors)}")

        model.eval()
        torch.set_grad_enabled(False)
        return cls(model=model, tokenizer=tokenizer, model_id=model_id_label or model_id)

    @contextmanager
    def residual_steering(
        self,
        layer_idx: int,
        direction: np.ndarray,
        delta_scale: float,
        last_k_tokens: int | None = None,
    ) -> Iterator[None]:
        """Add delta_scale * direction to the residual output of decoder block `layer_idx`."""
        torch = self.torch
        vec_np = np.asarray(direction, dtype=np.float32) * float(delta_scale)
        cached_tensors: dict[tuple[Any, Any], Any] = {}

        def _hook(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            key = (hidden.device, hidden.dtype)
            if key not in cached_tensors:
                cached_tensors[key] = torch.tensor(vec_np, device=hidden.device, dtype=hidden.dtype)
            shift = cached_tensors[key]
            modified = hidden.clone()
            if last_k_tokens is None:
                modified = modified + shift
            else:
                modified[..., -last_k_tokens:, :] = modified[..., -last_k_tokens:, :] + shift
            if isinstance(output, tuple):
                return (modified,) + tuple(output[1:])
            return modified

        handle = self.layers[layer_idx].register_forward_hook(_hook)
        try:
            yield
        finally:
            handle.remove()

    @staticmethod
    def build_subspace_clamp_matrices(
        directions: Sequence[np.ndarray] | np.ndarray,
        neutral_mean: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute `(q_f32 [D, K], mu_proj_f32 [D])` for mean-centered QR subspace clamping."""
        arr = np.asarray(directions, dtype=np.float64)
        if arr.ndim == 1:
            cols = arr[:, None]
        else:
            cols = arr.T if arr.shape[0] < arr.shape[1] else arr
        q_np, _ = np.linalg.qr(cols, mode="reduced")
        q_f32_np = q_np.astype(np.float32)
        if neutral_mean is not None:
            mu_np = np.asarray(neutral_mean, dtype=np.float64).reshape(-1)
            mu_proj_np = ((mu_np @ q_np) @ q_np.T).astype(np.float32)
        else:
            mu_proj_np = np.zeros(q_np.shape[0], dtype=np.float32)
        return q_f32_np, mu_proj_np

    @contextmanager
    def residual_subspace_clamp(
        self,
        layer_idx: int,
        directions: Sequence[np.ndarray] | np.ndarray,
        neutral_mean: np.ndarray | None = None,
    ) -> Iterator[None]:
        """Project out span(directions) around `neutral_mean` at `layer_idx`:
        h_clamped = h - ((h - mu) @ Q) @ Q.T
        """
        torch = self.torch
        q_f32_np, mu_proj_np = self.build_subspace_clamp_matrices(directions, neutral_mean)

        cached: dict[Any, tuple[Any, Any]] = {}

        def _hook(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            dev = hidden.device
            if dev not in cached:
                cached[dev] = (
                    torch.tensor(q_f32_np, device=dev, dtype=torch.float32),
                    torch.tensor(mu_proj_np, device=dev, dtype=torch.float32),
                )
            q_t, mu_p_t = cached[dev]
            h32 = hidden.to(torch.float32)
            proj = torch.matmul(torch.matmul(h32, q_t), q_t.transpose(0, 1)) - mu_p_t
            modified = (h32 - proj).to(hidden.dtype)
            if isinstance(output, tuple):
                return (modified,) + tuple(output[1:])
            return modified

        handle = self.layers[layer_idx].register_forward_hook(_hook)
        try:
            yield
        finally:
            handle.remove()

    def extract_last_token_states(
        self,
        prompts: Sequence[str],
        batch_size: int = 8,
    ) -> np.ndarray:
        """Return residual stream activations [N, L, D] at the final token of each prompt."""
        torch = self.torch
        all_batches: list[np.ndarray] = []

        for start in range(0, len(prompts), batch_size):
            chunk = list(prompts[start : start + batch_size])
            enc = self._tokenize_rendered_batch(chunk)

            captured: dict[int, Any] = {}
            handles = []

            def _make_capture(idx: int):
                def _cap(_mod: Any, _inp: Any, out: Any) -> None:
                    h = out[0] if isinstance(out, tuple) else out
                    captured[idx] = h[:, -1, :].detach().to(torch.float32).cpu()

                return _cap

            for ell in range(self.n_layers):
                handles.append(self.layers[ell].register_forward_hook(_make_capture(ell)))

            try:
                with torch.no_grad():
                    self._forward_last_logits(enc)
            finally:
                for h in handles:
                    h.remove()

            stacked = torch.stack([captured[ell] for ell in range(self.n_layers)], dim=1).numpy()
            all_batches.append(stacked)

        return np.concatenate(all_batches, axis=0)

    def score_counterbalanced_choices(
        self,
        scenarios: Sequence[str],
        target_options: Sequence[str],
        reference_options: Sequence[str],
        batch_size: int = 8,
        capture_decision_states: bool = False,
    ) -> tuple[list[ChoiceScore], np.ndarray | None]:
        """Score binary choices across both (A=target, B=ref) and (A=ref, B=target) orderings."""
        torch = self.torch
        n = len(scenarios)
        if not (len(target_options) == n and len(reference_options) == n):
            raise ValueError("Mismatched lengths in score_counterbalanced_choices")

        prompts: list[str] = []
        for scen, tgt, ref in zip(scenarios, target_options, reference_options):
            p_ab = format_chat_prompt(self.tokenizer, build_choice_user_prompt(scen, tgt, ref))
            p_ba = format_chat_prompt(self.tokenizer, build_choice_user_prompt(scen, ref, tgt))
            prompts.extend([p_ab, p_ba])

        log_p_a_all: list[float] = []
        log_p_b_all: list[float] = []
        mass_all: list[float] = []
        state_chunks: list[np.ndarray] = []

        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start : start + batch_size]
            enc = self._tokenize_rendered_batch(chunk)

            captured: dict[int, Any] = {}
            handles = []
            if capture_decision_states:
                def _make_capture(idx: int):
                    def _cap(_mod: Any, _inp: Any, out: Any) -> None:
                        h = out[0] if isinstance(out, tuple) else out
                        captured[idx] = h[:, -1, :].detach().to(torch.float32).cpu()

                    return _cap

                for ell in range(self.n_layers):
                    handles.append(self.layers[ell].register_forward_hook(_make_capture(ell)))

            try:
                with torch.no_grad():
                    out = self._forward_last_logits(enc)
            finally:
                for h in handles:
                    h.remove()

            logits = out.logits[:, -1, :].to(torch.float32)
            log_probs = torch.log_softmax(logits, dim=-1)

            lpa = torch.logsumexp(log_probs[:, self.a_token_ids], dim=-1).cpu().numpy()
            lpb = torch.logsumexp(log_probs[:, self.b_token_ids], dim=-1).cpu().numpy()
            ab_mass = (np.exp(lpa) + np.exp(lpb)).clip(0.0, 1.0)

            log_p_a_all.extend(float(x) for x in lpa)
            log_p_b_all.extend(float(x) for x in lpb)
            mass_all.extend(float(x) for x in ab_mass)

            if capture_decision_states:
                stacked = torch.stack([captured[ell] for ell in range(self.n_layers)], dim=1).numpy()
                state_chunks.append(stacked)

        scores: list[ChoiceScore] = []
        for i in range(n):
            lo_ab = log_p_a_all[2 * i] - log_p_b_all[2 * i]          # A=target, B=ref
            lo_ba = log_p_b_all[2 * i + 1] - log_p_a_all[2 * i + 1]  # B=target, A=ref
            mean_lo = 0.5 * (lo_ab + lo_ba)
            prob = float(1.0 / (1.0 + np.exp(-np.clip(mean_lo, -40.0, 40.0))))
            mean_mass = 0.5 * (mass_all[2 * i] + mass_all[2 * i + 1])
            scores.append(
                ChoiceScore(
                    target_log_odds=float(mean_lo),
                    target_prob=prob,
                    order_ab_log_odds=float(lo_ab),
                    order_ba_log_odds=float(lo_ba),
                    ab_mass=float(mean_mass),
                )
            )

        decision_states: np.ndarray | None = None
        if capture_decision_states:
            all_2n = np.concatenate(state_chunks, axis=0)  # [2N, L, D]
            decision_states = 0.5 * (all_2n[0::2] + all_2n[1::2])

        return scores, decision_states

    def _resolve_stop_token_ids(self) -> list[int]:
        stop_ids: list[int] = []

        def _add(val: Any) -> None:
            if isinstance(val, int) and val >= 0 and val not in stop_ids:
                stop_ids.append(val)
            elif isinstance(val, (list, tuple, set)):
                for item in val:
                    _add(item)

        _add(getattr(self.tokenizer, "eos_token_id", None))
        gen_cfg = getattr(self.model, "generation_config", None)
        _add(getattr(gen_cfg, "eos_token_id", None))

        unk_id = getattr(self.tokenizer, "unk_token_id", None)
        if hasattr(self.tokenizer, "convert_tokens_to_ids"):
            for tok in ("<end_of_turn>", "<|im_end|>", "<|end|>", "<|eot_id|>"):
                try:
                    tid = self.tokenizer.convert_tokens_to_ids(tok)
                except Exception:
                    continue
                if isinstance(tid, int) and tid >= 0 and tid != unk_id:
                    _add(tid)
        return stop_ids

    def generate_texts(
        self,
        prompts: Sequence[str],
        max_new_tokens: int = 84,
        batch_size: int = 8,
    ) -> list[str]:
        """Greedy-decode continuations for already-formatted prompts (compatible with residual_steering)."""
        torch = self.torch
        stop_ids = self._resolve_stop_token_ids()
        stop_set = set(stop_ids)
        pad_id = int(self.tokenizer.pad_token_id or 0)
        outputs: list[str] = []

        for start in range(0, len(prompts), batch_size):
            chunk = list(prompts[start : start + batch_size])
            enc = self._tokenize_rendered_batch(chunk)
            prompt_len = int(enc["input_ids"].shape[1])

            gen_tokens = None
            if hasattr(self.model, "generate"):
                gen_kwargs: dict[str, Any] = {
                    "max_new_tokens": int(max_new_tokens),
                    "do_sample": False,
                    "pad_token_id": pad_id,
                }
                if stop_ids:
                    gen_kwargs["eos_token_id"] = stop_ids if len(stop_ids) > 1 else stop_ids[0]
                try:
                    with torch.no_grad():
                        seq = self.model.generate(**enc, **gen_kwargs)
                    gen_tokens = seq[:, prompt_len:].detach().cpu().tolist()
                except Exception:
                    gen_tokens = None

            if gen_tokens is None:
                # Fallback token-by-token greedy decode
                cur_ids = enc["input_ids"]
                cur_mask = enc["attention_mask"]
                bsz = int(cur_ids.shape[0])
                done = [False] * bsz
                step_cols: list[list[int]] = [[] for _ in range(bsz)]
                for _ in range(int(max_new_tokens)):
                    with torch.no_grad():
                        out = self._forward_last_logits({"input_ids": cur_ids, "attention_mask": cur_mask})
                    next_ids = torch.argmax(out.logits[:, -1, :], dim=-1)
                    next_list = next_ids.detach().cpu().tolist()
                    for bi, tid in enumerate(next_list):
                        if not done[bi]:
                            if int(tid) in stop_set:
                                done[bi] = True
                            else:
                                step_cols[bi].append(int(tid))
                    if all(done):
                        break
                    cur_ids = torch.cat([cur_ids, next_ids.unsqueeze(-1)], dim=1)
                    cur_mask = torch.cat([cur_mask, torch.ones((bsz, 1), dtype=cur_mask.dtype, device=cur_mask.device)], dim=1)
                gen_tokens = step_cols

            for row in gen_tokens:
                trimmed: list[int] = []
                for tid in row:
                    if int(tid) in stop_set:
                        break
                    trimmed.append(int(tid))
                txt = self.tokenizer.decode(trimmed, skip_special_tokens=True)
                outputs.append(txt.strip())

        return outputs

