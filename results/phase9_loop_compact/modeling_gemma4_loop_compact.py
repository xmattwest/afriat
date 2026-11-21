"""Standalone loader for `Gemma-4-Loop-Compact` (`Hourglass INT4/INT8 + Procrustes Recurrent Funnel + Halting`)."""

from pathlib import Path
import json
import numpy as np
import torch

from afriat.models import TransformerHarness
from afriat.loop_compact import (
    Gemma4LoopCompactController,
    build_step_bridge_and_halting_modules,
    compress_decoder_layers_inplace,
)


def load_gemma4_loop_compact(package_dir: str | Path, base_model_id: str | None = None, device_map: str = "cuda"):
    pkg = Path(package_dir)
    cfg = json.loads((pkg / "config_loop_compact.json").read_text(encoding="utf-8"))
    model_id = base_model_id or cfg["base_model_id"]
    harness = TransformerHarness.from_pretrained(model_id, dtype="bfloat16", device_map=device_map)

    imp_npz = np.load(pkg / "neuron_importance.npz")
    importance = {int(k.split("_")[1]): imp_npz[k] for k in imp_npz.files}
    plan = {int(k): v for k, v in cfg["compression_plan"].items()}
    loop_ctx = {"loop_idx": 0}

    compress_decoder_layers_inplace(
        harness,
        importance=importance,
        plan=plan,
        funnel_start=int(cfg["funnel_start"]),
        funnel_end=int(cfg["funnel_end"]),
        attach_loop_lora=True,
        max_loops=int(cfg["max_loops"]),
        lora_rank=int(cfg["lora_rank"]),
        lora_alpha=float(cfg["lora_alpha"]),
        loop_ctx=loop_ctx,
    )

    state_bundle = torch.load(pkg / "loop_compact_adapters.pt", map_location=harness.input_device, weights_only=True)
    b_sd = dict(state_bundle["bridge_state_dict"])
    if "r_ortho" not in b_sd and "r_ortho_base" in b_sd:
        b_sd["r_ortho"] = b_sd["r_ortho_base"].clone()
    p_fit = dict(state_bundle["procrustes_meta"])
    p_fit["r_0"] = b_sd["r_ortho_base"].cpu().numpy()

    bridge, halting = build_step_bridge_and_halting_modules(
        torch,
        procrustes_fit=p_fit,
        max_loops=int(cfg["max_loops"]),
        rank=int(cfg["bridge_rank"]),
        device=harness.input_device,
    )
    bridge.load_state_dict(b_sd)
    halting.load_state_dict(state_bundle["halting_state_dict"])

    for ell_str, sd in state_bundle["funnel_lora_state_dicts"].items():
        ell = int(ell_str)
        harness.layers[ell].load_state_dict(sd, strict=False)

    return Gemma4LoopCompactController(
        harness=harness,
        bridge=bridge,
        halting_head=halting,
        loop_ctx=loop_ctx,
        start_layer=int(cfg["funnel_start"]),
        end_layer=int(cfg["funnel_end"]),
        max_loops=int(cfg["max_loops"]),
    )
