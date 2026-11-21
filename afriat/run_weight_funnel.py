"""Local Real-Weight Read/Write Funnel Dissection on pinned `google/gemma-4-31B-it`.

Streams exact byte-range slices of MLP `gate_proj`, `up_proj`, `down_proj`,
`pre_feedforward_layernorm`, `post_feedforward_layernorm`, `layer_scalar`,
`norm.weight`, and lexical rows of `embed_tokens.weight` from the pinned commit
(`842da3794eaa0b77d5f08bae87a17459d91ff475` in `config/pins.json`), one layer at a
time (< 1.1 GB RAM), and evaluates against the locked `basis_vectors.npz`:

  1. Static `W_down` Writer Concentration (`L8, L16, L22, L26, L30, L33, L35, L56`):
     Tests whether `Gemma 4 31B` has dedicated heavy-tailed `W_down` writer neurons
     for `Spite`, `Contamination`, `Guilt`, and `Awe/Dread` vs. 500 random isotropic
     directions and Goodfire lexical token directions (`" an"`, `" unfair"`, `" disgusting"`).
  2. Direct `READ` (`W_gate`, `W_up`) vs. `WRITE` (`W_down`) Dimensionality & Fan-In:
     Measures Participation Ratio (`PR`, effective rank out of 64) and pairwise cosine
     coherence of input `READ` weights vs. output `WRITE` weights on the top-64 writers,
     testing Monolithic Community (Hyp B) vs. Many-to-One Funnel (Hyp C).
  3. Non-Affective-to-Moral Translation Ratio:
     Measures what fraction of top funnel writers' `READ` norm lies outside the incoming
     10-axis affective subspace (`span(all 10 concepts at layer l)`), testing whether
     `L26..L35` MLPs translate non-emotional features onto the `Layer 35` moral bus.
  4. Late-Layer (`L56`) Corporate Scrubbing / Affect-Token Suppression Check:
     Tests whether late-layer neurons reading `+Spite` / `+Contamination` actively write
     negative logits onto explicit affect words (`" unfair"`, `" greedy"`, `" disgusting"`)
     while boosting corporate register (`" policy"`, `" standard"`, `" market"`, `" rational"`).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import http.client
import json
from pathlib import Path
import struct
import threading
import time
from typing import Any
import urllib.parse
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from afriat.funnel import excess_kurtosis, gini_abs, jaccard_at_k
from afriat.inequity import LockedBasis
from afriat.schema import ALL_CONCEPTS, CORE_CONCEPTS, ROOT_DIR

IVORY = "#FFFFF0"
PARCHMENT = "#F5EEDC"
ONYX = "#0F0F0F"
SLATE_DEEP = "#1E3447"
BLUE_SLATE = "#2C3E50"
SLATE_MIST = "#808B90"
GOLDEN_ORANGE = "#F39C12"
COPPER_DEEP = "#96590A"
SAGE_DEEP = "#5F7F4A"
MINUS = "\u2212"

TRANSECT_LAYERS: tuple[int, ...] = (8, 16, 22, 26, 30, 33, 35, 56)

LEXICAL_PROBES: dict[str, str] = {
    "tok_an": " an",
    "tok_are": " are",
    "tok_unfair": " unfair",
    "tok_greedy": " greedy",
    "tok_disgusting": " disgusting",
    "tok_spite": " spite",
    "tok_rational": " rational",
    "tok_market": " market",
    "tok_standard": " standard",
    "tok_policy": " policy",
}


class PersistentRangeReader:
    """Thread-local persistent HTTPS range reader for HuggingFace CDN shards."""

    def __init__(self, repo: str, rev: str):
        self.repo = repo
        self.rev = rev
        self._tls = threading.local()
        self.shards: dict[str, dict[str, Any]] = {}

    def _resolve_cdn(self, hf_url: str) -> str:
        req = urllib.request.Request(hf_url, headers={"Range": "bytes=0-0", "User-Agent": "afriat-weight-probe/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.geturl() or hf_url

    def _get_conn(self, host: str) -> http.client.HTTPSConnection:
        conns = getattr(self._tls, "conns", None)
        if conns is None:
            conns = {}
            self._tls.conns = conns
        conn = conns.get(host)
        if conn is None:
            conn = http.client.HTTPSConnection(host, timeout=45)
            conns[host] = conn
        return conn

    def fetch_bytes(self, full_url: str, start: int, end: int) -> bytes:
        parsed = urllib.parse.urlsplit(full_url)
        host = parsed.netloc
        path_q = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        expected = end - start + 1
        for attempt in range(4):
            try:
                conn = self._get_conn(host)
                conn.request(
                    "GET",
                    path_q,
                    headers={
                        "Host": host,
                        "Range": f"bytes={start}-{end}",
                        "User-Agent": "afriat-weight-probe/1.0",
                        "Connection": "keep-alive",
                    },
                )
                resp = conn.getresponse()
                data = resp.read()
                if resp.status not in (200, 206) or len(data) != expected:
                    raise RuntimeError(f"HTTP {resp.status}, got {len(data)} != {expected}")
                return data
            except Exception:
                conns = getattr(self._tls, "conns", None) or {}
                old = conns.pop(host, None)
                if old is not None:
                    try:
                        old.close()
                    except Exception:
                        pass
                if attempt == 3:
                    # Fallback to urllib
                    req = urllib.request.Request(full_url, headers={"Range": f"bytes={start}-{end}"})
                    with urllib.request.urlopen(req, timeout=60) as r:
                        return r.read()
                time.sleep(0.4 * (attempt + 1))
        raise RuntimeError("unreachable")

    def init_shard(self, shard_name: str) -> None:
        if shard_name in self.shards:
            return
        hf_url = f"https://huggingface.co/{self.repo}/resolve/{self.rev}/{shard_name}"
        cdn_url = self._resolve_cdn(hf_url)
        raw8 = self.fetch_bytes(cdn_url, 0, 7)
        hlen = struct.unpack("<Q", raw8)[0]
        hj = json.loads(self.fetch_bytes(cdn_url, 8, 8 + hlen - 1).decode("utf-8"))
        self.shards[shard_name] = {
            "cdn_url": cdn_url,
            "base": 8 + hlen,
            "header": hj,
        }

    def fetch_tensor_f32(self, shard_name: str, key: str, max_workers: int = 16) -> np.ndarray:
        self.init_shard(shard_name)
        st = self.shards[shard_name]
        info = st["header"][key]
        s = st["base"] + int(info["data_offsets"][0])
        e = st["base"] + int(info["data_offsets"][1]) - 1
        chunk = 16 * 1024 * 1024
        ranges = [(pos, min(pos + chunk - 1, e)) for pos in range(s, e + 1, chunk)]
        if len(ranges) == 1:
            raw = self.fetch_bytes(st["cdn_url"], ranges[0][0], ranges[0][1])
        else:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(ranges))) as pool:
                parts = list(pool.map(lambda r: self.fetch_bytes(st["cdn_url"], r[0], r[1]), ranges))
            raw = b"".join(parts)
        dt = info["dtype"]
        if dt == "BF16":
            u16 = np.frombuffer(raw, dtype=np.uint16)
            arr = (u16.astype(np.uint32) << 16).view(np.float32)
        elif dt == "F16":
            arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        elif dt == "F32":
            arr = np.frombuffer(raw, dtype=np.float32).copy()
        else:
            raise ValueError(f"Unsupported dtype {dt}")
        return arr.reshape(info["shape"])

    def fetch_embedding_row_f32(self, shard_name: str, key: str, row_idx: int) -> np.ndarray:
        self.init_shard(shard_name)
        st = self.shards[shard_name]
        info = st["header"][key]
        vocab, d_model = info["shape"]
        assert 0 <= row_idx < vocab
        bytes_per_elem = 2 if info["dtype"] in ("BF16", "F16") else 4
        row_bytes = d_model * bytes_per_elem
        s = st["base"] + int(info["data_offsets"][0]) + row_idx * row_bytes
        e = s + row_bytes - 1
        raw = self.fetch_bytes(st["cdn_url"], s, e)
        if info["dtype"] == "BF16":
            u16 = np.frombuffer(raw, dtype=np.uint16)
            return (u16.astype(np.uint32) << 16).view(np.float32).copy()
        return np.frombuffer(raw, dtype=np.float32).copy()


def participation_ratio_matrix(mat: np.ndarray) -> float:
    """Effective rank `PR = (sum s_i^2)^2 / sum s_i^4` in `[1, min(K, D)]` via Gram matrix `M @ M.T`."""
    g = np.asarray(mat, dtype=np.float64) @ np.asarray(mat, dtype=np.float64).T  # [K, K]
    eigvals = np.maximum(np.linalg.eigvalsh(g), 0.0)
    s1 = float(np.sum(eigvals))
    s2 = float(np.sum(eigvals * eigvals))
    if s2 <= 1e-24:
        return 0.0
    return float((s1 * s1) / s2)


def mean_offdiag_abs_cos(unit_rows: np.ndarray) -> float:
    sim = np.abs(unit_rows @ unit_rows.T)
    k = sim.shape[0]
    return float((np.sum(sim) - np.trace(sim)) / max(k * (k - 1), 1))


def mean_offdiag_signed_cos(unit_rows: np.ndarray) -> float:
    sim = unit_rows @ unit_rows.T
    k = sim.shape[0]
    return float((np.sum(sim) - np.trace(sim)) / max(k * (k - 1), 1))


def resolve_single_token_ids(repo: str, rev: str, cache_dir: Path) -> dict[str, int]:
    """Download `tokenizer.json` once and look up exact single-token IDs for `LEXICAL_PROBES`."""
    tok_path = cache_dir / "tokenizer.json"
    if not tok_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{repo}/resolve/{rev}/tokenizer.json"
        req = urllib.request.Request(url, headers={"User-Agent": "afriat-weight-probe/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            tok_path.write_bytes(r.read())
    tok_data = json.loads(tok_path.read_text(encoding="utf-8"))
    vocab: dict[str, int] = tok_data["model"]["vocab"]
    out: dict[str, int] = {}
    for label, surface in LEXICAL_PROBES.items():
        # SentencePiece / Gemma uses `\u2581` for leading space
        sp_form = surface.replace(" ", "\u2581")
        for cand in (sp_form, surface.strip(), surface):
            if cand in vocab:
                out[label] = int(vocab[cand])
                break
        if label not in out:
            raise KeyError(f"Could not find token {surface!r} ({sp_form!r}) in tokenizer.json vocab")
    return out


def run_real_weight_funnel(out_dir: Path) -> dict[str, Any]:
    t0 = time.time()
    pins = json.loads((ROOT_DIR / "config" / "pins.json").read_text(encoding="utf-8"))
    m_entry = next(m for m in pins["models"] if m["name"] == "gemma-4-31b-it")
    repo, rev = m_entry["repo"], m_entry["revision"]

    p0_dir = ROOT_DIR / "runs" / "gemma-4-31b-it_20261114T152233Z"
    locked = LockedBasis.from_phase0_files(p0_dir / "basis_vectors.npz", p0_dir / "results.json")
    pl = locked.primary_layer  # 35

    idx_blob = Path.home() / ".cache/huggingface/hub/models--google--gemma-4-31B-it/blobs/17fcef4a0ceb639c8ad4e2f7719f70aaff7de63b"
    if idx_blob.exists():
        weight_map = json.loads(idx_blob.read_text(encoding="utf-8"))["weight_map"]
    else:
        url_idx = f"https://huggingface.co/{repo}/resolve/{rev}/model.safetensors.index.json"
        weight_map = json.loads(urllib.request.urlopen(url_idx, timeout=30).read().decode("utf-8"))["weight_map"]

    reader = PersistentRangeReader(repo, rev)
    tok_ids = resolve_single_token_ids(repo, rev, out_dir / "_cache")
    print(f"[WeightFunnel] Resolved lexical token IDs: {tok_ids}", flush=True)

    # Fetch final RMSNorm weight and exact unembedding rows for lexical tokens
    norm_key = "model.language_model.norm.weight"
    final_norm_w = reader.fetch_tensor_f32(weight_map[norm_key], norm_key)  # [5376]
    emb_key = "model.language_model.embed_tokens.weight"
    emb_shard = weight_map[emb_key]

    lex_dirs: dict[str, np.ndarray] = {}
    for lbl, tid in tok_ids.items():
        row = reader.fetch_embedding_row_f32(emb_shard, emb_key, tid)
        eff = final_norm_w * row  # exact pre-final-norm residual logit direction!
        lex_dirs[lbl] = (eff / max(float(np.linalg.norm(eff)), 1e-12)).astype(np.float32)

    # Read all 60 layer_scalars so we compute exact skip-connection gain Gamma_{l -> 35}!
    layer_scalars = np.ones(60, dtype=np.float64)
    for ell in range(60):
        k_ls = f"model.language_model.layers.{ell}.layer_scalar"
        layer_scalars[ell] = float(reader.fetch_tensor_f32(weight_map[k_ls], k_ls).reshape(-1)[0])

    gamma_to_35 = np.zeros(60, dtype=np.float64)
    for ell in range(pl + 1):
        prod_after = float(np.prod(layer_scalars[ell + 1 : pl + 1])) if ell < pl else 1.0
        gamma_to_35[ell] = layer_scalars[ell] * prod_after

    d_model = 5376
    rng = np.random.default_rng(20261118)
    rand_dirs = rng.normal(size=(500, d_model)).astype(np.float32)
    rand_dirs /= np.maximum(np.linalg.norm(rand_dirs, axis=-1, keepdims=True), 1e-12)

    focus_concepts = ("spite", "contamination", "guilt", "awe_dread")
    u35_map = {c: locked.controlled_dirs[c][pl].astype(np.float32) for c in CORE_CONCEPTS}

    per_layer_records: list[dict[str, Any]] = []
    scrubbing_l56: dict[str, Any] = {}

    for ell in TRANSECT_LAYERS:
        tl0 = time.time()
        prefix = f"model.language_model.layers.{ell}"
        g_pre = reader.fetch_tensor_f32(weight_map[f"{prefix}.pre_feedforward_layernorm.weight"], f"{prefix}.pre_feedforward_layernorm.weight")
        g_post = reader.fetch_tensor_f32(weight_map[f"{prefix}.post_feedforward_layernorm.weight"], f"{prefix}.post_feedforward_layernorm.weight")
        w_gate = reader.fetch_tensor_f32(weight_map[f"{prefix}.mlp.gate_proj.weight"], f"{prefix}.mlp.gate_proj.weight")  # [21504, 5376]
        w_up = reader.fetch_tensor_f32(weight_map[f"{prefix}.mlp.up_proj.weight"], f"{prefix}.mlp.up_proj.weight")      # [21504, 5376]
        w_down = reader.fetch_tensor_f32(weight_map[f"{prefix}.mlp.down_proj.weight"], f"{prefix}.mlp.down_proj.weight")  # [5376, 21504]

        d_ff = w_down.shape[1]
        # Effective post-norm write columns `[5376, 21504]`
        w_write_eff = g_post[:, None] * w_down
        write_col_norms = np.maximum(np.linalg.norm(w_write_eff, axis=0), 1e-12)
        w_write_unit = w_write_eff / write_col_norms[None, :]  # [5376, 21504]

        # Effective pre-norm read rows `[21504, 5376]`
        w_gate_eff = w_gate * g_pre[None, :]
        w_gate_unit = w_gate_eff / np.maximum(np.linalg.norm(w_gate_eff, axis=1, keepdims=True), 1e-12)
        w_up_eff = w_up * g_pre[None, :]
        w_up_unit = w_up_eff / np.maximum(np.linalg.norm(w_up_eff, axis=1, keepdims=True), 1e-12)

        # 1. Null distribution over 500 random isotropic directions in R^5376 (accounts for exact W_down anisotropy!)
        rand_cos = rand_dirs @ w_write_unit  # [500, 21504]
        null_kurt = np.array([excess_kurtosis(rand_cos[j]) for j in range(500)])
        null_top64_rms = np.sqrt(np.mean(np.sort(rand_cos ** 2, axis=1)[:, -64:], axis=1))
        null_max_abs = np.max(np.abs(rand_cos), axis=1)

        # Also random 64-neuron baseline for PR_gate, PR_up, PR_write
        rand_idx64 = rng.choice(d_ff, size=64, replace=False)
        rand_pr_gate = participation_ratio_matrix(w_gate_unit[rand_idx64])
        rand_pr_up = participation_ratio_matrix(w_up_unit[rand_idx64])
        rand_pr_write = participation_ratio_matrix(w_write_unit[:, rand_idx64].T)
        rand_gate_abs_cos = mean_offdiag_abs_cos(w_gate_unit[rand_idx64])
        rand_write_cos = mean_offdiag_signed_cos(w_write_unit[:, rand_idx64].T)

        # Incoming 10-concept orthonormal basis at layer `ell` (for non-affective-to-moral translation fraction!)
        in_mat = np.column_stack([locked.raw_dirs[c][ell].astype(np.float32) for c in ALL_CONCEPTS])  # [5376, 10]
        q_in10, _ = np.linalg.qr(in_mat, mode="reduced")  # [5376, 10]

        def _analyze_direction(u_vec: np.ndarray, k_top: int = 64) -> dict[str, Any]:
            cos_all = u_vec @ w_write_unit  # [21504]
            kurt = excess_kurtosis(cos_all)
            top64_rms = float(np.sqrt(np.mean(np.sort(cos_all ** 2)[-k_top:])))
            max_abs = float(np.max(np.abs(cos_all)))

            z_kurt = float((kurt - np.mean(null_kurt)) / max(float(np.std(null_kurt, ddof=1)), 1e-12))
            p_kurt = float((np.sum(null_kurt >= kurt) + 1) / 501.0)
            z_top64 = float((top64_rms - np.mean(null_top64_rms)) / max(float(np.std(null_top64_rms, ddof=1)), 1e-12))
            p_top64 = float((np.sum(null_top64_rms >= top64_rms) + 1) / 501.0)

            # Top-64 positive writer neurons onto `+u_vec`
            top_pos = np.argsort(cos_all)[-k_top:][::-1]
            g_sub = w_gate_unit[top_pos]        # [64, 5376]
            u_sub = w_up_unit[top_pos]          # [64, 5376]
            d_sub = w_write_unit[:, top_pos].T  # [64, 5376]

            pr_gate = participation_ratio_matrix(g_sub)
            pr_up = participation_ratio_matrix(u_sub)
            pr_write = participation_ratio_matrix(d_sub)

            gate_abs_cos = mean_offdiag_abs_cos(g_sub)
            up_abs_cos = mean_offdiag_abs_cos(u_sub)
            write_signed_cos = mean_offdiag_signed_cos(d_sub)
            write_abs_cos = mean_offdiag_abs_cos(d_sub)

            # Fraction of READ energy (`gate` and `up`) outside incoming 10-concept affective subspace!
            gate_in_10d = np.sum((g_sub @ q_in10) ** 2, axis=1)  # [64] in [0, 1]
            up_in_10d = np.sum((u_sub @ q_in10) ** 2, axis=1)
            non_affective_read_share = float(1.0 - 0.5 * (np.mean(gate_in_10d) + np.mean(up_in_10d)))

            return {
                "kurtosis": round(kurt, 4),
                "kurtosis_z_vs_500_random": round(z_kurt, 2),
                "kurtosis_p": round(p_kurt, 4),
                "top64_rms_cos": round(top64_rms, 5),
                "top64_rms_z_vs_500_random": round(z_top64, 2),
                "top64_rms_p": round(p_top64, 4),
                "max_abs_cos": round(max_abs, 5),
                "top64_mean_pos_cos": round(float(np.mean(cos_all[top_pos])), 5),
                "pr_read_gate_of_64": round(pr_gate, 2),
                "pr_read_up_of_64": round(pr_up, 2),
                "pr_write_down_of_64": round(pr_write, 2),
                "fan_in_pr_ratio_gate_over_write": round(pr_gate / max(pr_write, 1e-12), 3),
                "read_gate_offdiag_abs_cos": round(gate_abs_cos, 5),
                "read_up_offdiag_abs_cos": round(up_abs_cos, 5),
                "write_down_offdiag_signed_cos": round(write_signed_cos, 5),
                "write_down_offdiag_abs_cos": round(write_abs_cos, 5),
                "non_affective_complement_read_share": round(non_affective_read_share, 5),
                "top10_writer_neurons": [int(i) for i in top_pos[:10]],
                "_cos_all": cos_all,
            }

        moral_u35_res = {c: _analyze_direction(u35_map[c]) for c in focus_concepts}
        moral_uloc_res = {c: _analyze_direction(locked.controlled_dirs[c][ell].astype(np.float32)) for c in focus_concepts}
        lex_res = {lbl: _analyze_direction(lex_dirs[lbl]) for lbl in ("tok_an", "tok_unfair", "tok_disgusting", "tok_rational")}

        # Cross-axis writer overlap at layer `ell` (e.g. do Spite and Guilt / Contamination share the same top-64 writers?)
        cross_axis_jaccard64 = {
            "spite_vs_contamination": round(jaccard_at_k(moral_u35_res["spite"]["_cos_all"], moral_u35_res["contamination"]["_cos_all"], 64), 4),
            "spite_vs_guilt": round(jaccard_at_k(moral_u35_res["spite"]["_cos_all"], moral_u35_res["guilt"]["_cos_all"], 64), 4),
            "spite_vs_awe_dread": round(jaccard_at_k(moral_u35_res["spite"]["_cos_all"], moral_u35_res["awe_dread"]["_cos_all"], 64), 4),
            "spite_vs_tok_unfair": round(jaccard_at_k(moral_u35_res["spite"]["_cos_all"], lex_res["tok_unfair"]["_cos_all"], 64), 4),
        }

        # Special Late-Layer (`L56`) Corporate Scrubbing / Affect-Suppression Test!
        if ell == 56:
            # Neurons at L56 that READ `+u_spite[35]` (or `+u_spite[56]`) strongly via `gate` or `up`
            for cname, u_bus in (("spite_u35", u35_map["spite"]), ("spite_u56", locked.controlled_dirs["spite"][56].astype(np.float32)), ("contam_u35", u35_map["contamination"])):
                read_score = 0.5 * (np.abs(w_gate_unit @ u_bus) + np.abs(w_up_unit @ u_bus))
                top_readers = np.argsort(read_score)[-64:][::-1]
                # Check what these L56 bus-readers WRITE onto affect words vs corporate words!
                affect_keys = ("tok_unfair", "tok_greedy", "tok_disgusting", "tok_spite")
                corp_keys = ("tok_rational", "tok_market", "tok_standard", "tok_policy")
                w_sub = w_write_unit[:, top_readers]  # [5376, 64]
                aff_dots = np.mean([lex_dirs[k] @ w_sub for k in affect_keys], axis=0)  # [64]
                corp_dots = np.mean([lex_dirs[k] @ w_sub for k in corp_keys], axis=0)    # [64]
                # Count dedicated suppressor neurons where |aff_dots| > 3 SD and opposite sign to corp_dots
                sd_lex = 1.0 / np.sqrt(d_model)
                n_strong_aff = int(np.sum(np.abs(aff_dots) > 2.5 * sd_lex))
                # Also find global top-10 single-token "unfair"/"disgusting" suppressor neurons across all 21,504 neurons in L56!
                unfair_w = lex_dirs["tok_unfair"] @ w_write_unit  # [21504]
                disgust_w = lex_dirs["tok_disgusting"] @ w_write_unit
                worst_unfair = np.argsort(unfair_w)[:10]  # most negative writers onto " unfair"
                scrubbing_l56[cname] = {
                    "top64_bus_readers_mean_abs_read_cos": round(float(np.mean(read_score[top_readers])), 5),
                    "n_readers_with_strong_affect_logit_gt_2p5sd": n_strong_aff,
                    "top10_anti_unfair_neuron_cos": [round(float(unfair_w[i]), 5) for i in worst_unfair],
                    "top10_anti_unfair_spite_read_cos": [round(float(float(w_gate_unit[i] @ u_bus)), 5) for i in worst_unfair],
                    "global_unfair_write_min_cos": round(float(np.min(unfair_w)), 5),
                    "global_disgusting_write_min_cos": round(float(np.min(disgust_w)), 5),
                }

        # Strip temporary `_cos_all` arrays before saving
        for d_grp in (moral_u35_res, moral_uloc_res, lex_res):
            for v in d_grp.values():
                v.pop("_cos_all", None)

        rec = {
            "layer": ell,
            "layer_scalar": round(float(layer_scalars[ell]), 5),
            "skip_gain_to_l35": round(float(gamma_to_35[ell]), 5),
            "random_baselines": {
                "mean_kurtosis": round(float(np.mean(null_kurt)), 4),
                "sd_kurtosis": round(float(np.std(null_kurt, ddof=1)), 4),
                "mean_top64_rms_cos": round(float(np.mean(null_top64_rms)), 5),
                "mean_max_abs_cos": round(float(np.mean(null_max_abs)), 5),
                "rand64_pr_gate": round(rand_pr_gate, 2),
                "rand64_pr_up": round(rand_pr_up, 2),
                "rand64_pr_write": round(rand_pr_write, 2),
                "rand64_gate_abs_cos": round(rand_gate_abs_cos, 5),
                "rand64_write_signed_cos": round(rand_write_cos, 5),
            },
            "moral_u35": moral_u35_res,
            "moral_ulocal": moral_uloc_res,
            "goodfire_lexical_controls": lex_res,
            "cross_axis_writer_jaccard64": cross_axis_jaccard64,
        }
        per_layer_records.append(rec)
        sp35 = moral_u35_res["spite"]
        an_c = lex_res["tok_an"]
        print(
            f"[WeightFunnel L{ell:02d}] ({time.time()-tl0:.1f}s) "
            f"Spite[35]: top64_z={sp35['top64_rms_z_vs_500_random']:+.2f} (p={sp35['top64_rms_p']:.3f}), "
            f"PR_read={sp35['pr_read_gate_of_64']:.1f}/64 vs PR_write={sp35['pr_write_down_of_64']:.1f}/64, "
            f"NonAffectRead={sp35['non_affective_complement_read_share']*100:.1f}% | "
            f"'an' ctrl: top64_z={an_c['top64_rms_z_vs_500_random']:+.2f}, PR_read={an_c['pr_read_gate_of_64']:.1f}",
            flush=True,
        )
        del w_gate, w_up, w_down, w_write_eff, w_write_unit, w_gate_eff, w_gate_unit, w_up_eff, w_up_unit

    payload = {
        "metadata": {
            "model_id": repo,
            "revision": rev,
            "primary_layer": pl,
            "transect_layers": list(TRANSECT_LAYERS),
            "n_neurons_per_layer": 21504,
            "n_neurons_inspected": len(TRANSECT_LAYERS) * 21504,
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "runtime_seconds": round(time.time() - t0, 2),
            "lexical_token_ids": tok_ids,
        },
        "layer_scalars_all_60": [round(float(x), 5) for x in layer_scalars],
        "skip_gain_to_l35_0_to_35": [round(float(gamma_to_35[i]), 5) for i in range(pl + 1)],
        "transect": per_layer_records,
        "late_layer_l56_scrubbing": scrubbing_l56,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "weight_funnel_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    render_weight_funnel_figure_and_report(payload, out_dir)
    return payload


def render_weight_funnel_figure_and_report(payload: dict[str, Any], out_dir: Path) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.edgecolor": BLUE_SLATE,
        "axes.linewidth": 1.5,
    })
    fig, axes = plt.subplots(2, 3, figsize=(19.0, 10.6), dpi=200, facecolor=IVORY)
    fig.subplots_adjust(hspace=0.40, wspace=0.28, top=0.86, bottom=0.09, left=0.05, right=0.98)

    def _style(ax: Any) -> None:
        ax.set_facecolor(IVORY)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(BLUE_SLATE)
        ax.spines["bottom"].set_color(BLUE_SLATE)
        ax.tick_params(colors=ONYX, labelsize=9)
        ax.grid(axis="y", color=BLUE_SLATE, alpha=0.14, linewidth=0.8)

    meta = payload["metadata"]
    tr = payload["transect"]
    layers = [r["layer"] for r in tr]

    fig.text(
        0.05,
        0.945,
        f"Real-Weight Read/Write MLP Dissection in {meta['model_id']} ({meta['n_neurons_inspected']:,} Neurons)",
        fontsize=17,
        fontweight="bold",
        color=ONYX,
    )
    fig.text(
        0.05,
        0.912,
        f"Pinned commit {meta['revision'][:12]} | Exact post-RMSNorm W_down write vs. pre-RMSNorm W_gate/W_up read across L8..L56",
        fontsize=10.5,
        color=SLATE_DEEP,
    )

    # Panel A: Dedicated Writer Z-Score (`top64_rms_z_vs_500_random`) across depth (`u35` Moral Axes vs Lexical Controls)
    ax = axes[0, 0]
    _style(ax)
    for c, col, mk, lw in (
        ("spite", GOLDEN_ORANGE, "o", 2.5),
        ("contamination", BLUE_SLATE, "s", 2.1),
        ("guilt", SAGE_DEEP, "^", 1.9),
        ("awe_dread", COPPER_DEEP, "D", 1.9),
    ):
        zs = [r["moral_u35"][c]["top64_rms_z_vs_500_random"] for r in tr]
        ax.plot(layers, zs, marker=mk, color=col, linewidth=lw, label=f"{c.replace('_',' ').title()} (u35)")
    ax.axhline(1.96, color=SLATE_MIST, linestyle="--", linewidth=1.1, label="p = 0.05 (+1.96z vs 500 rand)")
    ax.axhline(0.0, color=ONYX, linewidth=0.9)
    ax.set_title("A. Dedicated W_down Writers onto Layer-35 Bus", fontsize=11, fontweight="bold", color=ONYX, loc="left")
    ax.set_xlabel("Decoder layer", fontsize=9.5, color=SLATE_DEEP)
    ax.set_ylabel("Top-64 W_down cosine Z-score (vs 500 rand)", fontsize=9.5, color=SLATE_DEEP)
    ax.legend(frameon=False, fontsize=8, loc="best")

    # Panel B: Local Concept Axis Z-Score vs Goodfire Lexical Token Controls (`" an"`, `" unfair"`)
    ax = axes[0, 1]
    _style(ax)
    ax.plot(layers, [r["moral_ulocal"]["spite"]["top64_rms_z_vs_500_random"] for r in tr], marker="o", color=GOLDEN_ORANGE, linewidth=2.4, label="Local Spite[l]")
    ax.plot(layers, [r["moral_ulocal"]["contamination"]["top64_rms_z_vs_500_random"] for r in tr], marker="s", color=BLUE_SLATE, linewidth=2.1, label="Local Contam[l]")
    ax.plot(layers, [r["goodfire_lexical_controls"]["tok_an"]["top64_rms_z_vs_500_random"] for r in tr], marker="^", linestyle="--", color=COPPER_DEEP, linewidth=2.0, label="Goodfire ' an' token")
    ax.plot(layers, [r["goodfire_lexical_controls"]["tok_unfair"]["top64_rms_z_vs_500_random"] for r in tr], marker="v", linestyle=":", color=SAGE_DEEP, linewidth=2.0, label="Lexical ' unfair' token")
    ax.axhline(1.96, color=SLATE_MIST, linestyle="--", linewidth=1.0)
    ax.set_title("B. Moral Axes vs. Goodfire Lexical Token Writers", fontsize=11, fontweight="bold", color=ONYX, loc="left")
    ax.set_xlabel("Decoder layer", fontsize=9.5, color=SLATE_DEEP)
    ax.set_ylabel("Top-64 W_down cosine Z-score", fontsize=9.5, color=SLATE_DEEP)
    ax.legend(frameon=False, fontsize=8, loc="best")

    # Panel C: READ Effective Rank (`PR_gate`) vs WRITE Effective Rank (`PR_write`) out of 64
    ax = axes[0, 2]
    _style(ax)
    pr_r_sp = [r["moral_u35"]["spite"]["pr_read_gate_of_64"] for r in tr]
    pr_w_sp = [r["moral_u35"]["spite"]["pr_write_down_of_64"] for r in tr]
    pr_r_rand = [r["random_baselines"]["rand64_pr_gate"] for r in tr]
    pr_w_rand = [r["random_baselines"]["rand64_pr_write"] for r in tr]
    ax.plot(layers, pr_r_sp, marker="o", color=BLUE_SLATE, linewidth=2.4, label="Spite Top-64 READ rank (W_gate)")
    ax.plot(layers, pr_w_sp, marker="s", color=GOLDEN_ORANGE, linewidth=2.4, label="Spite Top-64 WRITE rank (W_down)")
    ax.plot(layers, pr_r_rand, linestyle="--", color=SLATE_MIST, linewidth=1.4, label="Random-64 READ rank")
    ax.plot(layers, pr_w_rand, linestyle=":", color=COPPER_DEEP, linewidth=1.4, label="Random-64 WRITE rank")
    ax.set_ylim(0, 64)
    ax.set_title("C. High-Rank Read Fan-In vs. Convergent Write", fontsize=11, fontweight="bold", color=ONYX, loc="left")
    ax.set_xlabel("Decoder layer", fontsize=9.5, color=SLATE_DEEP)
    ax.set_ylabel("Participation Ratio (effective rank / 64)", fontsize=9.5, color=SLATE_DEEP)
    ax.legend(frameon=False, fontsize=7.8, loc="lower right")

    # Panel D: Non-Affective-to-Moral Translation Share (`% of READ energy outside incoming 10-concept span`)
    ax = axes[1, 0]
    _style(ax)
    for c, col, mk in (("spite", GOLDEN_ORANGE, "o"), ("contamination", BLUE_SLATE, "s"), ("guilt", SAGE_DEEP, "^"), ("awe_dread", COPPER_DEEP, "D")):
        sh = [r["moral_u35"][c]["non_affective_complement_read_share"] * 100.0 for r in tr]
        ax.plot(layers, sh, marker=mk, color=col, linewidth=2.1, label=c.replace("_", " ").title())
    ax.set_ylim(80, 101)
    ax.set_title("D. Non-Affective Complement Read Share (%)", fontsize=11, fontweight="bold", color=ONYX, loc="left")
    ax.set_xlabel("Decoder layer", fontsize=9.5, color=SLATE_DEEP)
    ax.set_ylabel("Read norm outside 10D affect span (%)", fontsize=9.5, color=SLATE_DEEP)
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    # Panel E: Layer-Scalar Skip Propagation Profile (`layer_scalar` & `Gamma_{l -> 35}` across all 60 layers)
    ax = axes[1, 1]
    _style(ax)
    ls60 = np.array(payload["layer_scalars_all_60"])
    gam36 = np.array(payload["skip_gain_to_l35_0_to_35"])
    ax.plot(np.arange(60), ls60, color=BLUE_SLATE, linewidth=2.0, label="Per-layer scalar (L0..L59)")
    ax.plot(np.arange(len(gam36)), gam36, color=GOLDEN_ORANGE, linewidth=2.4, label="Skip gain Gamma_{l -> 35}")
    ax.axvspan(24, 35, color=PARCHMENT, zorder=0, label="Funnel ignition (L24..L35)")
    ax.set_title("E. Gemma 4 31B Residual Skip Gain into Layer 35", fontsize=11, fontweight="bold", color=ONYX, loc="left")
    ax.set_xlabel("Decoder layer (0 .. 59)", fontsize=9.5, color=SLATE_DEEP)
    ax.set_ylabel("Multiplier", fontsize=9.5, color=SLATE_DEEP)
    ax.legend(frameon=False, fontsize=8, loc="best")

    # Panel F: At Layer 33 & Layer 35, Compare Moral Axes vs Lexical Controls on Kurtosis Z-Score
    ax = axes[1, 2]
    _style(ax)
    r33 = next(r for r in tr if r["layer"] == 33)
    r35 = next(r for r in tr if r["layer"] == 35)
    r56 = next(r for r in tr if r["layer"] == 56)
    cats = ["Spite\n(u35)", "Contam\n(u35)", "Guilt\n(u35)", "Vastness\n(u35)", "' an'\n(lex)", "' unfair'\n(lex)"]
    v33 = [
        r33["moral_u35"]["spite"]["kurtosis_z_vs_500_random"],
        r33["moral_u35"]["contamination"]["kurtosis_z_vs_500_random"],
        r33["moral_u35"]["guilt"]["kurtosis_z_vs_500_random"],
        r33["moral_u35"]["awe_dread"]["kurtosis_z_vs_500_random"],
        r33["goodfire_lexical_controls"]["tok_an"]["kurtosis_z_vs_500_random"],
        r33["goodfire_lexical_controls"]["tok_unfair"]["kurtosis_z_vs_500_random"],
    ]
    v56 = [
        r56["moral_u35"]["spite"]["kurtosis_z_vs_500_random"],
        r56["moral_u35"]["contamination"]["kurtosis_z_vs_500_random"],
        r56["moral_u35"]["guilt"]["kurtosis_z_vs_500_random"],
        r56["moral_u35"]["awe_dread"]["kurtosis_z_vs_500_random"],
        r56["goodfire_lexical_controls"]["tok_an"]["kurtosis_z_vs_500_random"],
        r56["goodfire_lexical_controls"]["tok_unfair"]["kurtosis_z_vs_500_random"],
    ]
    xf = np.arange(len(cats))
    wf = 0.36
    ax.bar(xf - wf / 2, v33, width=wf, color=BLUE_SLATE, edgecolor=BLUE_SLATE, linewidth=1.4, label="Funnel L33 Kurtosis Z")
    ax.bar(xf + wf / 2, v56, width=wf, color=GOLDEN_ORANGE, edgecolor=BLUE_SLATE, linewidth=1.6, label="Late L56 Kurtosis Z")
    ax.axhline(1.96, color=SLATE_MIST, linestyle="--", linewidth=1.0)
    ax.axhline(0.0, color=ONYX, linewidth=1.0)
    ax.set_xticks(xf)
    ax.set_xticklabels(cats, fontsize=8.3)
    ax.set_ylabel("Excess kurtosis Z-score (vs 500 rand)", fontsize=9.5, color=SLATE_DEEP)
    ax.set_title("F. Writer Sparsity Shift: Funnel (L33) vs. Late (L56)", fontsize=11, fontweight="bold", color=ONYX, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="best")

    fig.savefig(out_dir / "fig_weight_funnel_31b.png", dpi=200, facecolor=IVORY)
    fig.savefig(out_dir / "fig_weight_funnel_31b.svg", facecolor=IVORY)
    plt.close(fig)

    # Markdown report
    lines = [
        f"# Real-Weight Read/Write MLP Dissection: `{meta['model_id']}`",
        "",
        f"- **Pinned Revision:** `{meta['revision']}` · **Neurons Inspected:** `{meta['n_neurons_inspected']:,}` across `L8, L16, L22, L26, L30, L33, L35, L56` · **Runtime:** `{meta['runtime_seconds']}s`",
        "",
        "![Real-Weight Read/Write Funnel](fig_weight_funnel_31b.png)",
        "",
        "## Layer-by-Layer `Spite[35]` & `Contamination[35]` vs. Goodfire `' an'` & `' unfair'` Controls",
        "",
        "| Layer | Skip Gain `->L35` | `Spite[35]` Top64 Z (`p`) | `Spite` `PR_read / PR_write` | `Spite` Non-Affect Read% | `Contam[35]` Top64 Z (`p`) | `' an'` Top64 Z (`PR_read`) | `' unfair'` Top64 Z |",
        "| :---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: |",
    ]
    for r in tr:
        sp = r["moral_u35"]["spite"]
        ct = r["moral_u35"]["contamination"]
        an = r["goodfire_lexical_controls"]["tok_an"]
        uf = r["goodfire_lexical_controls"]["tok_unfair"]
        lines.append(
            f"| `L{r['layer']}` | `{r['skip_gain_to_l35']:.4f}` | **`{sp['top64_rms_z_vs_500_random']:+.2f}z`** (`{sp['top64_rms_p']:.3f}`) | "
            f"`{sp['pr_read_gate_of_64']:.1f} / {sp['pr_write_down_of_64']:.1f}` | `{sp['non_affective_complement_read_share']*100:.1f}%` | "
            f"**`{ct['top64_rms_z_vs_500_random']:+.2f}z`** (`{ct['top64_rms_p']:.3f}`) | "
            f"`{an['top64_rms_z_vs_500_random']:+.2f}z` (`{an['pr_read_gate_of_64']:.1f}`) | `{uf['top64_rms_z_vs_500_random']:+.2f}z` |"
        )
    (out_dir / "weight_funnel_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_real_weight_funnel(ROOT_DIR / "runs" / "weight_funnel_gemma-4-31b-it")
