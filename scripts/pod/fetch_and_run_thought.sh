#!/bin/bash
# fetch_and_run_thought.sh (POD-SIDE):
#   1. Downloads a pinned HuggingFace model at exact commit REV and verifies every file's
#      exact byte count and SHA-256 against /workspace/config/manifests/<NAME>.tsv.
#   2. Runs Gemma 4 31B `<thought>` Scratchpad Pass + Token-by-Token Layer 35 Moral Highlighter
#      against the locked Phase 0 non-financial concept basis with HF_HUB_OFFLINE=1.
#   3. Removes /models/<NAME> afterward so container disk stays clean.

set +x +v +a
set -euo pipefail
export LC_ALL=C

NAME="${1:?missing model name}"
REPO="${2:?missing model repo}"
REV="${3:?missing 40-char revision}"
BATCH_SIZE="${4:-16}"

MAN="/workspace/config/manifests/${NAME}.tsv"
BASIS_NPZ="/workspace/phase0/${NAME}/basis_vectors.npz"
P0_JSON="/workspace/phase0/${NAME}/results.json"
DIR="/models/${NAME}"
OUT="/workspace/runs_thought/${NAME}"
LOGDIR="/var/log/afriat"
mkdir -p "$LOGDIR" /models /workspace/runs_thought

[ -f "$MAN" ] || { printf '[af-thought] ERROR: manifest missing: %s\n' "$MAN" >&2; exit 2; }
[ -f "$BASIS_NPZ" ] || { printf '[af-thought] ERROR: locked basis missing: %s\n' "$BASIS_NPZ" >&2; exit 2; }
[ -f "$P0_JSON" ] || { printf '[af-thought] ERROR: phase0 json missing: %s\n' "$P0_JSON" >&2; exit 2; }

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1
}
bytes_of() { wc -c < "$1" | tr -d ' '; }

download_parallel_range() {
  python3 - "$1" "$2" "$3" <<'PY'
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys
import time
import urllib.request

url, dest, total = sys.argv[1], sys.argv[2], int(sys.argv[3])
chunk_sz = 64 * 1024 * 1024
ranges = [(s, min(s + chunk_sz - 1, total - 1)) for s in range(0, total, chunk_sz)]
fd = os.open(dest, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644)
try:
    os.ftruncate(fd, total)

    def fetch_slice(bounds):
        start, end = bounds
        expected = end - start + 1
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    url,
                    headers={"Range": f"bytes={start}-{end}", "User-Agent": "afriat-fetch/1.0"},
                )
                with urllib.request.urlopen(req, timeout=90) as resp:
                    if resp.status != 206:
                        raise RuntimeError(f"HTTP {resp.status} (expected 206)")
                    pos = start
                    got = 0
                    while True:
                        buf = resp.read(4 * 1024 * 1024)
                        if not buf:
                            break
                        written = 0
                        while written < len(buf):
                            written += os.pwrite(fd, memoryview(buf)[written:], pos + written)
                        pos += len(buf)
                        got += len(buf)
                    if got != expected:
                        raise RuntimeError(f"short range {got} != {expected}")
                    return
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1.0 * (attempt + 1))

    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(fetch_slice, r) for r in ranges]
        for f in as_completed(futs):
            f.result()
finally:
    os.close(fd)
PY
}

rm -rf "$DIR"
mkdir -p "$DIR"

re_int='^[0-9]+$'
re_path='^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$'
re_sha='^[0-9a-f]{64}$'
n=0
total=0
seen=" "
TAB="$(printf '\t')"
dl_t0="$(date +%s)"

printf '[af-thought] Downloading & SHA-256 verifying %s (%s @ %s)...\n' "$NAME" "$REPO" "${REV:0:12}"
while IFS="$TAB" read -r path size sha src extra; do
  case "$path" in ''|'#'*) continue ;; esac
  [[ "$path" =~ $re_path ]] || { printf '[af-thought] bad path in manifest: %s\n' "$path" >&2; exit 2; }
  case "/$path/" in */./*|*/../*) printf '[af-thought] traversal in manifest: %s\n' "$path" >&2; exit 2 ;; esac
  [[ "$size" =~ $re_int ]] || { printf '[af-thought] bad size for %s\n' "$path" >&2; exit 2; }
  [[ "$sha" =~ $re_sha ]] || { printf '[af-thought] bad sha256 for %s\n' "$path" >&2; exit 2; }
  [ -z "${extra:-}" ] || { printf '[af-thought] extra column for %s\n' "$path" >&2; exit 2; }
  case "$seen" in *" $path "*) printf '[af-thought] duplicate %s\n' "$path" >&2; exit 2 ;; esac
  seen="$seen$path "

  dest="$DIR/$path"
  url="https://huggingface.co/$REPO/resolve/$REV/$path"
  mkdir -p "$(dirname "$dest")"
  ok=0
  for try in 1 2 3; do
    rm -f "$dest.part"
    dl_ok=0
    if [ "$size" -gt 67108864 ] && [ "$try" = 1 ]; then
      if download_parallel_range "$url" "$dest.part" "$size" 2>> "$LOGDIR/download_thought_${NAME}.log"; then
        dl_ok=1
      fi
    fi
    if [ "$dl_ok" = 0 ]; then
      rm -f "$dest.part"
      if curl -q -sS --fail -L --proto '=https' --proto-redir '=https' --max-redirs 5 --tlsv1.2 \
           --connect-timeout 20 --max-time 3600 -o "$dest.part" \
           "$url" 2>> "$LOGDIR/download_thought_${NAME}.log"; then
        dl_ok=1
      fi
    fi
    if [ "$dl_ok" = 1 ]; then
      if [ "$(bytes_of "$dest.part")" = "$size" ] && [ "$(sha256_of "$dest.part")" = "$sha" ]; then
        mv "$dest.part" "$dest"
        ok=1
        break
      fi
      printf '[af-thought] WEIGHTS_MISMATCH %s (try %s)\n' "$path" "$try"
    else
      printf '[af-thought] download failed: %s (try %s)\n' "$path" "$try"
    fi
  done
  [ "$ok" = 1 ] || { printf '[af-thought] WEIGHTS_BAD %s after 3 tries\n' "$path" >&2; exit 3; }
  n=$((n + 1))
  total=$((total + size))
done < "$MAN"

[ "$n" -gt 0 ] || { printf '[af-thought] empty manifest\n' >&2; exit 3; }
[ "$(find "$DIR" -type f | grep -c .)" = "$n" ] || { printf '[af-thought] unexpected files in %s\n' "$DIR" >&2; exit 3; }
dl_dt=$(( $(date +%s) - dl_t0 ))
[ "$dl_dt" -gt 0 ] || dl_dt=1
printf '[af-thought] WEIGHTS_OK model=%s files=%s gb=%.2f secs=%s\n' \
  "$NAME" "$n" "$(awk -v b="$total" 'BEGIN {printf "%.2f", b/1e9}')" "$dl_dt"

rm -rf "$OUT"
mkdir -p "$OUT"

printf '[af-thought] Starting offline `<thought>` scratchpad & Moral Highlighter pass for %s...\n' "$NAME"
(
  cd /workspace
  env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      PYTHONPATH=/workspace \
      python3 -m afriat.thought_probe \
        --model "$DIR" \
        --model-label "$REPO" \
        --basis-npz "$BASIS_NPZ" \
        --phase0-json "$P0_JSON" \
        --dtype bfloat16 \
        --device-map cuda \
        --batch-size "$BATCH_SIZE" \
        --gen-batch-size 3 \
        --max-new-tokens 768 \
        --no-plots \
        --out-dir "$OUT"
) 2>&1 | tee "$LOGDIR/thought_${NAME}.log"

[ -s "$OUT/thought_results.json" ] || { printf '[af-thought] ERROR: thought_results.json missing for %s\n' "$NAME" >&2; exit 4; }
rm -rf "$DIR"
printf '[af-thought] MODEL_DONE %s\n' "$NAME"
