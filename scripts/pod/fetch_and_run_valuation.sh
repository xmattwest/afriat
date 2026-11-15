#!/bin/bash
# fetch_and_run_valuation.sh (POD-SIDE):
#   1. Downloads a pinned HuggingFace model at exact commit REV and verifies every file's
#      exact byte count and SHA-256 against /workspace/config/manifests/<NAME>.tsv.
#   2. Runs Project Afriat Phase 2 Valuation & Rationalization (Option A + Option B)
#      against the locked Phase 0 non-financial concept basis with HF_HUB_OFFLINE=1.
#   3. Removes /models/<NAME> afterward so sequential models never exhaust container disk.

set +x +v +a
set -euo pipefail
export LC_ALL=C

NAME="${1:?missing model name}"
REPO="${2:?missing model repo}"
REV="${3:?missing 40-char revision}"
BATCH_SIZE="${4:-24}"

MAN="/workspace/config/manifests/${NAME}.tsv"
BASIS_NPZ="/workspace/phase0/${NAME}/basis_vectors.npz"
P0_JSON="/workspace/phase0/${NAME}/results.json"
DIR="/models/${NAME}"
OUT="/workspace/runs_valuation/${NAME}"
LOGDIR="/var/log/afriat"
mkdir -p "$LOGDIR" /models /workspace/runs_valuation

[ -f "$MAN" ] || { printf '[af-val] ERROR: manifest missing: %s\n' "$MAN" >&2; exit 2; }
[ -f "$BASIS_NPZ" ] || { printf '[af-val] ERROR: locked basis missing: %s\n' "$BASIS_NPZ" >&2; exit 2; }
[ -f "$P0_JSON" ] || { printf '[af-val] ERROR: phase0 json missing: %s\n' "$P0_JSON" >&2; exit 2; }

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1
}
bytes_of() { wc -c < "$1" | tr -d ' '; }

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

printf '[af-val] Downloading & SHA-256 verifying %s (%s @ %s)...\n' "$NAME" "$REPO" "${REV:0:12}"
while IFS="$TAB" read -r path size sha src extra; do
  case "$path" in ''|'#'*) continue ;; esac
  [[ "$path" =~ $re_path ]] || { printf '[af-val] bad path in manifest: %s\n' "$path" >&2; exit 2; }
  case "/$path/" in */./*|*/../*) printf '[af-val] traversal in manifest: %s\n' "$path" >&2; exit 2 ;; esac
  [[ "$size" =~ $re_int ]] || { printf '[af-val] bad size for %s\n' "$path" >&2; exit 2; }
  [[ "$sha" =~ $re_sha ]] || { printf '[af-val] bad sha256 for %s\n' "$path" >&2; exit 2; }
  [ -z "${extra:-}" ] || { printf '[af-val] extra column for %s\n' "$path" >&2; exit 2; }
  case "$seen" in *" $path "*) printf '[af-val] duplicate %s\n' "$path" >&2; exit 2 ;; esac
  seen="$seen$path "

  dest="$DIR/$path"
  mkdir -p "$(dirname "$dest")"
  ok=0
  for try in 1 2 3; do
    rm -f "$dest.part"
    if curl -q -sS --fail -L --proto '=https' --proto-redir '=https' --max-redirs 5 --tlsv1.2 \
         --connect-timeout 20 --max-time 3600 -o "$dest.part" \
         "https://huggingface.co/$REPO/resolve/$REV/$path" 2>> "$LOGDIR/download_val_${NAME}.log"; then
      if [ "$(bytes_of "$dest.part")" = "$size" ] && [ "$(sha256_of "$dest.part")" = "$sha" ]; then
        mv "$dest.part" "$dest"
        ok=1
        break
      fi
      printf '[af-val] WEIGHTS_MISMATCH %s (try %s)\n' "$path" "$try"
    else
      printf '[af-val] download failed: %s (try %s)\n' "$path" "$try"
    fi
  done
  [ "$ok" = 1 ] || { printf '[af-val] WEIGHTS_BAD %s after 3 tries\n' "$path" >&2; exit 3; }
  n=$((n + 1))
  total=$((total + size))
done < "$MAN"

[ "$n" -gt 0 ] || { printf '[af-val] empty manifest\n' >&2; exit 3; }
[ "$(find "$DIR" -type f | grep -c .)" = "$n" ] || { printf '[af-val] unexpected files in %s\n' "$DIR" >&2; exit 3; }
dl_dt=$(( $(date +%s) - dl_t0 ))
[ "$dl_dt" -gt 0 ] || dl_dt=1
printf '[af-val] WEIGHTS_OK model=%s files=%s gb=%.2f secs=%s\n' \
  "$NAME" "$n" "$(awk -v b="$total" 'BEGIN {printf "%.2f", b/1e9}')" "$dl_dt"

rm -rf "$OUT"
mkdir -p "$OUT"

printf '[af-val] Starting offline Phase 2 valuation & rationalization for %s...\n' "$NAME"
(
  cd /workspace
  env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
      PYTHONPATH=/workspace \
      python3 -m afriat.valuation \
        --model "$DIR" \
        --model-label "$REPO" \
        --basis-npz "$BASIS_NPZ" \
        --phase0-json "$P0_JSON" \
        --dtype bfloat16 \
        --device-map cuda \
        --batch-size "$BATCH_SIZE" \
        --gen-batch-size 8 \
        --max-new-tokens 84 \
        --no-plots \
        --out-dir "$OUT"
) 2>&1 | tee "$LOGDIR/val_${NAME}.log"

[ -s "$OUT/valuation_results.json" ] || { printf '[af-val] ERROR: valuation_results.json missing for %s\n' "$NAME" >&2; exit 4; }
rm -rf "$DIR"
printf '[af-val] MODEL_DONE %s\n' "$NAME"
