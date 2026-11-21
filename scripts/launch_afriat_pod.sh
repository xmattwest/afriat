#!/bin/bash
# launch_afriat_pod.sh: launch one ephemeral RunPod GPU pod, run Project Afriat
# experiments (Phase 0 discovery, Phase 1 inequity, or Phase 2 valuation) on pinned
# open-weight models, retrieve results locally, render Dawn State reports, and tear
# down the pod cleanly (verifying 0 pods remain).
#
# Security & cost guardrails:
#   - Write key read ONLY from macOS Keychain ($AFRIAT_KEYCHAIN_SERVICE) into an
#     unexported variable and passed to curl via stdin (-H @-).
#   - Screen-safe stdout (no API keys, pod IDs, or public IPs printed).
#   - Dry-run unless --confirm-cost X.XX matches the live worst-case bound.
#   - Dual-layer pod deletion: pod-side lifetime watchdog + laptop-side EXIT/INT/TERM trap.

set +x +v +a
set -uo pipefail
umask 077
export LC_ALL=C
unset SSLKEYLOGFILE

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
readonly API="https://api.runpod.io/v2"
PULL_ALLOWANCE_MIN=20
readonly KEYCHAIN_SERVICE="${AFRIAT_KEYCHAIN_SERVICE:-runpod-api-key}"
readonly KEY_FILE="${AFRIAT_RUNPOD_KEY_FILE:-/etc/afriat/runpod.key}"
readonly SSH_KEY="${AFRIAT_SSH_KEY:-$HOME/.ssh/id_ed25519}"
LEDGER="$ROOT/logs/budget.tsv"

MODELS_CSV="gemma-4-12b-it,phi-4-14b"
TASK_MODE="discover"
ESCALATE_MODEL=""
LIFETIME_MIN="45"
GPU_ID="NVIDIA RTX PRO 6000 Blackwell Server Edition"
DISK_GB="90"
BATCH_SIZE="16"
STEER_ALPHA="1.25"
DC="us"
CONFIRM=""
EXFIL_AZURE=0
DETACH=0
USE_ACR_SLIM=0
CUSTOM_IMAGE=""
SMOKE_BOOT_ONLY=0
RESUME_FUNNEL=""
AZ_ACCOUNT="${AFRIAT_AZURE_STORAGE_ACCOUNT:-}"
AZ_CONTAINER="${AFRIAT_AZURE_CONTAINER:-afriat-runs}"

while [ $# -gt 0 ]; do
  case "$1" in
    --task) TASK_MODE="${2:-}"; shift ;;
    --models) MODELS_CSV="${2:-}"; shift ;;
    --escalate-if-both-fail) ESCALATE_MODEL="${2:-}"; shift ;;
    --lifetime-min) LIFETIME_MIN="${2:-}"; shift ;;
    --gpu-id) GPU_ID="${2:-}"; shift ;;
    --disk-gb) DISK_GB="${2:-}"; shift ;;
    --batch-size) BATCH_SIZE="${2:-}"; shift ;;
    --steering-alpha) STEER_ALPHA="${2:-}"; shift ;;
    --dc) DC="${2:-}"; shift ;;
    --exfil-azure) EXFIL_AZURE=1 ;;
    --detach) DETACH=1; EXFIL_AZURE=1 ;;
    --use-acr-slim) USE_ACR_SLIM=1 ;;
    --image) CUSTOM_IMAGE="${2:-}"; shift ;;
    --smoke-boot-only) SMOKE_BOOT_ONLY=1 ;;
    --resume-funnel) RESUME_FUNNEL="${2:-}"; shift ;;
    --az-account) AZ_ACCOUNT="${2:-}"; shift ;;
    --az-container) AZ_CONTAINER="${2:-}"; shift ;;
    --confirm-cost) CONFIRM="${2:-}"; shift ;;
    *) printf 'unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
  shift
done

say() { printf '%s\n' "$*"; }
die() { local c="$1"; shift; printf 'STOP: %s\n' "$*" >&2; exit "$c"; }

for t in curl jq ssh ssh-keygen python3 perl tar shasum; do
  command -v "$t" >/dev/null 2>&1 || die 2 "$t not found"
done

[[ "$LIFETIME_MIN" =~ ^[0-9]+$ ]] && [ "$LIFETIME_MIN" -ge 10 ] && [ "$LIFETIME_MIN" -le 240 ] ||
  die 2 "--lifetime-min must be 10..240"
[[ "$DISK_GB" =~ ^[0-9]+$ ]] && [ "$DISK_GB" -ge 40 ] && [ "$DISK_GB" -le 300 ] ||
  die 2 "--disk-gb must be 40..300"
[[ "$DC" =~ ^(us|any|[A-Z0-9-]+(,[A-Z0-9-]+)*)$ ]] ||
  die 2 "--dc must be us, any, or comma-separated DC IDs"

mkdir -p "$ROOT/logs" "$ROOT/runs"

PINS="$ROOT/config/pins.json"
[ -f "$PINS" ] || die 2 "missing $PINS"
if [ -n "$CUSTOM_IMAGE" ]; then
  IMAGE="$CUSTOM_IMAGE"
  PULL_ALLOWANCE_MIN=5
elif [ "$USE_ACR_SLIM" = 1 ]; then
  ACR_REPO="$(jq -r '.acr_slim_image.repo // empty' "$PINS")"
  ACR_DIGEST="$(jq -r '.acr_slim_image.digest // empty' "$PINS")"
  [ -n "$ACR_REPO" ] && [[ "$ACR_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]] ||
    die 2 "missing .acr_slim_image.repo/digest in pins.json"
  IMAGE="${ACR_REPO}@${ACR_DIGEST}"
  PULL_ALLOWANCE_MIN=5
else
  DIGEST="$(jq -r '.vllm_image.index_digest' "$PINS")"
  [[ "$DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]] || die 2 "bad image digest in pins.json"
  IMAGE="vllm/vllm-openai@$DIGEST"
fi

[ "$TASK_MODE" = "loop-compact" ] && TASK_MODE="loop_compact"
case "$TASK_MODE" in discover|inequity|valuation|thought|showcase|showcase_r2|frontier|vastness|funnel|optimize|loop_compact) ;; *) die 2 "--task must be discover, inequity, valuation, thought, showcase, showcase_r2, frontier, vastness, funnel, optimize, or loop_compact" ;; esac

BOOT="$HERE/pod/boot_afriat.sh"
RUNNER="$HERE/pod/fetch_and_run_model.sh"
INEQ_RUNNER="$HERE/pod/fetch_and_run_inequity.sh"
VAL_RUNNER="$HERE/pod/fetch_and_run_valuation.sh"
THOUGHT_RUNNER="$HERE/pod/fetch_and_run_thought.sh"
SHOWCASE_RUNNER="$HERE/pod/fetch_and_run_showcase.sh"
FRONTIER_RUNNER="$HERE/pod/fetch_and_run_frontier.sh"
VASTNESS_RUNNER="$HERE/pod/fetch_and_run_vastness.sh"
FUNNEL_RUNNER="$HERE/pod/fetch_and_run_funnel.sh"
OPTIMIZE_RUNNER="$HERE/pod/fetch_and_run_optimize.sh"
LOOP_COMPACT_RUNNER="$HERE/pod/fetch_and_run_loop_compact.sh"
EXFIL_RUNNER="$HERE/pod/run_autonomous_exfil.sh"
bash -n "$BOOT" || die 2 "syntax error in boot_afriat.sh"
bash -n "$RUNNER" || die 2 "syntax error in fetch_and_run_model.sh"
bash -n "$INEQ_RUNNER" || die 2 "syntax error in fetch_and_run_inequity.sh"
bash -n "$VAL_RUNNER" || die 2 "syntax error in fetch_and_run_valuation.sh"
bash -n "$THOUGHT_RUNNER" || die 2 "syntax error in fetch_and_run_thought.sh"
bash -n "$SHOWCASE_RUNNER" || die 2 "syntax error in fetch_and_run_showcase.sh"
bash -n "$FRONTIER_RUNNER" || die 2 "syntax error in fetch_and_run_frontier.sh"
bash -n "$VASTNESS_RUNNER" || die 2 "syntax error in fetch_and_run_vastness.sh"
bash -n "$FUNNEL_RUNNER" || die 2 "syntax error in fetch_and_run_funnel.sh"
bash -n "$OPTIMIZE_RUNNER" || die 2 "syntax error in fetch_and_run_optimize.sh"
bash -n "$LOOP_COMPACT_RUNNER" || die 2 "syntax error in fetch_and_run_loop_compact.sh"
bash -n "$EXFIL_RUNNER" || die 2 "syntax error in run_autonomous_exfil.sh"

[ -f "$SSH_KEY" ] && [ -f "$SSH_KEY.pub" ] || die 2 "SSH key missing ($SSH_KEY)"
PUBKEY="$(cat "$SSH_KEY.pub")"
FPR="$(ssh-keygen -lf "$SSH_KEY.pub" | cut -d' ' -f2)"

IFS=',' read -r -a MODEL_LIST <<< "$MODELS_CSV"
[ "${#MODEL_LIST[@]}" -ge 1 ] || die 2 "--models must list at least one model"
if [ "$DETACH" = 1 ] && [ "${#MODEL_LIST[@]}" -ne 1 ]; then
  die 2 "--detach currently requires a single model in --models"
fi

AZ_SAS=""
if [ "$EXFIL_AZURE" = 1 ]; then
  command -v az >/dev/null 2>&1 || die 2 "az CLI not found (required for --exfil-azure / --detach)"
  SAS_EXP_UTC="$(python3 - "$LIFETIME_MIN" <<'PY'
import datetime, sys
mins = int(sys.argv[1]) + 45
print((datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=mins)).strftime("%Y-%m-%dT%H:%M:%SZ"))
PY
  )"
  AZ_SAS="$(az storage container generate-sas \
    --account-name "$AZ_ACCOUNT" \
    --name "$AZ_CONTAINER" \
    --permissions cw \
    --expiry "$SAS_EXP_UTC" \
    --https-only \
    --as-user \
    --auth-mode login \
    --output tsv 2>/dev/null)" || die 2 "failed to mint User Delegation SAS on $AZ_ACCOUNT/$AZ_CONTAINER (check 'az login')"
  [ -n "$AZ_SAS" ] || die 2 "empty User Delegation SAS returned for $AZ_ACCOUNT/$AZ_CONTAINER"
fi

find_phase0_dir() {
  local m="$1" d
  d="$(ls -dt "$ROOT"/runs/"${m}"_20* 2>/dev/null | head -1 || true)"
  [ -n "$d" ] && [ -s "$d/basis_vectors.npz" ] && [ -s "$d/results.json" ] && printf '%s' "$d"
}

check_model_pin() {
  local m="$1" repo rev p0
  repo="$(jq -r --arg m "$m" '.models[] | select(.name == $m) | .repo' "$PINS")"
  rev="$(jq -r --arg m "$m" '.models[] | select(.name == $m) | .revision' "$PINS")"
  [ -n "$repo" ] && [ -n "$rev" ] || die 2 "unknown model '$m' in pins.json"
  [ -f "$ROOT/config/manifests/$m.tsv" ] || die 2 "missing manifest config/manifests/$m.tsv"
  if [ "$TASK_MODE" = inequity ] || [ "$TASK_MODE" = valuation ] || [ "$TASK_MODE" = thought ] || [ "$TASK_MODE" = showcase ] || [ "$TASK_MODE" = showcase_r2 ] || [ "$TASK_MODE" = frontier ] || [ "$TASK_MODE" = vastness ] || [ "$TASK_MODE" = funnel ]; then
    p0="$(find_phase0_dir "$m" || true)"
    [ -n "$p0" ] || die 2 "missing Phase 0 basis_vectors.npz/results.json for '$m' in runs/"
  fi
}
for m in "${MODEL_LIST[@]}"; do check_model_pin "$m"; done
[ -n "$ESCALATE_MODEL" ] && check_model_pin "$ESCALATE_MODEL"

if [ -f "$KEY_FILE" ] && [ -r "$KEY_FILE" ]; then
  RP_KEY="$(tr -d '\r\n' < "$KEY_FILE")"
elif command -v security >/dev/null 2>&1; then
  RP_KEY="$(security find-generic-password -s "$KEYCHAIN_SERVICE" -a "${USER:-$(id -un)}" -w 2>/dev/null)" ||
    die 2 "no write key in Keychain (service $KEYCHAIN_SERVICE)"
else
  die 2 "no readable RunPod key file ($KEY_FILE) and macOS security CLI unavailable"
fi
[ -n "$RP_KEY" ] || die 2 "write key is empty"

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT

rp() {
  local m="$1" p="$2" out="$3" body="${4:-}" code
  if [ -n "$body" ]; then
    code="$(printf 'Authorization: Bearer %s\n' "$RP_KEY" | curl -q -sS --proto '=https' --tlsv1.2 \
      --connect-timeout 10 --max-time 60 -o "$out" -w '%{http_code}' -X "$m" -H @- \
      -H 'Content-Type: application/json' --data-binary @"$body" "$API$p" 2>/dev/null)"
  else
    code="$(printf 'Authorization: Bearer %s\n' "$RP_KEY" | curl -q -sS --proto '=https' --tlsv1.2 \
      --connect-timeout 10 --max-time 60 -o "$out" -w '%{http_code}' -X "$m" -H @- "$API$p" 2>/dev/null)"
  fi
  printf '%s' "${code:-000}"
}

rp_get() {
  local code i
  for i in 1 2 3; do
    code="$(rp GET "$1" "$2")"
    case "$code" in 000|429|5??) sleep 3 ;; *) break ;; esac
  done
  printf '%s' "$code"
}

active_pods_count() {
  local code
  code="$(rp_get /pods "$T/pods.json")"
  [ "$code" = 200 ] || { printf 'ERR'; return; }
  jq '[(.pods // .)[] | select((.status // "") != "TERMINATED")] | length' "$T/pods.json"
}

# Build clean reproducible bundle of afriat package + stimuli + manifests + pod runners
BUNDLE_TGZ="$T/afriat_bundle.tgz"
COPYFILE_DISABLE=1 tar --no-xattrs -C "$ROOT" -czf "$BUNDLE_TGZ" \
  afriat stimuli config/pins.json config/manifests \
  scripts/pod/fetch_and_run_model.sh scripts/pod/fetch_and_run_inequity.sh \
  scripts/pod/fetch_and_run_valuation.sh scripts/pod/fetch_and_run_thought.sh \
  scripts/pod/fetch_and_run_showcase.sh scripts/pod/fetch_and_run_frontier.sh \
  scripts/pod/fetch_and_run_vastness.sh scripts/pod/fetch_and_run_funnel.sh \
  scripts/pod/fetch_and_run_optimize.sh scripts/pod/fetch_and_run_loop_compact.sh \
  scripts/pod/run_autonomous_exfil.sh
BUNDLE_SHA="$(shasum -a 256 "$BUNDLE_TGZ" | cut -d' ' -f1)"
BUNDLE_KB="$(awk -v b="$(wc -c < "$BUNDLE_TGZ")" 'BEGIN {printf "%.1f", b/1024}')"

DCJSON="[]"
DC_SHOW="any (globalNetwork=true preferred)"
code="$(rp_get "/catalog/datacenters?include=GPU_AVAILABILITY" "$T/dcs.json")"
[ "$code" = 200 ] || die 3 "datacenter catalog: HTTP $code"
STOCKED="$(jq -c --arg g "$GPU_ID" '[(.dataCenters // .)[] | select([(.gpuAvailability // [])[] | .id] | index($g)) | {id, region, globalNetwork: (.globalNetwork // false), comp: ((.compliance // []) | length)}]' "$T/dcs.json")"
if [ "$DC" = any ]; then
  # Prefer proven high-speed DCs & globalNetwork=true nodes; always exclude NAT-stalled US-PA-1
  DCJSON="$(printf '%s' "$STOCKED" | jq -c '[.[] | select(.id != "US-PA-1") | .id]')"
  [ "$(printf '%s' "$DCJSON" | jq 'length')" -gt 0 ] && DC_SHOW="$(printf '%s' "$DCJSON" | jq -r 'join(", ")') (US-PA-1 excluded)"
elif [ "$DC" = us ]; then
  DCJSON="$(printf '%s' "$STOCKED" | jq -c '[.[] | select(.region == "NORTH_AMERICA" and .id != "US-PA-1") | .id] | sort')"
  [ "$(printf '%s' "$DCJSON" | jq 'length')" -gt 0 ] || die 3 "no direct-TCP DC in --dc us lists $GPU_ID; try --dc any"
  DC_SHOW="$(printf '%s' "$DCJSON" | jq -r 'join(", ")') (--dc us, US-PA-1 excluded)"
else
  DCJSON="$(printf '%s' "$DC" | jq -R -c 'split(",")')"
  [ "$(printf '%s' "$DCJSON" | jq 'length')" -gt 0 ] || die 3 "empty --dc list"
  DC_SHOW="$(printf '%s' "$DCJSON" | jq -r 'join(", ")') (--dc $DC)"
fi

code="$(rp_get "/catalog/gpus?include=AVAILABILITY&product=POD" "$T/gpus.json")"
[ "$code" = 200 ] || die 3 "GPU catalog: HTTP $code"
PRICE="$(jq -r --arg g "$GPU_ID" '[(if type=="array" then . else (.gpus // .data // .items) end)[] | select(.id == $g)][0].price.secure // empty' "$T/gpus.json")"
STOCK="$(jq -r --arg g "$GPU_ID" '[(if type=="array" then . else (.gpus // .data // .items) end)[] | select(.id == $g)][0].availability // "?"' "$T/gpus.json")"
[[ "$PRICE" =~ ^[0-9]+(\.[0-9]+)?$ ]] && awk -v p="$PRICE" 'BEGIN {exit !(p > 0)}' || die 3 "no Secure Cloud price for $GPU_ID"

WORST="$(awk -v p="$PRICE" -v l="$LIFETIME_MIN" -v a="$PULL_ALLOWANCE_MIN" -v d="$DISK_GB" \
  'BEGIN { h = (l + a) / 60; c = p * h + d * 0.10 / 730 * h; printf "%.2f", int(c * 100 + 0.999999) / 100 }')"

LIVE_N="$(active_pods_count)"
[ "$LIVE_N" = ERR ] && die 3 "could not list pods; aborting"

build_create() {
  local pod_name="$1"
  local envjson
  envjson="$(jq -n --arg life "$((LIFETIME_MIN * 60))" --arg pk "$PUBKEY" --arg fpr "$FPR" \
    '{AF_LIFETIME_S: $life, PUBLIC_KEY: $pk, AF_SSH_KEY_FPR: $fpr}')"
  local min_cuda="12.8"
  case "$GPU_ID" in *Blackwell*) min_cuda="13.0" ;; esac
  jq -n --arg name "$pod_name" --arg image "$IMAGE" --arg gpu "$GPU_ID" --arg cuda "$min_cuda" --argjson disk "$DISK_GB" \
    --rawfile boot "$BOOT" --argjson env "$envjson" --argjson dcs "$DCJSON" \
    '{name: $name, image: $image, cloud: "SECURE",
      gpu: {id: $gpu, count: 1, minCudaVersion: $cuda, minRamPerGpu: 120},
      ports: ["22/tcp"], disk: $disk, startSsh: false, startJupyter: false,
      entrypoint: ["/bin/bash", "-c"], cmd: [$boot], env: $env}
     + (if ($dcs | length) > 0 then {dataCenterIds: $dcs} else {} end)' > "$T/create.json"
}

build_create "af-discover-$(date -u +%Y%m%dT%H%M%SZ)-0000"
CREATE_BYTES="$(wc -c < "$T/create.json" | tr -d ' ')"
[ "$CREATE_BYTES" -lt 100000 ] || die 2 "create payload $CREATE_BYTES bytes exceeds 100 KB"

say "Project Afriat Launch Plan ($TASK_MODE)"
say "  models:         $MODELS_CSV${ESCALATE_MODEL:+ (auto-escalate if both fail gate: $ESCALATE_MODEL)}"
say "  GPU:            $GPU_ID, Secure Cloud, \$$PRICE/hr (stock: $STOCK)"
say "  data centers:   $DC_SHOW"
say "  image:          $IMAGE"
say "  lifetime cap:   $LIFETIME_MIN min (enforced by pod watchdog + laptop trap)"
say "  container disk: $DISK_GB GB (weights cleaned after each model)"
say "  code bundle:    ${BUNDLE_KB} KB (sha256 ${BUNDLE_SHA:0:16}...)"
if [ "$EXFIL_AZURE" = 1 ]; then
  say "  azure exfil:    https://${AZ_ACCOUNT}.blob.core.windows.net/${AZ_CONTAINER} (write-only User Delegation SAS, detach=$DETACH)"
fi
say "  worst case:     \$$WORST  = ($LIFETIME_MIN + $PULL_ALLOWANCE_MIN min pull) x \$$PRICE/hr + disk"
say "  active pods:    $LIVE_N on account right now"

[ "$LIVE_N" = 0 ] || die 4 "account already has $LIVE_N non-terminated pod(s); refusing to launch"

if [ "$CONFIRM" != "$WORST" ]; then
  say ""
  say "Dry run complete: nothing created. To launch, pass:  --confirm-cost $WORST"
  rm -rf "$T"
  exit 0
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RAND="$(od -An -N2 -tx1 /dev/urandom | tr -d ' \n')"
NAME="af-discover-$STAMP-$RAND"
DETAILS="$ROOT/logs/launch_$NAME.details"
BOOTLOG="$ROOT/logs/launch_$NAME.bootlog"
: > "$DETAILS"
detail() { printf '%s %s\n' "$(date -u +%H:%M:%SZ)" "$*" >> "$DETAILS"; }

POD_ID=""
CREATE_SENT=0
TORN=0
DETACHED_HANDOFF=0
DETACHED_PREFIX=""
CREATE_T=0
GONE=0
DEADLINE=0
SSH_HOST=""
SSH_PORT=""
KNOWN_HOSTS=""

af_ssh() {
  local timeout_s="$1"; shift
  local opts=(-F /dev/null -i "$SSH_KEY" -o IdentitiesOnly=yes -o IdentityAgent=none
    -o StrictHostKeyChecking=yes -o UserKnownHostsFile="$KNOWN_HOSTS"
    -o GlobalKnownHostsFile=/dev/null -o ForwardAgent=no -o ForwardX11=no
    -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=6
    -o LogLevel=ERROR -p "$SSH_PORT")
  perl -e 'alarm shift; exec @ARGV' "$timeout_s" ssh "${opts[@]}" "root@$SSH_HOST" "$@"
}

ship_exfil_env() {
  local run_prefix="$1"
  local task_name="$2"
  local model_name="$3"
  local out_dir_pod="$4"
  local res_json_name="$5"
  [ "$EXFIL_AZURE" = 1 ] || return 0
  local envf="$T/exfil_env"
  umask 077
  cat > "$envf" <<EOF
EXFIL_CONTAINER_URL='https://${AZ_ACCOUNT}.blob.core.windows.net/${AZ_CONTAINER}'
EXFIL_SAS='${AZ_SAS}'
EXFIL_PREFIX='${run_prefix}'
EXFIL_TASK='${task_name}'
EXFIL_MODEL='${model_name}'
EXFIL_OUT_DIR='${out_dir_pod}'
EXFIL_RES_JSON='${res_json_name}'
EOF
  af_ssh 30 "umask 077 && mkdir -p /run/afriat && cat > /run/afriat/exfil_env && chmod 600 /run/afriat/exfil_env" < "$envf"
  rm -f "$envf"
}

teardown() {
  [ "$TORN" = 1 ] && return
  TORN=1
  trap '' INT TERM HUP
  if [ "$DETACHED_HANDOFF" = 1 ]; then
    [ -f "$LEDGER" ] || printf 'utc\tpod_name\tworst_usd\testimate_usd\tstatus\n' > "$LEDGER"
    printf '%s\t%s\t%s\t%s\t%s\n' "$STAMP" "$NAME" "$WORST" "pending_exfil" "detached:$DETACHED_PREFIX" >> "$LEDGER"
    rm -rf "$T"
    say ""
    say "✓ Pod handed off to autonomous Azure Blob exfil + pod-side self-delete watchdog!"
    say "  Azure prefix:   ${AZ_ACCOUNT}/${AZ_CONTAINER}/${DETACHED_PREFIX}"
    say "  Safe to close your laptop or walk away now."
    say "  Check status / pull & render anytime with:"
    say "    ./scripts/sync_afriat_runs.sh --watch"
    return 0
  fi
  say ""
  say "Teardown"
  if [ -n "$SSH_HOST" ] && [ -n "$KNOWN_HOSTS" ] && [ -f "$KNOWN_HOSTS" ]; then
    local dst="$ROOT/logs/pod_$NAME"
    mkdir -p "$dst"
    if af_ssh 45 'tar -C /var/log/afriat -czf - .' > "$dst/var_log_afriat.tgz" 2>/dev/null &&
       tar -xzf "$dst/var_log_afriat.tgz" -C "$dst" 2>/dev/null; then
      say "  pod logs archived to logs/pod_$NAME/"
    fi
    af_ssh 15 'echo laptop-teardown > /run/afriat/delete_now' >/dev/null 2>&1 || true
  fi
  if [ -z "$POD_ID" ] && [ "$CREATE_SENT" = 1 ]; then
    local j
    for j in 1 2 3 4 5 6; do
      if [ "$(rp_get /pods "$T/pods.json")" = 200 ]; then
        POD_ID="$(jq -r --arg n "$NAME" '[(.pods // .)[] | select(.name == $n)][0].id // empty' "$T/pods.json")"
      fi
      [ -n "$POD_ID" ] && break
      sleep 8
    done
  fi
  if [ -n "$POD_ID" ]; then
    local i code
    for i in 1 2 3 4 5 6; do
      code="$(rp DELETE "/pods/$POD_ID" "$T/del.json")"
      detail "DELETE try $i: HTTP $code"
      code="$(rp_get "/pods/$POD_ID" "$T/pod.json")"
      if [ "$code" = 404 ] || [ "$(jq -r '.status // empty' "$T/pod.json" 2>/dev/null)" = TERMINATED ]; then
        GONE=1
        break
      fi
      sleep $((i * 5))
    done
  elif [ "$CREATE_SENT" = 1 ]; then
    GONE=1
  fi

  local left
  left="$(active_pods_count)"
  if [ "$CREATE_SENT" = 1 ]; then
    local now charged status
    now="$(date +%s)"
    charged="$(awk -v p="$PRICE" -v s="$(( now - CREATE_T ))" 'BEGIN {printf "%.2f", int(p * s / 3600 * 100 + 0.999999) / 100}')"
    status=closed
    [ "$GONE" = 1 ] || status=UNCONFIRMED
    [ -f "$LEDGER" ] || printf 'utc\tpod_name\tworst_usd\testimate_usd\tstatus\n' > "$LEDGER"
    printf '%s\t%s\t%s\t%s\t%s\n' "$STAMP" "$NAME" "$WORST" "$charged" "$status" >> "$LEDGER"
    say "  pod wall time: $(( (now - CREATE_T) / 60 )) min $(( (now - CREATE_T) % 60 )) s; estimated cost \$$charged"
  fi
  rm -rf "$T"
  if [ "$GONE" = 1 ] && [ "$left" = 0 ]; then
    say "  ✓ pod deleted; active pods remaining on account: 0"
  else
    say "  !!! WARNING: pod deletion unconfirmed (active pods remaining: $left); check RunPod console!"
    exit 9
  fi
}

trap teardown EXIT
trap 'say "(interrupted)"; exit 130' INT TERM HUP

build_create "$NAME"
say ""
say "Creating pod $NAME (single attempt; never retried)..."
CREATE_SENT=1
CREATE_T="$(date +%s)"
DEADLINE=$(( CREATE_T + LIFETIME_MIN * 60 ))
code="$(rp POST /pods "$T/created.json" "$T/create.json")"
detail "POST /pods: HTTP $code"
case "$code" in
  200|201)
    POD_ID="$(jq -r '.id // empty' "$T/created.json")"
    [ -n "$POD_ID" ] || die 5 "created but missing pod id"
    ;;
  4??)
    CREATE_SENT=0
    say "  create failed (HTTP $code): $(jq -r '.detail // .title // empty' "$T/created.json" 2>/dev/null | tr -cd '[:print:]' | head -c 240)"
    die 5 "not retrying"
    ;;
  *)
    say "  create failed (HTTP $code): $(jq -r '.detail // .title // empty' "$T/created.json" 2>/dev/null | tr -cd '[:print:]' | head -c 240)"
    die 5 "not retrying"
    ;;
esac
detail "pod id $POD_ID"
say "  pod created; waiting for RUNNING state & SSH..."

BOOT_DEADLINE=$(( CREATE_T + 900 ))
past_deadline() { [ "$(date +%s)" -ge "$DEADLINE" ]; }
past_boot_deadline() { [ "$(date +%s)" -ge "$BOOT_DEADLINE" ]; }

last=""
next_hb=$(( CREATE_T + 60 ))
while :; do
  past_boot_deadline && die 6 "boot timeout (540s) waiting for RUNNING + direct TCP port"
  code="$(rp_get "/pods/$POD_ID" "$T/pod.json")"
  if [ "$code" = 200 ]; then
    st="$(jq -r '.status // "?"' "$T/pod.json")"
    dc_cur="$(jq -r '.machine.dataCenterId // .dataCenterId // "?"' "$T/pod.json")"
    has_rt="$(jq -r '.runtime != null' "$T/pod.json")"
    [ "$st" != "$last" ] && say "  status: $st in $dc_cur ($(( $(date +%s) - CREATE_T )) s)" && last="$st"
    case "$st" in EXITED|ERROR|TERMINATED) die 6 "pod entered $st" ;; esac
    hp="$(jq -r '(.ssh.direct // empty | "\(.host) \(.port)"),
                 ((.runtime.ports // [])[] | select(.private == 22) | "\(.ip) \(.public)")' "$T/pod.json" 2>/dev/null | head -1)"
    if [ "$st" = RUNNING ] && [ -n "$hp" ]; then
      SSH_HOST="${hp% *}"
      SSH_PORT="${hp#* }"
      break
    fi
    if [ "$st" = RUNNING ] && [ "$has_rt" = "true" ]; then
      # Container started (.runtime != null), give 16s grace for ssh.direct to populate, else host lacks public IP
      sleep 16
      code2="$(rp_get "/pods/$POD_ID" "$T/pod.json")"
      hp="$(jq -r '(.ssh.direct // empty | "\(.host) \(.port)"),
                   ((.runtime.ports // [])[] | select(.private == 22) | "\(.ip) \(.public)")' "$T/pod.json" 2>/dev/null | head -1)"
      if [ -n "$hp" ]; then
        SSH_HOST="${hp% *}"
        SSH_PORT="${hp#* }"
        break
      fi
      die 6 "host node in $dc_cur started container without a public IP (ssh.direct=null)"
    fi
    if [ "$(date +%s)" -ge "$next_hb" ]; then
      say "  pulling container image on $dc_cur ($(( $(date +%s) - CREATE_T )) s elapsed)..."
      next_hb=$(( $(date +%s) + 60 ))
    fi
  fi
  sleep 8
done

T_RUNNING="$(date +%s)"
[[ "$SSH_PORT" =~ ^[0-9]+$ ]] && [[ "$SSH_HOST" =~ ^[0-9A-Za-z.:-]+$ ]] || die 6 "malformed SSH endpoint"
DC_ID="$(jq -r '.machine.dataCenterId // .dataCenterId // "?"' "$T/pod.json")"
detail "dc=$DC_ID ssh=$SSH_HOST:$SSH_PORT"
say "  placed in data center: $DC_ID (POST->RUNNING: $(( T_RUNNING - CREATE_T )) s)"

fetch_log() {
  printf 'Authorization: Bearer %s\n' "$RP_KEY" | curl -q -sS --proto '=https' --tlsv1.2 \
    --connect-timeout 10 --max-time 10 -o "$T/sse.txt" -H @- -H 'Accept: text/event-stream' \
    "$API/pods/$POD_ID/logs?source=container&tail=2000" 2>/dev/null
  grep '^data:' "$T/sse.txt" 2>/dev/null | sed 's/^data: *//' | jq -r '.line // empty' 2>/dev/null > "$T/log.txt"
  [ -s "$T/log.txt" ] && cp "$T/log.txt" "$BOOTLOG"
  return 0
}
marker() { grep -F "[af-boot] $1" "$BOOTLOG" 2>/dev/null | tail -1; }

HK=""
while :; do
  past_boot_deadline && die 6 "boot timeout (240s) waiting for SSH_READY"
  fetch_log
  f="$(marker "BOOT_FAILED")"
  [ -n "$f" ] && die 6 "pod boot failed: ${f#*BOOT_FAILED }"
  HK="$(marker "SSH_HOST_KEY ssh-ed25519 " | sed 's/.*SSH_HOST_KEY //')"
  [ -n "$HK" ] && [ -n "$(marker "SSH_READY")" ] && break
  sleep 2
done
T_SSH_READY="$(date +%s)"
[[ "$HK" =~ ^ssh-ed25519\ [A-Za-z0-9+/]+=*$ ]] || die 6 "malformed SSH_HOST_KEY"
KNOWN_HOSTS="$ROOT/logs/known_hosts_$NAME"
printf '[%s]:%s %s\n' "$SSH_HOST" "$SSH_PORT" "$HK" > "$KNOWN_HOSTS"
say "  SSH host key pinned ($(printf '%s\n' "$HK" > "$T/hk.pub"; ssh-keygen -lf "$T/hk.pub" | cut -d' ' -f2)) [POST->SSH_READY: $(( T_SSH_READY - CREATE_T )) s, RUNNING->SSH_READY: $(( T_SSH_READY - T_RUNNING )) s]"
sleep 1

say "Shipping hash-verified Project Afriat bundle to /workspace..."
af_ssh 60 "cat > /tmp/afriat_bundle.tgz && \
  got=\$(sha256sum /tmp/afriat_bundle.tgz | cut -d' ' -f1) && \
  [ \"\$got\" = '$BUNDLE_SHA' ] && \
  mkdir -p /workspace && tar -xzf /tmp/afriat_bundle.tgz -C /workspace && \
  chmod +x /workspace/scripts/pod/fetch_and_run_model.sh /workspace/scripts/pod/fetch_and_run_inequity.sh /workspace/scripts/pod/fetch_and_run_valuation.sh /workspace/scripts/pod/fetch_and_run_thought.sh /workspace/scripts/pod/fetch_and_run_showcase.sh /workspace/scripts/pod/fetch_and_run_frontier.sh /workspace/scripts/pod/fetch_and_run_vastness.sh /workspace/scripts/pod/fetch_and_run_funnel.sh /workspace/scripts/pod/fetch_and_run_optimize.sh /workspace/scripts/pod/run_autonomous_exfil.sh && \
  PYTHONPATH=/workspace python3 -m afriat.run_discovery --audit-only >/dev/null && \
  PYTHONPATH=/workspace python3 -m afriat.inequity --audit-only >/dev/null && \
  PYTHONPATH=/workspace python3 -m afriat.valuation --audit-only >/dev/null && \
  PYTHONPATH=/workspace python3 -m afriat.thought_probe --audit-only >/dev/null && \
  PYTHONPATH=/workspace python3 -m afriat.showcase --audit-only >/dev/null && \
  PYTHONPATH=/workspace python3 -m afriat.showcase --suite r2 --audit-only >/dev/null && \
  PYTHONPATH=/workspace python3 -m afriat.frontier --audit-only >/dev/null && \
  PYTHONPATH=/workspace python3 -m afriat.vastness --audit-only >/dev/null && \
  PYTHONPATH=/workspace python3 -m afriat.funnel --audit-only >/dev/null && \
  PYTHONPATH=/workspace python3 -m afriat.optimize --audit-only >/dev/null" < "$BUNDLE_TGZ" ||
  die 6 "failed to ship or audit afriat bundle over SSH"
T_BUNDLE_READY="$(date +%s)"
say "  bundle verified & stimulus firewall passed on pod [POST->BUNDLE_OK: $(( T_BUNDLE_READY - CREATE_T )) s]"

if [ "$SMOKE_BOOT_ONLY" = 1 ]; then
  say "Running GPU + PyTorch + Transformers startup benchmark..."
  PROBE_JSON="$(af_ssh 60 'python3 - <<\"PY\"
import json, time
t0 = time.time()
import torch, transformers
from transformers import Gemma4ForConditionalGeneration
dt = round(time.time() - t0, 3)
print(json.dumps({
  "torch_version": torch.__version__,
  "cuda_version": torch.version.cuda,
  "transformers_version": transformers.__version__,
  "cuda_available": bool(torch.cuda.is_available()),
  "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
  "import_seconds": dt
}))
PY')" || die 7 "smoke-boot GPU Python probe failed"
  T_TORCH_READY="$(date +%s)"
  APT_SKIPPED=false
  grep -Fq "openssh-server pre-installed in image" "$BOOTLOG" 2>/dev/null && APT_SKIPPED=true
  BENCH_OUT="$ROOT/runs/boot_bench_${STAMP}.json"
  jq -n \
    --arg stamp "$STAMP" \
    --arg image "$IMAGE" \
    --arg dc "$DC_ID" \
    --arg gpu_req "$GPU_ID" \
    --argjson apt_skipped "$APT_SKIPPED" \
    --argjson post_to_running_s "$(( T_RUNNING - CREATE_T ))" \
    --argjson running_to_ssh_s "$(( T_SSH_READY - T_RUNNING ))" \
    --argjson post_to_ssh_s "$(( T_SSH_READY - CREATE_T ))" \
    --argjson post_to_bundle_s "$(( T_BUNDLE_READY - CREATE_T ))" \
    --argjson post_to_torch_gpu_ready_s "$(( T_TORCH_READY - CREATE_T ))" \
    --argjson probe "$PROBE_JSON" \
    '{
      timestamp_utc: $stamp,
      image: $image,
      data_center: $dc,
      requested_gpu: $gpu_req,
      apt_openssh_skipped: $apt_skipped,
      timings_seconds: {
        post_to_running_s: $post_to_running_s,
        running_to_ssh_ready_s: $running_to_ssh_s,
        post_to_ssh_ready_s: $post_to_ssh_s,
        post_to_bundle_audit_s: $post_to_bundle_s,
        post_to_torch_gpu_ready_s: $post_to_torch_gpu_ready_s
      },
      gpu_python_probe: $probe
    }' > "$BENCH_OUT"
  say ""
  say "=== Startup Benchmark Result ($BENCH_OUT) ==="
  cat "$BENCH_OUT"
  if [ "$EXFIL_AZURE" = 1 ]; then
    printf 'url = "https://%s.blob.core.windows.net/%s/_benchmarks/boot_bench_%s.json?%s"\n' \
      "$AZ_ACCOUNT" "$AZ_CONTAINER" "$STAMP" "$AZ_SAS" |
      curl -q -sS --proto '=https' --tlsv1.2 --max-time 20 -o /dev/null -X PUT \
        -H 'x-ms-blob-type: BlockBlob' -H 'Content-Type: application/json' \
        --data-binary @"$BENCH_OUT" --config - 2>/dev/null || true
  fi
  exit 0
fi

run_one_model() {
  local m="$1"
  local repo rev out_local
  repo="$(jq -r --arg m "$m" '.models[] | select(.name == $m) | .repo' "$PINS")"
  rev="$(jq -r --arg m "$m" '.models[] | select(.name == $m) | .revision' "$PINS")"
  say ""
  say "=== Running Phase 0 Discovery: $m ($repo) ==="
  af_ssh 1800 "/workspace/scripts/pod/fetch_and_run_model.sh '$m' '$repo' '$rev' '$BATCH_SIZE' '$STEER_ALPHA'" ||
    die 7 "discovery failed on pod for $m"

  out_local="$ROOT/runs/${m}_${STAMP}"
  mkdir -p "$out_local"
  say "Pulling artifacts for $m -> runs/${m}_${STAMP}/"
  af_ssh 120 "tar -C '/workspace/runs/$m' -czf - ." > "$T/${m}_res.tgz" ||
    die 7 "failed to pull results tarball for $m"
  tar -xzf "$T/${m}_res.tgz" -C "$out_local" || die 7 "failed to unpack results tarball for $m"
  [ -s "$out_local/results.json" ] || die 7 "missing results.json after pull for $m"

  say "Rendering Dawn State SVG/PNG/Markdown/HTML reports locally..."
  (cd "$ROOT" && uv run afriat-discover --render-dir "$out_local") ||
    die 7 "local report rendering failed for $m"
}

run_one_inequity() {
  local m="$1"
  local repo rev p0_dir p0_tgz p0_sha out_local
  repo="$(jq -r --arg m "$m" '.models[] | select(.name == $m) | .repo' "$PINS")"
  rev="$(jq -r --arg m "$m" '.models[] | select(.name == $m) | .revision' "$PINS")"
  p0_dir="$(find_phase0_dir "$m")"
  p0_tgz="$T/p0_${m}.tgz"
  COPYFILE_DISABLE=1 tar --no-xattrs -C "$p0_dir" -czf "$p0_tgz" basis_vectors.npz results.json
  p0_sha="$(shasum -a 256 "$p0_tgz" | cut -d' ' -f1)"

  say ""
  say "=== Running Phase 1 Inequity Dissection: $m ($repo) ==="
  say "Shipping locked Phase 0 basis (${p0_dir##*/}) to pod..."
  af_ssh 180 "mkdir -p '/workspace/phase0/$m' && cat > '/tmp/p0_${m}.tgz' && \
    got=\$(sha256sum '/tmp/p0_${m}.tgz' | cut -d' ' -f1) && [ \"\$got\" = '$p0_sha' ] && \
    tar -xzf '/tmp/p0_${m}.tgz' -C '/workspace/phase0/$m' && rm -f '/tmp/p0_${m}.tgz'" < "$p0_tgz" ||
    die 7 "failed to ship Phase 0 basis for $m"

  af_ssh 1800 "/workspace/scripts/pod/fetch_and_run_inequity.sh '$m' '$repo' '$rev' '$BATCH_SIZE'" ||
    die 7 "inequity probe failed on pod for $m"

  out_local="$ROOT/runs/inequity_${m}_${STAMP}"
  mkdir -p "$out_local"
  say "Pulling inequity artifacts for $m -> runs/inequity_${m}_${STAMP}/"
  af_ssh 120 "tar -C '/workspace/runs_inequity/$m' -czf - ." > "$T/${m}_ineq.tgz" ||
    die 7 "failed to pull inequity results for $m"
  tar -xzf "$T/${m}_ineq.tgz" -C "$out_local" || die 7 "failed to unpack inequity results for $m"
  [ -s "$out_local/inequity_results.json" ] || die 7 "missing inequity_results.json after pull for $m"

  say "Rendering Dawn State Inequity SVG/PNG/Markdown/HTML reports locally..."
  (cd "$ROOT" && uv run afriat-inequity --render-dir "$out_local") ||
    die 7 "local inequity report rendering failed for $m"
}

run_one_valuation() {
  local m="$1"
  local repo rev p0_dir p0_tgz p0_sha out_local
  repo="$(jq -r --arg m "$m" '.models[] | select(.name == $m) | .repo' "$PINS")"
  rev="$(jq -r --arg m "$m" '.models[] | select(.name == $m) | .revision' "$PINS")"
  p0_dir="$(find_phase0_dir "$m")"
  p0_tgz="$T/p0_${m}.tgz"
  COPYFILE_DISABLE=1 tar --no-xattrs -C "$p0_dir" -czf "$p0_tgz" basis_vectors.npz results.json
  p0_sha="$(shasum -a 256 "$p0_tgz" | cut -d' ' -f1)"

  say ""
  say "=== Running Phase 2 Valuation & Rationalization: $m ($repo) ==="
  say "Shipping locked Phase 0 basis (${p0_dir##*/}) to pod..."
  af_ssh 180 "mkdir -p '/workspace/phase0/$m' && cat > '/tmp/p0_${m}.tgz' && \
    got=\$(sha256sum '/tmp/p0_${m}.tgz' | cut -d' ' -f1) && [ \"\$got\" = '$p0_sha' ] && \
    tar -xzf '/tmp/p0_${m}.tgz' -C '/workspace/phase0/$m' && rm -f '/tmp/p0_${m}.tgz'" < "$p0_tgz" ||
    die 7 "failed to ship Phase 0 basis for $m"

  af_ssh 2400 "/workspace/scripts/pod/fetch_and_run_valuation.sh '$m' '$repo' '$rev' '$BATCH_SIZE'" ||
    die 7 "valuation probe failed on pod for $m"

  out_local="$ROOT/runs/valuation_${m}_${STAMP}"
  mkdir -p "$out_local"
  say "Pulling valuation artifacts for $m -> runs/valuation_${m}_${STAMP}/"
  af_ssh 120 "tar -C '/workspace/runs_valuation/$m' -czf - ." > "$T/${m}_val.tgz" ||
    die 7 "failed to pull valuation results for $m"
  tar -xzf "$T/${m}_val.tgz" -C "$out_local" || die 7 "failed to unpack valuation results for $m"
  [ -s "$out_local/valuation_results.json" ] || die 7 "missing valuation_results.json after pull for $m"

  say "Rendering Dawn State Valuation SVG/PNG/Markdown/HTML reports locally..."
  (cd "$ROOT" && uv run afriat-valuation --render-dir "$out_local") ||
    die 7 "local valuation report rendering failed for $m"
}

run_one_thought() {
  local m="$1"
  local repo rev p0_dir p0_tgz p0_sha out_local
  repo="$(jq -r --arg m "$m" '.models[] | select(.name == $m) | .repo' "$PINS")"
  rev="$(jq -r --arg m "$m" '.models[] | select(.name == $m) | .revision' "$PINS")"
  p0_dir="$(find_phase0_dir "$m")"
  p0_tgz="$T/p0_${m}.tgz"
  COPYFILE_DISABLE=1 tar --no-xattrs -C "$p0_dir" -czf "$p0_tgz" basis_vectors.npz results.json
  p0_sha="$(shasum -a 256 "$p0_tgz" | cut -d' ' -f1)"

  say ""
  say "=== Running Gemma 4 <thought> Scratchpad & Moral Highlighter: $m ($repo) ==="
  say "Shipping locked Phase 0 basis (${p0_dir##*/}) to pod..."
  af_ssh 180 "mkdir -p '/workspace/phase0/$m' && cat > '/tmp/p0_${m}.tgz' && \
    got=\$(sha256sum '/tmp/p0_${m}.tgz' | cut -d' ' -f1) && [ \"\$got\" = '$p0_sha' ] && \
    tar -xzf '/tmp/p0_${m}.tgz' -C '/workspace/phase0/$m' && rm -f '/tmp/p0_${m}.tgz'" < "$p0_tgz" ||
    die 7 "failed to ship Phase 0 basis for $m"

  af_ssh 2400 "/workspace/scripts/pod/fetch_and_run_thought.sh '$m' '$repo' '$rev' '$BATCH_SIZE'" ||
    die 7 "thought probe failed on pod for $m"

  out_local="$ROOT/runs/thought_${m}_${STAMP}"
  mkdir -p "$out_local"
  say "Pulling thought artifacts for $m -> runs/thought_${m}_${STAMP}/"
  af_ssh 120 "tar -C '/workspace/runs_thought/$m' -czf - ." > "$T/${m}_thought.tgz" ||
    die 7 "failed to pull thought results for $m"
  tar -xzf "$T/${m}_thought.tgz" -C "$out_local" || die 7 "failed to unpack thought results for $m"
  [ -s "$out_local/thought_results.json" ] || die 7 "missing thought_results.json after pull for $m"

  say "Rendering Warm Precision <thought> PNG/HTML/Markdown reports locally..."
  (cd "$ROOT" && uv run python -m afriat.thought_probe --render-dir "$out_local") ||
    die 7 "local thought report rendering failed for $m"
}

run_one_showcase() {
  local m="$1"
  local suite="${2:-r1}"
  local repo rev p0_dir p0_tgz p0_sha out_local tag
  tag="showcase"
  [ "$suite" = "r2" ] && tag="showcase_r2"
  repo="$(jq -r --arg m "$m" '.models[] | select(.name == $m) | .repo' "$PINS")"
  rev="$(jq -r --arg m "$m" '.models[] | select(.name == $m) | .revision' "$PINS")"
  p0_dir="$(find_phase0_dir "$m")"
  p0_tgz="$T/p0_${m}.tgz"
  COPYFILE_DISABLE=1 tar --no-xattrs -C "$p0_dir" -czf "$p0_tgz" basis_vectors.npz results.json
  p0_sha="$(shasum -a 256 "$p0_tgz" | cut -d' ' -f1)"

  say ""
  say "=== Running Non-Financial Behavioral Steering Showcase ($suite): $m ($repo) ==="
  say "Shipping locked Phase 0 basis (${p0_dir##*/}) to pod..."
  af_ssh 180 "mkdir -p '/workspace/phase0/$m' && cat > '/tmp/p0_${m}.tgz' && \
    got=\$(sha256sum '/tmp/p0_${m}.tgz' | cut -d' ' -f1) && [ \"\$got\" = '$p0_sha' ] && \
    tar -xzf '/tmp/p0_${m}.tgz' -C '/workspace/phase0/$m' && rm -f '/tmp/p0_${m}.tgz'" < "$p0_tgz" ||
    die 7 "failed to ship Phase 0 basis for $m"

  af_ssh 3600 "/workspace/scripts/pod/fetch_and_run_showcase.sh '$m' '$repo' '$rev' '$BATCH_SIZE' '$suite'" ||
    die 7 "showcase ($suite) probe failed on pod for $m"

  out_local="$ROOT/runs/${tag}_${m}_${STAMP}"
  mkdir -p "$out_local"
  say "Pulling showcase ($suite) artifacts for $m -> runs/${tag}_${m}_${STAMP}/"
  af_ssh 120 "tar -C '/workspace/runs_showcase/$m' -czf - ." > "$T/${m}_${tag}.tgz" ||
    die 7 "failed to pull showcase results for $m"
  tar -xzf "$T/${m}_${tag}.tgz" -C "$out_local" || die 7 "failed to unpack showcase results for $m"
  [ -s "$out_local/showcase_results.json" ] || die 7 "missing showcase_results.json after pull for $m"

  say "Rendering Warm Precision Showcase PNG/HTML/Markdown reports locally..."
  (cd "$ROOT" && uv run python -m afriat.showcase --render-dir "$out_local") ||
    die 7 "local showcase report rendering failed for $m"
}

if [ "$TASK_MODE" = inequity ]; then
  for m in "${MODEL_LIST[@]}"; do
    run_one_inequity "$m"
  done
elif [ "$TASK_MODE" = valuation ]; then
  for m in "${MODEL_LIST[@]}"; do
    run_one_valuation "$m"
  done
elif [ "$TASK_MODE" = thought ]; then
  for m in "${MODEL_LIST[@]}"; do
    run_one_thought "$m"
  done
elif [ "$TASK_MODE" = showcase ]; then
  for m in "${MODEL_LIST[@]}"; do
    run_one_showcase "$m" "r1"
  done
elif [ "$TASK_MODE" = showcase_r2 ]; then
  for m in "${MODEL_LIST[@]}"; do
    run_one_showcase "$m" "r2"
  done
elif [ "$TASK_MODE" = frontier ]; then
  for m in "${MODEL_LIST[@]}"; do
    repo="$(jq -r --arg m "$m" '.models[] | select(.name == $m) | .repo' "$PINS")"
    rev="$(jq -r --arg m "$m" '.models[] | select(.name == $m) | .revision' "$PINS")"
    p0_dir="$(find_phase0_dir "$m")"
    p0_tgz="$T/p0_${m}.tgz"
    COPYFILE_DISABLE=1 tar --no-xattrs -C "$p0_dir" -czf "$p0_tgz" basis_vectors.npz results.json
    p0_sha="$(shasum -a 256 "$p0_tgz" | cut -d' ' -f1)"

    say ""
    say "=== Running Phase 5 Frontier Suite (Prospect + Multi-Turn Retaliation): $m ($repo) ==="
    say "Shipping locked Phase 0 basis (${p0_dir##*/}) to pod..."
    af_ssh 180 "mkdir -p '/workspace/phase0/$m' && cat > '/tmp/p0_${m}.tgz' && \
      got=\$(sha256sum '/tmp/p0_${m}.tgz' | cut -d' ' -f1) && [ \"\$got\" = '$p0_sha' ] && \
      tar -xzf '/tmp/p0_${m}.tgz' -C '/workspace/phase0/$m' && rm -f '/tmp/p0_${m}.tgz'" < "$p0_tgz" ||
      die 7 "failed to ship Phase 0 basis for $m"

    af_ssh 3600 "/workspace/scripts/pod/fetch_and_run_frontier.sh '$m' '$repo' '$rev'" ||
      die 7 "frontier probe failed on pod for $m"

    out_local="$ROOT/runs/frontier_${m}_${STAMP}"
    mkdir -p "$out_local"
    say "Pulling frontier artifacts for $m -> runs/frontier_${m}_${STAMP}/"
    af_ssh 120 "tar -C '/workspace/runs_frontier/$m' -czf - ." > "$T/${m}_frontier.tgz" ||
      die 7 "failed to pull frontier results for $m"
    tar -xzf "$T/${m}_frontier.tgz" -C "$out_local" || die 7 "failed to unpack frontier results for $m"
    [ -s "$out_local/frontier_results.json" ] || die 7 "missing frontier_results.json after pull for $m"

    say "Rendering Warm Precision Frontier PNG/HTML/Markdown reports locally..."
    (cd "$ROOT" && uv run python -m afriat.frontier --render-dir "$out_local") ||
      die 7 "local frontier report rendering failed for $m"
  done
elif [ "$TASK_MODE" = vastness ]; then
  for m in "${MODEL_LIST[@]}"; do
    repo="$(jq -r --arg m "$m" '.models[] | select(.name == $m) | .repo' "$PINS")"
    rev="$(jq -r --arg m "$m" '.models[] | select(.name == $m) | .revision' "$PINS")"
    p0_dir="$(find_phase0_dir "$m")"
    p0_tgz="$T/p0_${m}.tgz"
    COPYFILE_DISABLE=1 tar --no-xattrs -C "$p0_dir" -czf "$p0_tgz" basis_vectors.npz results.json
    p0_sha="$(shasum -a 256 "$p0_tgz" | cut -d' ' -f1)"

    say ""
    say "=== Running Phase 6 Vastness Dissection (Pure Vastness vs. Tiny Stakes): $m ($repo) ==="
    say "Shipping locked Phase 0 basis (${p0_dir##*/}) to pod..."
    af_ssh 180 "mkdir -p '/workspace/phase0/$m' && cat > '/tmp/p0_${m}.tgz' && \
      got=\$(sha256sum '/tmp/p0_${m}.tgz' | cut -d' ' -f1) && [ \"\$got\" = '$p0_sha' ] && \
      tar -xzf '/tmp/p0_${m}.tgz' -C '/workspace/phase0/$m' && rm -f '/tmp/p0_${m}.tgz'" < "$p0_tgz" ||
      die 7 "failed to ship Phase 0 basis for $m"

    af_ssh 3600 "/workspace/scripts/pod/fetch_and_run_vastness.sh '$m' '$repo' '$rev'" ||
      die 7 "vastness probe failed on pod for $m"

    out_local="$ROOT/runs/vastness_${m}_${STAMP}"
    mkdir -p "$out_local"
    say "Pulling vastness artifacts for $m -> runs/vastness_${m}_${STAMP}/"
    af_ssh 120 "tar -C '/workspace/runs_vastness/$m' -czf - ." > "$T/${m}_vastness.tgz" ||
      die 7 "failed to pull vastness results for $m"
    tar -xzf "$T/${m}_vastness.tgz" -C "$out_local" || die 7 "failed to unpack vastness results for $m"
    [ -s "$out_local/vastness_results.json" ] || die 7 "missing vastness_results.json after pull for $m"

    say "Rendering Warm Precision Vastness Dissection PNG/HTML/Markdown reports locally..."
    (cd "$ROOT" && uv run python -m afriat.vastness --render-dir "$out_local") ||
      die 7 "local vastness report rendering failed for $m"
  done
elif [ "$TASK_MODE" = funnel ]; then
  for m in "${MODEL_LIST[@]}"; do
    repo="$(jq -r --arg m "$m" '.models[] | select(.name == $m) | .repo' "$PINS")"
    rev="$(jq -r --arg m "$m" '.models[] | select(.name == $m) | .revision' "$PINS")"
    p0_dir="$(find_phase0_dir "$m")"
    p0_tgz="$T/p0_${m}.tgz"
    COPYFILE_DISABLE=1 tar --no-xattrs -C "$p0_dir" -czf "$p0_tgz" basis_vectors.npz results.json
    p0_sha="$(shasum -a 256 "$p0_tgz" | cut -d' ' -f1)"

    say ""
    say "=== Running Phase 7 Read/Write Funnel Dissection (Goodfire vs. Residual Bus): $m ($repo) ==="
    say "Shipping locked Phase 0 basis (${p0_dir##*/}) to pod..."
    af_ssh 180 "mkdir -p '/workspace/phase0/$m' && cat > '/tmp/p0_${m}.tgz' && \
      got=\$(sha256sum '/tmp/p0_${m}.tgz' | cut -d' ' -f1) && [ \"\$got\" = '$p0_sha' ] && \
      tar -xzf '/tmp/p0_${m}.tgz' -C '/workspace/phase0/$m' && rm -f '/tmp/p0_${m}.tgz'" < "$p0_tgz" ||
      die 7 "failed to ship Phase 0 basis for $m"

    if [ -n "$RESUME_FUNNEL" ]; then
      [ -s "$RESUME_FUNNEL" ] || die 2 "--resume-funnel file not found: $RESUME_FUNNEL"
      say "Shipping verified Blocks 0..4 checkpoint ($RESUME_FUNNEL) to resume straight at Block 5..."
      af_ssh 60 "cat > '/workspace/phase0/$m/resume_funnel_results.json'" < "$RESUME_FUNNEL" ||
        die 7 "failed to ship resume_funnel_results.json"
    fi

    run_prefix="runs/funnel_${m}_${STAMP}"
    if [ "$EXFIL_AZURE" = 1 ]; then
      ship_exfil_env "$run_prefix" "funnel" "$m" "/workspace/runs_funnel/$m" "funnel_results.json" ||
        die 7 "failed to ship Azure exfil env to pod"
    fi

    if [ "$DETACH" = 1 ]; then
      say "Starting autonomous background runner on pod (verified against Azure Blob)..."
      af_ssh 30 "nohup /workspace/scripts/pod/run_autonomous_exfil.sh /workspace/scripts/pod/fetch_and_run_funnel.sh '$m' '$repo' '$rev' '$BATCH_SIZE' >/var/log/afriat/autonomous_nohup.log 2>&1 </dev/null & sleep 1 && pgrep -f run_autonomous_exfil.sh >/dev/null" ||
        die 7 "failed to start autonomous background runner on pod"
      # Verify pod actually wrote initial status.json to Azure Blob before detaching!
      verified_az=0
      for _try in 1 2 3 4 5; do
        if [ "$(az storage blob exists --account-name "$AZ_ACCOUNT" --container-name "$AZ_CONTAINER" --name "$run_prefix/status.json" --auth-mode login --query exists --output tsv 2>/dev/null)" = "true" ]; then
          verified_az=1
          break
        fi
        sleep 2
      done
      [ "$verified_az" = 1 ] || die 7 "pod started but initial status.json did not appear in Azure Blob ($AZ_ACCOUNT/$AZ_CONTAINER/$run_prefix); aborting & tearing down!"
      say "  ✓ Verified live pod->Azure Blob handshake ($run_prefix/status.json)!"
      DETACHED_PREFIX="$run_prefix"
      DETACHED_HANDOFF=1
      exit 0
    fi

    if [ "$EXFIL_AZURE" = 1 ]; then
      af_ssh 3600 "/workspace/scripts/pod/run_autonomous_exfil.sh /workspace/scripts/pod/fetch_and_run_funnel.sh '$m' '$repo' '$rev' '$BATCH_SIZE'" ||
        die 7 "funnel probe failed on pod for $m"
      (cd "$ROOT" && ./scripts/sync_afriat_runs.sh --prefix "$run_prefix") ||
        die 7 "failed to sync/render $run_prefix from Azure Blob"
    else
      af_ssh 3600 "/workspace/scripts/pod/fetch_and_run_funnel.sh '$m' '$repo' '$rev' '$BATCH_SIZE'" ||
        die 7 "funnel probe failed on pod for $m"

      out_local="$ROOT/runs/funnel_${m}_${STAMP}"
      mkdir -p "$out_local"
      say "Pulling funnel artifacts for $m -> runs/funnel_${m}_${STAMP}/"
      af_ssh 120 "tar -C '/workspace/runs_funnel/$m' -czf - ." > "$T/${m}_funnel.tgz" ||
        die 7 "failed to pull funnel results for $m"
      tar -xzf "$T/${m}_funnel.tgz" -C "$out_local" || die 7 "failed to unpack funnel results for $m"
      [ -s "$out_local/funnel_results.json" ] || die 7 "missing funnel_results.json after pull for $m"

      say "Rendering Dawn State Funnel Dissection PNG/SVG/HTML/Markdown reports locally..."
      (cd "$ROOT" && uv run python -m afriat.funnel --render-dir "$out_local") ||
        die 7 "local funnel report rendering failed for $m"
    fi
  done
elif [ "$TASK_MODE" = optimize ]; then
  for m in "${MODEL_LIST[@]}"; do
    repo="$(jq -r --arg m "$m" '.models[] | select(.name == $m) | .repo' "$PINS")"
    rev="$(jq -r --arg m "$m" '.models[] | select(.name == $m) | .revision' "$PINS")"

    say ""
    say "=== Running Phase 8 Funnel-Guided Optimization (Hourglass Sparsity + Procrustes Looping): $m ($repo) ==="

    run_prefix="runs/optimize_${m}_${STAMP}"
    if [ "$EXFIL_AZURE" = 1 ]; then
      ship_exfil_env "$run_prefix" "optimize" "$m" "/workspace/runs_optimize/$m" "optimize_results.json" ||
        die 7 "failed to ship Azure exfil env to pod"
    fi

    if [ "$DETACH" = 1 ]; then
      say "Starting autonomous background runner on pod (verified against Azure Blob)..."
      af_ssh 30 "nohup /workspace/scripts/pod/run_autonomous_exfil.sh /workspace/scripts/pod/fetch_and_run_optimize.sh '$m' '$repo' '$rev' '$BATCH_SIZE' >/var/log/afriat/autonomous_nohup.log 2>&1 </dev/null & sleep 1 && pgrep -f run_autonomous_exfil.sh >/dev/null" ||
        die 7 "failed to start autonomous background runner on pod"
      verified_az=0
      for _try in 1 2 3 4 5; do
        if [ "$(az storage blob exists --account-name "$AZ_ACCOUNT" --container-name "$AZ_CONTAINER" --name "$run_prefix/status.json" --auth-mode login --query exists --output tsv 2>/dev/null)" = "true" ]; then
          verified_az=1
          break
        fi
        sleep 2
      done
      [ "$verified_az" = 1 ] || die 7 "pod started but initial status.json did not appear in Azure Blob ($AZ_ACCOUNT/$AZ_CONTAINER/$run_prefix); aborting & tearing down!"
      say "  ✓ Verified live pod->Azure Blob handshake ($run_prefix/status.json)!"
      DETACHED_PREFIX="$run_prefix"
      DETACHED_HANDOFF=1
      exit 0
    fi

    if [ "$EXFIL_AZURE" = 1 ]; then
      af_ssh 5400 "/workspace/scripts/pod/run_autonomous_exfil.sh /workspace/scripts/pod/fetch_and_run_optimize.sh '$m' '$repo' '$rev' '$BATCH_SIZE'" ||
        die 7 "optimize probe failed on pod for $m"
      (cd "$ROOT" && ./scripts/sync_afriat_runs.sh --prefix "$run_prefix") ||
        die 7 "failed to sync/render $run_prefix from Azure Blob"
    else
      af_ssh 5400 "/workspace/scripts/pod/fetch_and_run_optimize.sh '$m' '$repo' '$rev' '$BATCH_SIZE'" ||
        die 7 "optimize probe failed on pod for $m"

      out_local="$ROOT/runs/optimize_${m}_${STAMP}"
      mkdir -p "$out_local"
      say "Pulling optimize artifacts for $m -> runs/optimize_${m}_${STAMP}/"
      af_ssh 120 "tar -C '/workspace/runs_optimize/$m' -czf - ." > "$T/${m}_optimize.tgz" ||
        die 7 "failed to pull optimize results for $m"
      tar -xzf "$T/${m}_optimize.tgz" -C "$out_local" || die 7 "failed to unpack optimize results for $m"
      [ -s "$out_local/optimize_results.json" ] || die 7 "missing optimize_results.json after pull for $m"

      say "Rendering Dawn State Optimization PNG/SVG/HTML/Markdown reports locally..."
      (cd "$ROOT" && uv run python -m afriat.optimize --render-dir "$out_local") ||
        die 7 "local optimize report rendering failed for $m"
    fi
  done
elif [ "$TASK_MODE" = loop_compact ]; then
  for m in "${MODEL_LIST[@]}"; do
    repo="$(jq -r --arg m "$m" '.models[] | select(.name == $m) | .repo' "$PINS")"
    rev="$(jq -r --arg m "$m" '.models[] | select(.name == $m) | .revision' "$PINS")"

    say ""
    say "=== Running Phase 9 Gemma-4-Loop-Compact (Actions 1-4: Smaller, Looped & Stronger): $m ($repo) ==="

    run_prefix="runs/loop_compact_${m}_${STAMP}"
    if [ "$EXFIL_AZURE" = 1 ]; then
      ship_exfil_env "$run_prefix" "loop_compact" "$m" "/workspace/runs_loop_compact/$m" "loop_compact_results.json" ||
        die 7 "failed to ship Azure exfil env to pod"
    fi

    if [ "$DETACH" = 1 ]; then
      say "Starting autonomous background runner on pod (verified against Azure Blob)..."
      af_ssh 30 "nohup /workspace/scripts/pod/run_autonomous_exfil.sh /workspace/scripts/pod/fetch_and_run_loop_compact.sh '$m' '$repo' '$rev' '$BATCH_SIZE' >/var/log/afriat/autonomous_nohup.log 2>&1 </dev/null & sleep 1 && pgrep -f run_autonomous_exfil.sh >/dev/null" ||
        die 7 "failed to start autonomous background runner on pod"
      verified_az=0
      for _try in 1 2 3 4 5; do
        if [ "$(az storage blob exists --account-name "$AZ_ACCOUNT" --container-name "$AZ_CONTAINER" --name "$run_prefix/status.json" --auth-mode login --query exists --output tsv 2>/dev/null)" = "true" ]; then
          verified_az=1
          break
        fi
        sleep 2
      done
      [ "$verified_az" = 1 ] || die 7 "pod started but initial status.json did not appear in Azure Blob ($AZ_ACCOUNT/$AZ_CONTAINER/$run_prefix); aborting & tearing down!"
      say "  ✓ Verified live pod->Azure Blob handshake ($run_prefix/status.json)!"
      DETACHED_PREFIX="$run_prefix"
      DETACHED_HANDOFF=1
      exit 0
    fi

    if [ "$EXFIL_AZURE" = 1 ]; then
      af_ssh 5400 "/workspace/scripts/pod/run_autonomous_exfil.sh /workspace/scripts/pod/fetch_and_run_loop_compact.sh '$m' '$repo' '$rev' '$BATCH_SIZE'" ||
        die 7 "loop_compact probe failed on pod for $m"
      (cd "$ROOT" && ./scripts/sync_afriat_runs.sh --prefix "$run_prefix") ||
        die 7 "failed to sync/render $run_prefix from Azure Blob"
    else
      af_ssh 5400 "/workspace/scripts/pod/fetch_and_run_loop_compact.sh '$m' '$repo' '$rev' '$BATCH_SIZE'" ||
        die 7 "loop_compact probe failed on pod for $m"

      out_local="$ROOT/runs/loop_compact_${m}_${STAMP}"
      mkdir -p "$out_local"
      say "Pulling loop_compact artifacts for $m -> runs/loop_compact_${m}_${STAMP}/"
      af_ssh 120 "tar -C '/workspace/runs_loop_compact/$m' -czf - ." > "$T/${m}_loop_compact.tgz" ||
        die 7 "failed to pull loop_compact results for $m"
      tar -xzf "$T/${m}_loop_compact.tgz" -C "$out_local" || die 7 "failed to unpack loop_compact results for $m"
      [ -s "$out_local/loop_compact_results.json" ] || die 7 "missing loop_compact_results.json after pull for $m"

      say "Rendering Dawn State Gemma-4-Loop-Compact PNG/SVG/HTML/Markdown reports locally..."
      (cd "$ROOT" && uv run python -m afriat.loop_compact --render-dir "$out_local") ||
        die 7 "local loop_compact report rendering failed for $m"
    fi
  done
else
  ANY_GATE_PASSED=0
  for m in "${MODEL_LIST[@]}"; do
    run_one_model "$m"
    passed="$(jq -r '.sanity_gate.passed_gate' "$ROOT/runs/${m}_${STAMP}/results.json")"
    [ "$passed" = "true" ] && ANY_GATE_PASSED=1
  done

  if [ "$ANY_GATE_PASSED" = 0 ] && [ -n "$ESCALATE_MODEL" ]; then
    say ""
    say "Both scout models scored below the 85% sanity gate; auto-escalating to $ESCALATE_MODEL..."
    run_one_model "$ESCALATE_MODEL"
  fi
fi

say ""
say "All requested $TASK_MODE runs completed; triggering immediate pod teardown."
exit 0
