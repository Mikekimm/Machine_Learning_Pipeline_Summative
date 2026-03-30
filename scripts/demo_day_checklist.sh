#!/usr/bin/env bash
set -euo pipefail

API_URL="${API_URL:-http://127.0.0.1:8000}"
SAMPLE_IMAGE="${1:-}"
UPLOAD_DIR="${UPLOAD_DIR:-}"
UPLOAD_CLASS="${UPLOAD_CLASS:-buildings}"
RETRAIN_EPOCHS="${RETRAIN_EPOCHS:-2}"
RETRAIN_LR="${RETRAIN_LR:-0.00001}"

step() {
  echo
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

step "1) Health Check"
curl -sS "$API_URL/health" | sed 's/.*/&\n/'

step "2) Model Info"
curl -sS "$API_URL/model/info" | sed 's/.*/&\n/'

if [[ -n "$SAMPLE_IMAGE" && -f "$SAMPLE_IMAGE" ]]; then
  step "3) Single Prediction"
  curl -sS -X POST "$API_URL/predict" -F "file=@$SAMPLE_IMAGE" | sed 's/.*/&\n/'
else
  step "3) Single Prediction (Skipped)"
  echo "Pass an image path as argument 1 to run this step."
  echo "Example: scripts/demo_day_checklist.sh data/test/forest/example.jpg"
fi

if [[ -n "$UPLOAD_DIR" && -d "$UPLOAD_DIR" ]]; then
  step "4) Upload New Data"
  mapfile -t upload_files < <(find "$UPLOAD_DIR" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | head -n 10)

  if [[ ${#upload_files[@]} -eq 0 ]]; then
    echo "No images found in UPLOAD_DIR=$UPLOAD_DIR"
  else
    curl_cmd=(curl -sS -X POST "$API_URL/upload?class_name=$UPLOAD_CLASS")
    for f in "${upload_files[@]}"; do
      curl_cmd+=( -F "files=@$f" )
    done
    "${curl_cmd[@]}" | sed 's/.*/&\n/'
  fi
else
  step "4) Upload New Data (Skipped)"
  echo "Set UPLOAD_DIR to a folder with images to run this step."
  echo "Example: UPLOAD_DIR=data/uploads/session_yyyymmdd_hhmmss/buildings scripts/demo_day_checklist.sh <sample_image>"
fi

step "5) Trigger Retraining"
curl -sS -X POST "$API_URL/retrain/trigger" \
  -H "Content-Type: application/json" \
  -d "{\"epochs\": $RETRAIN_EPOCHS, \"learning_rate\": $RETRAIN_LR, \"reason\": \"demo_day\"}" | sed 's/.*/&\n/'

step "6) Poll Retrain Status"
for i in {1..20}; do
  status_json="$(curl -sS "$API_URL/retrain/status")"
  echo "[$i/20] $status_json"

  if echo "$status_json" | grep -qi '"status"[[:space:]]*:[[:space:]]*"completed"'; then
    echo "Retraining completed."
    break
  fi

  if echo "$status_json" | grep -qi '"status"[[:space:]]*:[[:space:]]*"failed"'; then
    echo "Retraining failed."
    break
  fi

  sleep 5
done

step "Done"
echo "Demo checklist run complete."
