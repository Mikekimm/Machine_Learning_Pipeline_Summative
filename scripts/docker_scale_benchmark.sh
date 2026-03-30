#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULT_DIR="$ROOT_DIR/locust_tests/results"
mkdir -p "$RESULT_DIR"

USERS="${USERS:-100}"
SPAWN_RATE="${SPAWN_RATE:-10}"
RUN_TIME="${RUN_TIME:-60s}"
HOST_URL="${HOST_URL:-http://127.0.0.1}"
SCALES=(1 2 3)

LOCUST_BIN="$ROOT_DIR/.venv312/bin/locust"
if [[ ! -x "$LOCUST_BIN" ]]; then
  LOCUST_BIN="locust"
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: Docker is not installed. Install Docker Desktop first."
  exit 1
fi

echo "Running Docker scale benchmark"
echo "Users=$USERS SpawnRate=$SPAWN_RATE RunTime=$RUN_TIME Host=$HOST_URL"

SUMMARY_CSV="$RESULT_DIR/docker_scale_summary.csv"
printf "containers,users,duration,total_requests,error_rate,avg_latency_ms,p95_ms,rps\n" > "$SUMMARY_CSV"

wait_for_api() {
  local retries=40
  local sleep_s=3
  for _ in $(seq 1 "$retries"); do
    if curl -fsS "$HOST_URL/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$sleep_s"
  done
  return 1
}

parse_run() {
  local file="$1"
  local containers="$2"
  awk -v c="$containers" -v users="$USERS" -v dur="$RUN_TIME" '
    BEGIN{req=0;fail=0;rps=0;wavg=0;p95=0}
    /^[A-Z]+[[:space:]]+[0-9]+[[:space:]]+[0-9]+/ {
      req += $2
      fail += $3
      wavg += $2 * $4
      if ($5 > p95) p95 = $5
      rps += $6
    }
    END {
      avg = (req ? wavg / req : 0)
      err = (req ? (fail * 100.0 / req) : 0)
      printf "%s,%s,%s,%d,%.2f%%,%.1f,%.1f,%.2f\n", c, users, dur, req, err, avg, p95, rps
    }
  ' "$file" >> "$SUMMARY_CSV"
}

for scale in "${SCALES[@]}"; do
  echo
  echo "=== Scale: api=$scale ==="

  (
    cd "$ROOT_DIR/docker"
    docker compose down --remove-orphans >/dev/null 2>&1 || true
    docker compose up -d --build --scale "api=$scale" api nginx
  )

  if ! wait_for_api; then
    echo "ERROR: API did not become healthy for scale=$scale"
    exit 1
  fi

  out_file="$RESULT_DIR/locust_docker_${scale}api_${USERS}u_${RUN_TIME}.txt"
  "$LOCUST_BIN" \
    -f "$ROOT_DIR/locust_tests/locustfile.py" \
    --host="$HOST_URL" \
    --users "$USERS" \
    --spawn-rate "$SPAWN_RATE" \
    --run-time "$RUN_TIME" \
    --headless | tee "$out_file"

  parse_run "$out_file" "$scale"
done

echo
echo "=== Docker Scale Summary CSV ==="
cat "$SUMMARY_CSV"

echo
echo "=== Markdown Table (paste into README) ==="
echo "| Containers | Users | Duration | Total Requests | Error Rate | Avg Latency | P95 Latency | RPS |"
echo "|------------|-------|----------|----------------|------------|-------------|-------------|-----|"
awk -F',' 'NR>1{printf "| %s | %s | %s | %s | %s | %sms | %sms | %s |\n", $1,$2,$3,$4,$5,$6,$7,$8}' "$SUMMARY_CSV"

echo
echo "Saved outputs under: $RESULT_DIR"
