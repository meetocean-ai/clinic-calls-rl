#!/bin/sh
# Env-server container entrypoint (roadmap P1-03b). Two ways in:
#   1. MEDPLUM_CLIENT_ID / MEDPLUM_CLIENT_SECRET already set (an existing project) → serve.
#   2. Nothing set → bootstrap the local Medplum at MEDPLUM_BASE_URL (idempotent: finds or creates the env project and
#      client, like scripts/medplum_local_bootstrap.py on the Mac), export its variables, then serve.
# The bootstrap refuses anything that is not a local address; inside the compose network the server is reached as
# http://medplum-server:8103/, so MEDPLUM_BOOTSTRAP_LOCAL_OK=1 lets that hostname through (the stack is private).
set -eu

BASE="${MEDPLUM_BASE_URL:-http://medplum-server:8103/}"
if [ -z "${MEDPLUM_CLIENT_ID:-}" ] || [ -z "${MEDPLUM_CLIENT_SECRET:-}" ]; then
  echo "env-server: waiting for Medplum at $BASE"
  i=0
  until python -c "import httpx,sys; sys.exit(0 if httpx.get('${BASE%/}/healthcheck', timeout=3).status_code == 200 else 1)" 2>/dev/null; do
    i=$((i+1)); [ "$i" -gt 120 ] && { echo "env-server: Medplum never became healthy"; exit 1; }
    sleep 2
  done
  echo "env-server: bootstrapping the environment project on $BASE"
  MEDPLUM_BOOTSTRAP_LOCAL_OK=1 python /app/scripts/medplum_local_bootstrap.py --base-url "$BASE" --out /tmp/medplum.env --env-only
  set -a; . /tmp/medplum.env; set +a
fi
export MEDPLUM_BASE_URL="${MEDPLUM_BASE_URL%/}"
echo "env-server: serving on ${ENV_HOST:-0.0.0.0}:${ENV_PORT:-8011} against $MEDPLUM_BASE_URL"
exec python -m env.server
