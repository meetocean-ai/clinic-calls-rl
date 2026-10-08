# The RL environment server as a container (roadmap P1-03b): the agent's tool code + the behavioral env package,
# serving the OpenEnv contract on :8011 against a Medplum reachable on the compose network. Built from the repo
# root (it needs services/agent and tests/behavioral):
#
#   docker build -f tests/behavioral/env/Dockerfile -t ocean-ehr-env-server .
#   docker compose -f docker-compose-env.yml up -d        # Medplum + bootstrap + this server
FROM python:3.13-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY services/agent/pyproject.toml ./services/agent/pyproject.toml
COPY tests/behavioral/pyproject.toml ./tests/behavioral/pyproject.toml
RUN uv venv /app/.venv \
    && uv pip install --python /app/.venv/bin/python -r services/agent/pyproject.toml \
    && uv pip install --python /app/.venv/bin/python "openenv-core[core]>=0.3.0" faker pytest pytest-asyncio python-dotenv

FROM python:3.13-slim
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
COPY services/agent ./services/agent
COPY tests/behavioral ./tests/behavioral
COPY scripts/medplum_local_bootstrap.py ./scripts/medplum_local_bootstrap.py
# every LLM/STT/TTS choice in the agent code flows through the model profiles (models/profiles.py reads ../../infra)
COPY infra/model_profiles.yaml ./infra/model_profiles.yaml
RUN rm -rf tests/behavioral/.venv-verifiers tests/behavioral/results/* tests/behavioral/.pytest_cache \
    && chmod +x tests/behavioral/env/docker-entrypoint.sh
ENV PYTHONPATH=/app/services/agent:/app/tests/behavioral \
    PYTHONUNBUFFERED=1 \
    ENV_HOST=0.0.0.0 \
    ENV_PORT=8011 \
    MAX_CONCURRENT_ENVS=16
EXPOSE 8011
WORKDIR /app/tests/behavioral
ENTRYPOINT ["/app/tests/behavioral/env/docker-entrypoint.sh"]
