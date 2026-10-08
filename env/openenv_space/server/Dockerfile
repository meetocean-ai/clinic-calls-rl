# OpenEnv layout expects server/Dockerfile. The real image is built from the repository root by
# tests/behavioral/env/Dockerfile (it needs services/agent, tests/behavioral and infra/model_profiles.yaml):
#
#   docker build -f tests/behavioral/env/Dockerfile -t ocean-ehr-env-server .
#
# This file only re-tags it for `openenv build` / `openenv push --registry`; the sandbox (Medplum) comes from
# docker-compose-env.yml next to it.
FROM ocean-ehr-env-server:latest
EXPOSE 8011
