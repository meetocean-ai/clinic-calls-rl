"""The scheduling call as an RL environment over the local Medplum (docs/rl-training-experiment.md, E3–E5).

Same shape as the assistant repo's `envs/scheduling` (seeded tasks, tiers, chaos, a binary state
reward, oracle/random brackets, trajectories with manifests) so one scorecard reads both. The world
here is a real FHIR server: a per-episode Organization on the local Medplum, seeded from the task.
"""
