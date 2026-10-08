# Handbook

Written for a reader who does not have the original machine at hand: everything needed to recreate the setup, to name the
tools and say why each was chosen, and to explain how the system is built and measured.

| file | read it when |
|---|---|
| [environment-setup.md](environment-setup.md) | you need to rebuild the development environment on a fresh Mac or Linux box (90 minutes, all free) |
| [tools-and-frameworks.md](tools-and-frameworks.md) | someone asks "what did you use and why", or you need the gotchas we hit with each tool |
| [system-design-notes.md](system-design-notes.md) | someone asks "how would you build a voice agent / an RL environment / a benchmark / HIPAA guardrails" |
| [talking-points.md](talking-points.md) | you want the numbers, the decisions and the lessons in two pages |
| [voice-agent-stack.md](voice-agent-stack.md) | you need the map of the production platform: services, libraries per service, the agent's modules, workflows, test layers, CI/release/infra |
| [experience-sheet.md](experience-sheet.md) | someone asks about running the production voice agent: latency, STT, evaluation layers, incidents, security habits (generic, no customer names) |
| [glossary.md](glossary.md) | a term in the plan, the data or the code is unfamiliar (families, knobs, tiers, reward checks, voice gap, …) |

Everything in these pages comes from work that is in this repo or in the private repos it was exported from; the numbers
are the ones recorded in `docs/clinic-calls/README.md` and the plan. Where something was not run (paid GPU training,
hosted-model columns) the pages say so.
