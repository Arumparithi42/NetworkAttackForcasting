# SIH26153 — AI-Based Network Attack Forecasting (World Model)

Smart India Hackathon 2026 · Theme: Blockchain & Cybersecurity · Category: Software

A host-centric **world model** learns how network traffic states evolve, `P(S_{t+1} | history)`, and
simulates the next K time windows. From that simulation it estimates the probability that a host is moving
toward an attack stage (mapped to MITRE ATT&CK), and it explains every forecast. It runs fully offline.

**Status:** design phase. There is no code and there are no results yet.

- 📐 Full architecture & implementation blueprint: [`docs/BLUEPRINT.md`](docs/BLUEPRINT.md)
- 📄 Official problem statement: [`docs/problem_statement_SIH26153.md`](docs/problem_statement_SIH26153.md)

Sections for installation, dataset preparation, training, inference, dashboard and evaluation will be
added as the components are built (see BLUEPRINT §15 and §21.2).
