# covertlens

Cross-protocol covert channel detection via traffic side-channel statistics — no payload inspection, no signatures.

## Overview

Covert channels smuggle data through traffic that resembles legitimate protocol use. DNS tunneling can encode information in queries and responses, while ICMP tunneling can conceal it within diagnostic traffic. These techniques may evade controls that depend on known payload patterns or protocol-specific signatures.

covertlens studies detection through side-channel statistics: packet-size distributions, inter-arrival timing and regularity, Shannon entropy, payload compression ratios, and protocol-field anomaly statistics. Payload bytes are used for statistical measurements, not decoded for content matching or tunnel signatures. It fuses these features and applies unsupervised anomaly detection because representative, accurately labeled covert-channel datasets are scarce and often tied to particular tools or lab conditions.

## Current scope

The current phase targets DNS and ICMP traffic only. The architecture is intended to support later research on other protocols and observable channels, including TLS SNI, NTP, and HTTP/2 timing, but these are future work rather than current capabilities.

Repository setup, isolated-lab data collection, the feature pipeline, and initial modeling/evaluation are implemented (Phases 0–3). The FastAPI/Streamlit dashboard is planned for Phase 4 and is not yet implemented. Lab validation uses Iodine and dnscat2 for DNS, and ptunnel and Hans for ICMP; the attempted icmpsh/Wine path was abandoned.

## Why this approach

- Signature-based intrusion detection systems rely on known indicators or protocol patterns; statistical anomaly detection can study behavior that changes across tunnel implementations.
- Size, timing, and entropy features remain measurable when content is encoded, obfuscated, or encrypted.
- Fusing multiple weak signals avoids relying on any single protocol field or heuristic.
- This work is distinct from the prior ThreatNet project: covertlens focuses on unsupervised, side-channel-based covert-channel detection rather than signature-based IDS detection.

## Related work

- Machine-learning methods for detecting DNS covert channels from query structure, traffic volume, entropy, and timing features.
- Support vector machine approaches to identifying anomalous ICMP payload and flow characteristics.
- Surveys and taxonomies of cross-protocol or protocol-agnostic covert timing channels.
- Unsupervised network anomaly detection under limited or unreliable labels.

**TODO: cite properly before publication.** Exact paper titles, authors, venues, and DOIs will be added after the literature review.

## Architecture

```mermaid
flowchart LR
    A[tshark capture] --> B[pyshark / Scapy feature extraction]
    B --> W[Bidirectional flows / non-overlapping time windows]
    W --> C[Feature store]
    C --> D{Unsupervised models}
    D --> E[Isolation Forest]
    D --> F[Autoencoder]
    E --> G[Anomaly score]
    F --> G
    G --> H[FastAPI / Streamlit dashboard - planned]
```

## Repository structure

- `data/` — ignored raw captures, processed feature data, and external baseline datasets; only directory markers are tracked.
- `src/` — the `covertlens` package, organized into capture, feature extraction, modeling, and dashboard components.
- `notebooks/` — exploratory analysis notebooks; generated notebook artifacts are not tracked.
- `tests/` — automated tests for the detection pipeline.
- `docs/` — [isolated-lab setup](docs/lab-setup.md), [Phase 0–3 findings and methodology notes](docs/notes.md), and literature-review notes.
- `scripts/` — one-off repository setup and isolated-lab support scripts.

## Setup

Use Python 3.11+ and install the package from the repository root:

```bash
python -m pip install -e .
```

For development tools, use `python -m pip install -e ".[dev]"`. The optional supervised XGBoost reference requires `python -m pip install -e ".[reference]"` (or `".[dev,reference]"` for both).

Install TShark separately at the OS level and make sure `tshark -v` works in the processing environment; pyshark is a wrapper and does not install TShark. See [requirements-lab.txt](requirements-lab.txt) for non-Python tools and [docs/lab-setup.md](docs/lab-setup.md) before any lab capture. Tunnel-tool installation and configuration remain manual.

With authorized lab captures in `data/raw/`, run:

```bash
python -m covertlens.capture.manifest
python -m covertlens.features.build_dataset --window-seconds 30 --min-duration-to-split 60
python scripts/inspect_features.py
python scripts/audit_dataset.py
python -m covertlens.models.run_loso_evaluation --protocol dns
python -m covertlens.models.run_loso_evaluation --protocol icmp
```

Raw captures, manifests, features, and generated evaluation results stay local and gitignored. The repository does not ship the lab dataset. `python scripts/compare_baseline_sessions.py` provides an additional DNS baseline-session diagnostic.

### Evaluation and initial findings

The primary evaluation is leave-one-session-out (LOSO), grouped by the original `source_file`, with scaling fitted on each training fold only. Non-overlapping windows from the same capture stay together to avoid session leakage. `run_comparison.py` remains an exploratory random-row quick check, not the evaluation used for research claims.

Isolation Forest and the Autoencoder train on the full unlabeled training fold, assuming it is sufficiently representative of normal traffic; they do not filter training rows by ground-truth labels. Their thresholds use the training-score 90th percentile by default. XGBoost uses training labels and a fixed 0.5 probability threshold: it is a supervised reference, not a like-for-like comparison or a guaranteed performance upper bound.

The current local dataset contains 12 capture runs and 965 flow/window samples: DNS has 568 legitimate and 269 covert samples; ICMP has 80 legitimate and 48 covert samples. Each protocol has four baseline sessions and two covert sessions. Windows remain correlated within sessions, and shared lab conditions limit independence across runs.

On the single held-out Iodine session, recall was 76.8% for Isolation Forest, 84.2% for the Autoencoder, and 30.5% for XGBoost. This is a preliminary result at different operating thresholds, not evidence of general superiority. With single-class held-out sessions, ROC-AUC is undefined and average precision cannot assess class discrimination; evaluation reports false-positive rate on legitimate sessions and recall on covert sessions separately.

Feature direction also depends on protocol: observed DNS covert traffic had lower packet-size variation than its baseline, while ICMP covert traffic had higher variation. Entropy is not a direct measure of unpredictability and was higher on average for legitimate ICMP in this dataset. See [the findings and methodology notes](docs/notes.md) for the detailed observations, diagnostic results, and limitations. More independent captures and tool diversity are needed before broad generalization claims.

## Ethical use & research disclaimer

covertlens is intended solely for defensive security research. Any tunneling tools used for validation—including Iodine, dnscat2, ptunnel, Hans, or icmpsh—must be run only in isolated lab virtual machines that are never connected to production networks. This repository does not include or distribute tunnel-building or exploit code.

Users are responsible for complying with their institution's or organization's policies and all applicable local laws when capturing or analyzing network traffic. See [SECURITY.md](SECURITY.md) for further guidance.

## License

MIT License, see [LICENSE](LICENSE) file.

## Status / roadmap

- [x] Phase 0 — Repository setup
- [x] Phase 1 — Initial isolated-lab data collection
- [x] Phase 2 — Feature pipeline with long-flow windowing
- [x] Phase 3 — Initial modeling and session-grouped evaluation
- [ ] Additional independent capture sessions and broader validation
- [ ] Phase 4 — FastAPI / Streamlit dashboard
- [ ] Phase 5 — Adversarial hardening *(stretch)*
