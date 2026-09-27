## 1. What did we learn (concepts)

**Headline finding (narrowly stated):** On the held-out Iodine session, the unsupervised models achieved higher recall than the supervised reference — Isolation Forest 76.8%, Autoencoder 84.2%, XGBoost 30.5%. With only two covert sessions per protocol and each model operating at its own threshold, this single comparison cannot establish general superiority of unsupervised methods, nor prove XGBoost specifically overfit to dnscat2's signature — though the result is consistent with that explanation and motivates testing against more tools before drawing a general conclusion.

**Feature-level findings:**
- **`size_cv` direction is protocol-dependent.** DNS tunnels pack queries near max size for throughput → lower variance than legit DNS. ICMP tunnels vary payload size to pack data efficiently → higher variance than legit ping. One feature, opposite signs per protocol.
- **Shannon entropy has a real limitation here.** Entropy measures the uniformity of the byte-value frequency distribution, not unpredictability directly. Observed legitimate ICMP entropy averaged 5.315 bits/byte — well below the 8-bit theoretical maximum, not near it. Our baseline capture script intentionally varies some ping payload sizes rather than using one fixed size throughout, which likely broadens the legitimate byte-distribution and contributes to this result. Covert ICMP entropy averaged lower (4.35) with high variance, from a mix of high-entropy data-bearing packets and near-empty/repetitive keepalive packets.
- **Compression ratio, added to probe this further, also inverted for ICMP but stayed useful.** Legitimate ICMP compression-ratio standard deviation was ~0.0044, versus ~0.5866 for covert — a wide separation in spread even with the mean pointing the "wrong" direction. Isolation Forest (a tree-based ensemble that isolates points via random recursive partitioning) and the Autoencoder (whose anomaly score is reconstruction error) can both exploit a feature that separates by spread rather than by mean.
- **DNS field features behaved as hypothesized:** `interarrival_cv`, `entropy_mean`, `compression_ratio_mean`, `mean_query_length`, and `txt_null_ratio` were all reliably higher for covert traffic.
- **Feature de-duplication:** `mean_query_length`, `max_query_length`, and `size_mean` correlated above 0.95 in DNS. Both `mean_query_length` and `size_mean` were retained; only `max_query_length` was dropped.

**Evaluation-methodology findings:**
- Random flow-level train/test splits leak information once flows are windowed sub-samples of the same continuous capture — leave-one-session-out (grouped by source pcap) was required.
- ROC-AUC requires both classes present in the test set to rank positive against negative examples, so it's mathematically undefined for a single-class holdout. Average precision (AP) can still be computed but is uninformative in this setting — trivially 1.0 when every held-out flow is covert, undefined when every held-out flow is legit (no positive examples to average over). FPR (legit-holdout) and recall (covert-holdout) were reported separately instead, never averaged together.
- Isolation Forest and the Autoencoder train on the unlabeled, mixed training fold (legit and covert flows both present, unfiltered by label) — this is unsupervised training, not semi-supervised training on label-filtered normal-only data. Both rely on an implicit assumption that legitimate flows form the effective majority in that mixed data. With only one baseline session available, holding it out left the Autoencoder's training fold entirely covert, violating that assumption — it then flagged 100% of the truly legitimate held-out flows as anomalous. Adding three more independent baseline sessions corrected this.
- Isolation Forest shows real recall variance across DNS tools (dnscat2 ~100% vs. iodine ~77%); the Autoencoder trades some peak recall for more consistent detection across tools.

**Verified diagnostic result:** The Autoencoder and XGBoost flagged exactly the same 32 of 170 flows as false positives on the original baseline holdout fold (0 autoencoder-only, 0 XGBoost-only), despite independent score vectors and different thresholds — ruling out a prediction-reuse bug. Those 32 flows are feature-distinct from the rest of that session (mean query length ~21 vs. ~10.75; `size_cv` 0 vs. ~0.102), indicating a real subset of that baseline session both models found anomalous for reasons not yet explained.

**Open, unconfirmed observation:** The original baseline capture contained 32 AAAA packets; three later baseline runs produced none. The cause is unconfirmed — no evidence currently points to the script or resolver specifically.

## 2. How did we do it (methodology)

Phases 0–4 implemented and completed:
- **Phase 0:** Repo scaffold (README, SECURITY.md, LICENSE, gitignore, pyproject.toml).
- **Phase 1:** Isolated dual-VM lab (VirtualBox host-only network, no external route, reverified before every capture). Tools: Iodine and dnscat2 (DNS), ptunnel and Hans (ICMP). icmpsh was attempted but abandoned — Windows-only slave, unreliable under Wine.
- **Phase 2:** Flow segmentation (30s gap timeout) plus time-windowing for long continuous sessions; per-flow size/timing/entropy/compression/field-anomaly features.
- **Phase 3:** Isolation Forest, Autoencoder, and an XGBoost reference model, evaluated via leave-one-session-out cross-validation grouped by source pcap, scaling fit per training fold to prevent leakage.

- **Phase 4:** FastAPI results/scoring backend and Streamlit research dashboard. The Evaluation Results tab presents saved LOSO findings; Live Scoring Demo uses separately saved, protocol-specific Isolation Forest and Autoencoder models trained on all available flows. Scalers, feature order, and training-score 90th-percentile thresholds are persisted for reproducible inference. These final models have nothing held out and are not equivalent evidence to LOSO evaluation.

**Dataset:** Twelve separate capture runs (4 DNS legit, 2 DNS covert, 4 ICMP legit, 2 ICMP covert) produced 965 flow/window samples after windowing: DNS 568 legit/269 covert; ICMP 80 legit/48 covert. Separate runs are distinct sessions, not a guarantee of statistical independence: they share tools and lab conditions. Windows within one run remain correlated. LOSO keeps each source capture entirely in one fold.

**External live-demo checks (not evaluation):** Four publicly available captures were scored through the API and uploaded through the browser. Both models returned finite scores; tables, charts, and the demo-only warning rendered. They were not added to training, the manifest, or LOSO results; the final models were not retrained for these checks.

| Capture | Source | Bytes | Parsed packets | Scored flows/windows | Flagged by either |
|---|---|---:|---:|---:|---:|
| `iodine.pcap` | [dmachard DNS samples](https://github.com/dmachard/datasets-malicious-dns) | 3,454 | 24 DNS | 1 | 1 |
| `dns2tcp.pcap` | [dmachard DNS samples](https://github.com/dmachard/datasets-malicious-dns) | 4,522 | 26 DNS | 1 | 1 |
| `dnscat2_dns_tunneling_1hr.pcap` | [Active Countermeasures](https://www.activecountermeasures.com/malware-of-the-day-dnscat2-dns-tunneling/) | 2,462,756 | 14,487 DNS | 7,085 | 7,085 |
| `5d176b7cb326f05a1985be5d4d4d9074_special_delivery.pcap` | [KITCTF Special Delivery](https://kitctf.de/writeups/hitbctf/special_delivery) | 92,015 | 267 ICMP | 1 | 1 |

Both models individually flagged every returned row in these checks. This is not a detection-accuracy result: no independently verified per-flow labels or benign control set were evaluated. The two tiny DNS files are smoke tests, not substantial benchmarks. The dnscat2 capture has 6,256 distinct UDP port pairs; port-aware grouping plus the 30-second timeout explains its many small flows and differs markedly from the training lab's flow distribution. The CTF capture has 308 frames, including 41 IP-fragment frames not returned as separate decoded ICMP rows. Its one scored flow lasts about 48 seconds, below the 60-second subdivision trigger. None of these observations establishes a false-positive rate or broad generalization.

Downloaded captures remain local in gitignored `samples/`; sources are linked instead of redistributing traffic or assuming redistribution rights. Screenshot checks are local artifacts, not research metrics.

## 3. Step-by-step guide (reproducing the pipeline)

```bash
python -m covertlens.capture.manifest
python -m covertlens.features.build_dataset --window-seconds 30 --min-duration-to-split 60
python scripts/inspect_features.py
python scripts/audit_dataset.py
python -m covertlens.models.run_loso_evaluation --protocol dns
python -m covertlens.models.run_loso_evaluation --protocol icmp
python -m covertlens.models.train_final
```

Install the package and native TShark first; see [README setup instructions](../README.md#setup-instructions). Generated datasets, evaluation CSVs, and final models are not shipped in Git. After the pipeline, launch the backend and frontend in separate terminals:

```bash
uvicorn covertlens.dashboard.api:app --host 127.0.0.1 --port 8000
streamlit run src/covertlens/dashboard/app.py --server.address 127.0.0.1 --theme.base dashboard-theme.toml
```

The results tab reads the latest saved LOSO run per protocol. The live tab applies saved thresholds without recalibrating them on uploads. Keep the API local: it is unauthenticated and accepts untrusted capture bytes for parsing. [Hosting guidance](../README.md#hosting-and-vercel) describes why the current application is not a drop-in Vercel deployment.

## 4. Anything extra

- **Real limitation:** only 2 covert sessions per protocol. Fold-to-fold recall variance signals limited tool diversity, not proof of full generalization.
- **Deferred, not abandoned:** window-concatenated compression ratio as a possible ICMP refinement; more independent baseline/covert sessions; GAN-based adversarial hardening stretch goal. The live Wireshark Lua plugin was cut early in favor of the now-implemented dashboard.
- **Dead end worth documenting:** icmpsh's Windows-only slave under Wine — unreliable raw-socket emulation, replaced with Hans.
