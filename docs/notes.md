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

Four phases implemented and completed:
- **Phase 0:** Repo scaffold (README, SECURITY.md, LICENSE, gitignore, pyproject.toml).
- **Phase 1:** Isolated dual-VM lab (VirtualBox host-only network, no external route, reverified before every capture). Tools: Iodine and dnscat2 (DNS), ptunnel and Hans (ICMP). icmpsh was attempted but abandoned — Windows-only slave, unreliable under Wine.
- **Phase 2:** Flow segmentation (30s gap timeout) plus time-windowing for long continuous sessions; per-flow size/timing/entropy/compression/field-anomaly features.
- **Phase 3:** Isolation Forest, Autoencoder, and an XGBoost reference model, evaluated via leave-one-session-out cross-validation grouped by source pcap, scaling fit per training fold to prevent leakage.

Phase 4 (results-visualization dashboard) is planned, not yet started.

**Dataset:** Twelve separate capture runs (4 DNS legit, 2 DNS covert, 4 ICMP legit, 2 ICMP covert) produced 965 flow/window samples after windowing. The 12 runs are independent of each other; windowed sub-samples within the same run are not fully independent of one another (same tool, same session, same conditions) — worth keeping in mind when interpreting sample counts.

## 3. Step-by-step guide (reproducing the pipeline)

```bash
python -m covertlens.capture.manifest
python -m covertlens.features.build_dataset --window-seconds 30 --min-duration-to-split 60
python scripts/inspect_features.py
python scripts/audit_dataset.py
python -m covertlens.models.run_loso_evaluation --protocol dns
python -m covertlens.models.run_loso_evaluation --protocol icmp
```

## 4. Anything extra

- **Real limitation:** only 2 covert sessions per protocol. Fold-to-fold recall variance signals limited tool diversity, not proof of full generalization.
- **Deferred, not abandoned:** window-concatenated compression ratio as a possible ICMP refinement; more independent ICMP baseline/covert sessions; FastAPI/Streamlit results dashboard (Phase 4); GAN-based adversarial hardening stretch goal; live Wireshark Lua plugin (cut early in favor of a dashboard).
- **Dead end worth documenting:** icmpsh's Windows-only slave under Wine — unreliable raw-socket emulation, replaced with Hans.
