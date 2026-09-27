# Security Policy

## Responsible Use Policy

covertlens is strictly for defensive security research and education. Covert-channel-generating tools referenced by this project—including Iodine, dnscat2, ptunnel, Hans, and icmpsh—are used only to generate labeled traffic samples in isolated, air-gapped, or otherwise authorized lab environments. They must never be used against networks or systems without explicit authorization. The current dataset uses Hans rather than the abandoned icmpsh/Wine approach.

Before capturing or analyzing network traffic, users must comply with all applicable laws and their institution's acceptable-use and research-ethics policies. This is especially important when traffic is not exclusively their own.

This repository will never include ready-to-use exploit or tunnel-building code. It contains only defensive detection and analysis logic.

## Reporting a Vulnerability

This is a student research repository, not a live production service. Report security issues in the detection code—such as an evasion-enabling bug—or vulnerabilities in the dashboard or API by opening a GitHub issue tagged `security` or emailing [dev.rehaann@gmail.com](mailto:dev.rehaann@gmail.com). Do not attach sensitive captures, credentials, or private network details to a public issue.

## Data Handling

Raw packet captures may contain sensitive information, including real IP addresses, DNS queries, hostnames, and traffic metadata. They are excluded from this repository by `.gitignore` and must never be committed.

The ignored `samples/`, `data/` contents, and generated `models_release/` artifacts must remain local. Do not force-add captures, credentials, model binaries, or local agent/editor configuration. Public capture downloads still require privacy and redistribution checks.

Before sharing sample pcaps, contributors must sanitize or anonymize them with an appropriate tool such as `tcprewrite` or equivalent packet-scrubbing software. Contributors remain responsible for confirming that shared data contains no private, identifying, or unauthorized traffic.

## Dashboard Safety

The unauthenticated API and Streamlit demo are intended for localhost or an authorized lab only. Uploading sends the entire capture to the configured backend; use only a trusted endpoint. Temporary uploaded captures are deleted after processing, but this does not guarantee erasure from host backups or infrastructure logs. Load only trusted model artifacts: serialized models are not safe untrusted inputs.

Public hosting requires authentication, upload/rate/concurrency limits, parser isolation, and a reviewed retention/logging policy. Do not expose the isolated capture VMs or log capture contents or credentials. Demo flags are research indicators, not proof of a covert channel; LOSO results remain the separate evaluation evidence.
