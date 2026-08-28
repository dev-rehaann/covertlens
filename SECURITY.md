# Security Policy

## Responsible Use Policy

covertlens is strictly for defensive security research and education. Covert-channel-generating tools referenced by this project—including Iodine, dnscat2, ptunnel, and icmpsh—are used only to generate labeled traffic samples in isolated, air-gapped, or otherwise authorized lab environments. They must never be used against networks or systems without explicit authorization.

Before capturing or analyzing network traffic, users must comply with all applicable laws and their institution's acceptable-use and research-ethics policies. This is especially important when traffic is not exclusively their own.

This repository will never include ready-to-use exploit or tunnel-building code. It contains only defensive detection and analysis logic.

## Reporting a Vulnerability

This is a student research repository, not a live production service. Report security issues in the detection code—such as an evasion-enabling bug—or vulnerabilities in the dashboard or API by opening a GitHub issue tagged `security` or emailing [dev.rehaann@gmail.com](mailto:dev.rehaann@gmail.com). Do not attach sensitive captures, credentials, or private network details to a public issue.

## Data Handling

Raw packet captures may contain sensitive information, including real IP addresses, DNS queries, hostnames, and traffic metadata. They are excluded from this repository by `.gitignore` and must never be committed.

Before sharing sample pcaps, contributors must sanitize or anonymize them with an appropriate tool such as `tcprewrite` or equivalent packet-scrubbing software. Contributors remain responsible for confirming that shared data contains no private, identifying, or unauthorized traffic.
