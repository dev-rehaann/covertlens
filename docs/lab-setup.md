# Isolated Capture Lab Setup

This lab exists only to generate labeled DNS and ICMP traffic for defensive
detection research. Never connect a tunnel-enabled VM to a production network,
the host LAN, or an internet-facing interface.

## Topology

```text
                         No routing or forwarding
                                  X
                     Internet / host production LAN
                                  X
                                  |
Host workstation (management and offline file transfer only)
  Host-only adapter: 192.168.56.1/24
                                  |
             VirtualBox host-only network: 192.168.56.0/24
             No gateway, no bridge, no upstream NAT
                    /                              \
 lab-client                                      lab-server
 192.168.56.11/24                               192.168.56.10/24
 - tshark capture                               - lab-only DNS service
 - baseline DNS/ICMP generator                  - iodined
 - iodine/dnscat2 clients                       - dnscat2 server
 - ptunnel/icmpsh clients                       - ptunnel server
                                                 - icmpsh listener
```

Use a host-only or hypervisor-internal network. Do not use a bridged adapter.
A normal VirtualBox NAT adapter permits internet egress and must be detached
before any tunnel tool starts. If another hypervisor offers an isolated
NAT-only network, use it only after proving it has no upstream route.

## VM provisioning checklist

### 1. Create the isolated network

- [ ] Create a dedicated `192.168.56.0/24` host-only or internal network.
- [ ] Disable DHCP, routing, forwarding, Internet Connection Sharing, and NAT
  on that network.
- [ ] Confirm neither VM has a second bridged, NAT, Wi-Fi, or physical-LAN
  adapter.
- [ ] If temporary NAT is used to patch a fresh OS, do so before installing or
  running tunnel tools. Power off both VMs and remove the NAT adapter afterward.

### 2. Create both VMs

- [ ] Install minimal Debian or Ubuntu Server as `lab-client` and `lab-server`.
- [ ] Attach only the isolated virtual adapter to each VM.
- [ ] Configure static addresses with no default gateway:

  | VM | Address | DNS during captures |
  |---|---|---|
  | `lab-server` | `192.168.56.10/24` | local only |
  | `lab-client` | `192.168.56.11/24` | `192.168.56.10` |

- [ ] Set the hostnames and confirm the two VMs can ping each other.
- [ ] Fully patch both operating systems before removing temporary package
  access, if temporary access is required.

### 3. Install common prerequisites

Run on both VMs while tunnel tools are not running:

```bash
sudo apt update
sudo apt install --yes git build-essential cmake pkg-config python3 python3-venv python3-pip ca-certificates
```

Install capture and traffic-diagnostic prerequisites on `lab-client`:

```bash
sudo apt install --yes tshark dnsutils iputils-ping traceroute
sudo usermod -aG wireshark "$USER"
```

Log out and back in after the group change, then verify `tshark -D` lists the
isolated interface. Set `CAPTURE_INTERFACE` when running the capture scripts if
the correct interface is not `any`.

Install server-side build/runtime prerequisites on `lab-server`:

```bash
sudo apt install --yes ruby ruby-dev libssl-dev libpcap-dev dnsmasq
```

Do not automate installation or configuration of Iodine, dnscat2, ptunnel, or
icmpsh. Obtain them manually from their official repositories, verify what will
run, and keep them inside these VMs. `lab-server` hosts the server/listener role;
`lab-client` hosts only the corresponding client role.

### 4. Prepare isolated baseline services

- [ ] Configure a lab-only DNS resolver or authoritative service on
  `lab-server`; it must not forward queries outside the lab.
- [ ] Populate local answers for names in `scripts/domains.txt` so DNS baseline
  captures contain request/response traffic without internet access.
- [ ] Do not bind the baseline resolver and a DNS tunnel server to the same
  address and port simultaneously. Use separate lab-only addresses or switch
  services between capture rounds.
- [ ] Replace `8.8.8.8` and `1.1.1.1` in `scripts/ping_targets.txt` with
  `lab-server` or additional lab-only IP aliases before collecting the ICMP
  baseline. Public targets must remain unreachable.
- [ ] Transfer the repository to `lab-client` through an offline shared folder
  or another controlled host-only mechanism.

## Mandatory safety checklist

Complete this checklist before starting **any** tunnel server or client:

- [ ] Both VM settings show exactly one host-only/internal adapter and no
  bridged or upstream NAT adapter.
- [ ] The host is not forwarding or sharing its internet connection onto the
  host-only adapter.
- [ ] `ip -br address` shows only loopback and the expected lab address.
- [ ] `ip route` shows the `192.168.56.0/24` route and **no default route**.
- [ ] Client-to-server connectivity succeeds:

  ```bash
  ping -c 2 192.168.56.10
  ```

- [ ] External connectivity fails as expected:

  ```bash
  ping -c 2 1.1.1.1
  traceroute -n 1.1.1.1
  ```

- [ ] A ping to the real host-LAN gateway also fails.
- [ ] A fresh snapshot of both powered-off VMs exists for rollback.
- [ ] The planned capture is authorized and contains only synthetic lab
  traffic.
- [ ] `tshark -D` confirms capture will use only the isolated interface.

If any isolation check unexpectedly succeeds, stop immediately, power off both
VMs, and correct the virtual network before continuing.

## Capture workflow

1. Run the safety checklist and record the VM snapshot names.
2. On `lab-client`, collect legitimate traffic with
   `scripts/capture_baseline.sh` and `scripts/capture_baseline_icmp.sh`.
3. For covert rounds, start the appropriate server/listener manually on
   `lab-server`, then use the matching `capture_covert_*.sh` script on
   `lab-client`. The script prompts for the manually started client.
4. Stop tunnel services after each round and verify the capture summary.
5. Move the pcaps through the controlled host-only/offline transfer path into
   the processing checkout's `data/raw/` directory.
6. Rebuild the manifest and feature dataset only after all expected files are
   present.

## Capture file naming

All captures must use:

```text
data/raw/{protocol}_{source}_{condition}_{timestamp}.pcap
```

- `protocol`: `dns` or `icmp`
- `source`: `baseline`, `iodine`, `dnscat2`, `ptunnel`, or `icmpsh`
- `condition`: `legit` or `covert`
- `timestamp`: UTC `YYYYMMDDTHHMMSSZ`

Examples:

```text
data/raw/dns_baseline_legit_20260829T120000Z.pcap
data/raw/dns_iodine_covert_20260829T121500Z.pcap
data/raw/icmp_ptunnel_covert_20260829T123000Z.pcap
```

Raw captures may contain addresses and query names. They are ignored by Git and
must not be committed; sanitize any capture before sharing it.
