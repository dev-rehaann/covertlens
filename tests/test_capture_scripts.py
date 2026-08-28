import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    ROOT / "scripts" / "capture_baseline.sh",
    ROOT / "scripts" / "capture_baseline_icmp.sh",
]


def bash_path(path: Path) -> str:
    if os.name != "nt":
        return str(path)
    return f"/mnt/{path.drive[0].lower()}/{path.relative_to(path.anchor).as_posix()}"


def test_capture_scripts() -> None:
    bash = shutil.which("bash")
    assert bash, "bash is required for script syntax checks"
    for script in SCRIPTS:
        subprocess.run([bash, "-n", bash_path(script)], check=True)

    domains = (ROOT / "scripts" / "domains.txt").read_text().splitlines()
    targets = (ROOT / "scripts" / "ping_targets.txt").read_text().splitlines()
    assert 25 <= len(domains) <= 35
    assert {"8.8.8.8", "1.1.1.1", "192.168.56.10"} <= set(targets)


if __name__ == "__main__":
    test_capture_scripts()
    print("capture script checks passed")
