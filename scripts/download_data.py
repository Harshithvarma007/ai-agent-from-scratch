"""Download the DABstep dataset into workspace/data/ (stdlib only, idempotent).

    python scripts/download_data.py            # skip files that are already there and verified
    python scripts/download_data.py --force    # re-download everything

DABstep by Adyen, https://huggingface.co/datasets/adyen/DABstep, CC-BY-4.0. Each file is checked
against the SHA-256 this repo was built and measured with, so your numbers match ours.
"""
import argparse
import hashlib
import os
import sys
import urllib.request

BASE = "https://huggingface.co/datasets/adyen/DABstep/resolve/main/data/context/"
FILES = {
    "payments.csv": "5fbb26210a45427d7a6560cfab3a362a08e4067f27cd03695f211a51c47ffc25",
    "merchant_data.json": "f158e834de27ee8407e99d9c32da690357981d2770958fd302a9bb79ced83581",
    "merchant_category_codes.csv": "83247d79f5ddcdd180e02f184b644f4912731942a2fc58d91f2a06b911d04b41",
    "acquirer_countries.csv": "6744cf1962e4b5086342821e627f4557c32f06c3c58e8b0ff3b449dcc634f37f",
}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dest", default=os.path.join(
        os.environ.get("AGENT_WORKSPACE", os.path.join(ROOT, "workspace")), "data"))
    p.add_argument("--force", action="store_true", help="re-download files that already exist")
    p.add_argument("--no-verify", action="store_true", help="skip the SHA-256 check")
    args = p.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    bad = []
    for name, digest in FILES.items():
        path = os.path.join(args.dest, name)
        if os.path.exists(path) and not args.force and (args.no_verify or sha256(path) == digest):
            print(f"ok      {name}")
            continue
        print(f"fetch   {name} ...", end=" ", flush=True)
        tmp = path + ".part"
        urllib.request.urlretrieve(BASE + name, tmp)
        os.replace(tmp, path)
        print(f"{os.path.getsize(path) / 1e6:.1f} MB")
        if not args.no_verify and sha256(path) != digest:
            bad.append(name)
    if bad:
        sys.exit(f"checksum mismatch: {', '.join(bad)}. The upstream files changed; results may "
                 "differ from docs/RESULTS.md. Re-run with --no-verify to keep them anyway.")
    print(f"dataset ready in {os.path.relpath(args.dest)}")


if __name__ == "__main__":
    main()
