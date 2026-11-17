"""Download and verify the pretrained SOBER epoch-10 checkpoint."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import tempfile
import urllib.request


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "checkpoints/sober_epoch10.pth")
    args = parser.parse_args()
    metadata = json.loads((root / "models/epoch10.json").read_text())
    if args.output.exists():
        if checksum(args.output) != metadata["sha256"]:
            raise ValueError(f"Checksum mismatch: {args.output}; choose another output path")
        print(f"Verified: {args.output}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {metadata['bytes'] / 1e6:.1f} MB")
    with tempfile.NamedTemporaryFile(dir=args.output.parent, suffix=".part", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with urllib.request.urlopen(metadata["url"], timeout=60) as response:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(chunk)
            stream.flush()
            if temporary.stat().st_size != metadata["bytes"] or checksum(temporary) != metadata["sha256"]:
                raise ValueError("Downloaded checkpoint failed SHA-256 verification")
            stream.close()
            os.replace(temporary, args.output)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"Verified: {args.output}")


if __name__ == "__main__":
    main()
