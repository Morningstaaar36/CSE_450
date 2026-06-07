"""
fetch_real_data.py
==================

Download the real-world point clouds used by the Milestone-1 test suite into
`data/`.  Stdlib only (urllib / tarfile / gzip) — no extra dependencies.

    python fetch_real_data.py            # fetch everything missing
    python fetch_real_data.py --force    # re-download even if present

Each file targets a specific part of the pipeline:

    indoor_real.ply  real indoor scan (Redwood living-room fragment)
                     → scene-vs-object classifier + "indoor aligns upright"
    luigi_3dgs.ply   real 3D Gaussian Splatting figurine (f_dc/opacity/scale/rot)
                     → loader's 3DGS SH-DC colour path + flat-object classification
    bonsai_3dgs.splat real 3DGS *indoor* 360° capture (antimatter15 .splat)
                     → loader's .splat path + object-centric-scene classification
    bicycle_3dgs.splat real 3DGS *outdoor* 360° capture (antimatter15 .splat)
                     → loader's .splat path on a large outdoor scene
    buddha.ply       Stanford Happy Buddha (tall statue)
                     → "obvious objects get roughly aligned"
    bunny.ply        Stanford Bunny (free-form, up is semantic)
                     → honest `up_ambiguous` flagging
    armadillo.ply    Stanford Armadillo (binary BIG-endian PLY)
                     → loader byte-order robustness

`big.ply` ships with the repo and is not downloaded here.
"""

from __future__ import annotations

import argparse
import gzip
import io
import os
import sys
import tarfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

UA = {"User-Agent": "Mozilla/5.0 (milestone1-fetch)"}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def fetch_plain(name: str, url: str) -> None:
    """Download a raw .ply straight to data/."""
    data = _get(url)
    if data[:3] != b"ply":
        raise ValueError(f"{url} did not return a PLY (got {data[:16]!r})")
    with open(os.path.join(DATA, name), "wb") as f:
        f.write(data)


def fetch_splat(name: str, url: str) -> None:
    """Download a binary antimatter15 `.splat` (32 bytes per Gaussian)."""
    data = _get(url)
    if len(data) == 0 or len(data) % 32 != 0:
        raise ValueError(
            f"{url} did not return a valid .splat "
            f"(size {len(data)} is not a multiple of 32)"
        )
    with open(os.path.join(DATA, name), "wb") as f:
        f.write(data)


def fetch_gz(name: str, url: str) -> None:
    """Download a gzip-compressed .ply.gz and decompress to data/."""
    raw = gzip.decompress(_get(url))
    with open(os.path.join(DATA, name), "wb") as f:
        f.write(raw)


def fetch_from_tar(name: str, url: str, member_suffix: str) -> None:
    """Download a .tar.gz and extract the member ending with `member_suffix`."""
    blob = _get(url)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        member = next((m for m in tar.getmembers()
                       if m.name.endswith(member_suffix)), None)
        if member is None:
            raise ValueError(f"{member_suffix} not found in {url}")
        with tar.extractfile(member) as src:
            with open(os.path.join(DATA, name), "wb") as f:
                f.write(src.read())


# (output name, fetcher, args...)
TARGETS = [
    ("indoor_real.ply", fetch_plain,
     ("https://github.com/isl-org/open3d_downloads/releases/download/"
      "20220201-data/fragment.ply",)),
    ("luigi_3dgs.ply", fetch_plain,
     ("https://huggingface.co/datasets/dylanebert/3dgs/resolve/main/"
      "luigi/luigi.ply",)),
    ("bonsai_3dgs.splat", fetch_splat,
     ("https://huggingface.co/datasets/dylanebert/3dgs/resolve/main/"
      "bonsai/bonsai-7k-mini.splat",)),
    ("bicycle_3dgs.splat", fetch_splat,
     ("https://huggingface.co/datasets/dylanebert/3dgs/resolve/main/"
      "bicycle/bicycle-7k-mini.splat",)),
    ("buddha.ply", fetch_from_tar,
     ("http://graphics.stanford.edu/pub/3Dscanrep/happy/happy_recon.tar.gz",
      "happy_vrip_res2.ply")),
    ("bunny.ply", fetch_from_tar,
     ("http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz",
      "reconstruction/bun_zipper.ply")),
    ("armadillo.ply", fetch_gz,
     ("http://graphics.stanford.edu/pub/3Dscanrep/armadillo/Armadillo.ply.gz",)),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true",
                    help="re-download even if the file already exists")
    args = ap.parse_args()

    os.makedirs(DATA, exist_ok=True)
    ok, failed, skipped = 0, 0, 0
    for name, fn, fnargs in TARGETS:
        dest = os.path.join(DATA, name)
        if os.path.isfile(dest) and not args.force:
            print(f"  {name:18s} present — skip")
            skipped += 1
            continue
        try:
            print(f"  {name:18s} downloading…", flush=True)
            fn(name, *fnargs)
            print(f"  {name:18s} OK ({os.path.getsize(dest):,} bytes)")
            ok += 1
        except Exception as exc:                       # noqa: BLE001
            print(f"  {name:18s} FAILED: {exc}")
            failed += 1

    print(f"\n{ok} downloaded, {skipped} already present, {failed} failed.")
    print("Run the tests with:  python tests/test_pipeline.py")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
