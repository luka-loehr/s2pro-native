#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare a parity WORK_DIR for build/s2p-parity from an oracle fixture set.

Pure Python (no numpy) so it runs on any host with python3.

  parity_prep.py FIXTURE_DIR WORK_DIR

Reads FIXTURE_DIR/manifest.json (+ frames.npy) and writes WORK_DIR/meta.json,
prompt_ids.bin (i64) and fixture_frames.bin (i32, frame-major [T][10]).
"""
import json
import os
import struct
import sys


def read_npy(path):
    """Minimal .npy v1/v2 reader -> (shape, fortran, dtype_str, raw_bytes)."""
    with open(path, "rb") as f:
        magic = f.read(6)
        assert magic == b"\x93NUMPY", f"{path}: not a .npy file"
        major, _minor = f.read(1)[0], f.read(1)[0]
        hlen = struct.unpack(
            "<I" if major >= 2 else "<H", f.read(4 if major >= 2 else 2))[0]
        header = eval(f.read(hlen).decode("latin1"),
                      {"__builtins__": {}}, {"False": False, "True": True})
        assert not header["fortran_order"], f"{path}: fortran order unsupported"
        return header["shape"], header["descr"], f.read()


DTYPE_FMT = {"<f4": "f", "<f8": "d", "<i4": "i", "<i8": "q", "<i2": "h"}


def npy_values(path):
    shape, descr, raw = read_npy(path)
    fmt = DTYPE_FMT[descr]
    n = len(raw) // struct.calcsize(fmt)
    return shape, list(struct.unpack(f"<{n}{fmt}", raw))


def main():
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    fixdir, work = sys.argv[1], sys.argv[2]
    os.makedirs(work, exist_ok=True)

    man = json.load(open(os.path.join(fixdir, "manifest.json")))
    ids = man["prompt_ids"]
    steps = int(man["steps"])

    with open(os.path.join(work, "prompt_ids.bin"), "wb") as f:
        f.write(struct.pack(f"<{len(ids)}q", *ids))

    shape, frames = npy_values(os.path.join(fixdir, "frames.npy"))
    assert len(shape) == 2 and shape[1] == 10, f"frames shape {shape}"
    with open(os.path.join(work, "fixture_frames.bin"), "wb") as f:
        f.write(struct.pack(f"<{len(frames)}i", *[int(v) for v in frames]))

    json.dump({"n_ids": len(ids), "steps": steps},
              open(os.path.join(work, "meta.json"), "w"))
    print(f"prepared {work}: {len(ids)} prompt ids, {steps} steps, "
          f"fixture frames {shape[0]}x{shape[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
