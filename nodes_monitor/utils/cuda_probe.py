#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight CUDA health probe for the GPU monitor.

Tries to select every visible GPU and allocate a tiny tensor.  If the driver
is stuck in the ``Reset required`` state, ``torch.cuda.set_device`` will fail
and the probe exits non-zero.

The probe DOES use the GPU: creating a context on a device shows up in
nvidia-smi as a few-hundred-MiB memory.used for as long as the probe lives.
That is why the GPUs are touched in PARALLEL threads — the footprint window
is ~one context-creation time (~1-2s) instead of ngpus times that, so a
concurrent nvidia-smi fetch (e.g. from a second monitor session) is far less
likely to observe a transient "USED" state on a healthy node.
"""

import sys
from concurrent.futures import ThreadPoolExecutor

try:
    import torch
except ImportError as exc:
    # Environment failure (interpreter without torch), NOT a GPU verdict.
    # Exit 2 so both callers treat it as transient, never toward BROKEN.
    print(exc, file=sys.stderr)
    sys.exit(2)


def touch_gpu(device_id: int) -> None:
    torch.cuda.set_device(device_id)
    torch.empty(1, device=f"cuda:{device_id}")


def main() -> int:
    ngpus = torch.cuda.device_count()
    if ngpus == 0:
        print("no gpus", file=sys.stderr)
        return 1

    with ThreadPoolExecutor(max_workers=ngpus) as pool:
        list(pool.map(touch_gpu, range(ngpus)))

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        print(exc, file=sys.stderr)
        sys.exit(1)
