#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Simulated training workload: occupy >30 GiB on every GPU and keep computing.
# The --tag argument is used by the monitor to identify its own processes.

import os
import sys
import time
import signal
import argparse
import threading

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

DEFAULT_TAG = "__UTILS_train_job__"
ALLOC_GIB = 30.5
MIN_ALLOC_GIB = 30.1
BYTES_PER_GIB = 1024 ** 3
CHUNK_BYTES = 1024 ** 3  # 1 GiB chunks

stop_event = threading.Event()


def handle_signal(signum, frame):
    stop_event.set()


signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


def allocate_on_gpu(device_id: int):
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    target = int(ALLOC_GIB * BYTES_PER_GIB)
    min_target = int(MIN_ALLOC_GIB * BYTES_PER_GIB)
    tensors = []

    while target >= min_target:
        try:
            n_chunks = target // CHUNK_BYTES
            remainder = target - n_chunks * CHUNK_BYTES

            for _ in range(n_chunks):
                tensors.append(
                    torch.empty(CHUNK_BYTES, dtype=torch.uint8, device=f"cuda:{device_id}")
                )
            if remainder > 0:
                tensors.append(
                    torch.empty(remainder, dtype=torch.uint8, device=f"cuda:{device_id}")
                )

            torch.cuda.synchronize(device_id)
            allocated = sum(t.numel() for t in tensors)
            print(
                f"GPU {device_id}: occupied {allocated / BYTES_PER_GIB:.2f} GiB",
                flush=True,
            )
            return tensors
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                tensors.clear()
                torch.cuda.empty_cache()
                # Reduce by a small factor so a retry can still land within
                # [MIN_ALLOC_GIB, ALLOC_GIB); 0.95 used to jump straight past
                # the floor, making the retry branch unreachable dead code.
                target = int(target * 0.99)
                print(
                    f"GPU {device_id}: OOM, retrying with {target / BYTES_PER_GIB:.2f} GiB",
                    flush=True,
                )
                continue
            raise

    raise RuntimeError(f"GPU {device_id}: could not allocate >{MIN_ALLOC_GIB} GiB")


def compute_loop(device_id: int):
    torch.cuda.set_device(device_id)

    size = 4096
    a = torch.randn(size, size, device=f"cuda:{device_id}", dtype=torch.float16)
    b = torch.randn(size, size, device=f"cuda:{device_id}", dtype=torch.float16)
    c = None

    while not stop_event.is_set():
        c = torch.matmul(a, b)
        torch.cuda.synchronize(device_id)
        # Small pause to avoid absolutely pinning the GPU scheduler.
        time.sleep(0.01)

    # c may not exist yet if the stop event fired before the first matmul.
    if c is not None:
        del a, b, c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=DEFAULT_TAG)
    args = parser.parse_args()

    ngpus = torch.cuda.device_count()
    print(f"Found {ngpus} GPU(s) [{args.tag}]", flush=True)
    if ngpus == 0:
        print("No GPUs found, exit.", flush=True)
        sys.exit(1)

    gpu_holders = []
    for i in range(ngpus):
        holders = allocate_on_gpu(i)
        gpu_holders.append(holders)

    threads = []
    for i in range(ngpus):
        t = threading.Thread(target=compute_loop, args=(i,), daemon=True)
        t.start()
        threads.append(t)

    print("Training workload started.", flush=True)
    while not stop_event.is_set():
        time.sleep(1)

    for t in threads:
        t.join(timeout=5)

    print("Stop signal received, releasing memory.", flush=True)


if __name__ == "__main__":
    main()
