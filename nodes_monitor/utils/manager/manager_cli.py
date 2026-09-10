#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nodes monitor one-shot CLI — non-interactive train / kill / release.

    python3 manager_cli.py [--config FILE] --train [TARGETS]
    python3 manager_cli.py [--config FILE] --kill TARGETS
    python3 manager_cli.py [--config FILE] --release [TARGETS]

TARGETS is `all` or comma-separated 1-based indices into the configured node
list (e.g. 1,2,3). --train / --release default to `all` when TARGETS is
omitted; --kill always requires an explicit target (a bare `--kill` must
never widen into "every node"). manager.sh forwards its non-GUI arguments
here; the sibling module manager.py does all node contact.

One-shot semantics — read before reusing this module:
  * Trainers launched by --train intentionally SURVIVE the exit of this
    process — a one-shot "occupy these nodes" command would be pointless if
    the trainers died with it. That is the deliberate opposite of the GUI
    (manager.py), which kills session-launched trainers on exit. So this
    module NEVER calls manager.cleanup() / manager._kill_launched_trainers():
    the fire-and-forget Popens from manager.launch_trainer() are abandoned
    on purpose, and `--release` is the explicit undo.
  * Every node is contacted at most once per phase, in parallel threads —
    a dead node never delays the others, and total runtime stays ~constant
    in the number of targets (the GUI's node_worker discipline).
  * Output is script-friendly: one '<idx> <ip>: <result>' line per target,
    in target order. Exit status: 0 iff EVERY target succeeded, 1 on any
    per-node failure, 2 on usage errors (including invalid targets).

Import-safe by construction: all work happens under __main__ via main(), and
even `import manager` is deferred until after argument validation, so
--help / usage-error paths never touch the backend (or the network).
"""

import argparse
import concurrent.futures
import sys
import time

# The backend (sibling manager.py): imported INSIDE main() and bound to this
# global so the helpers below can use it. Placeholder documents the intent;
# see main() for why the import is deferred.
manager = None


# ---------------------------------------------------------------------------
# Argument parsing / target resolution
# ---------------------------------------------------------------------------
def parse_args(argv):
    ap = argparse.ArgumentParser(
        description="one-shot nodes monitor CLI: train / kill / release, print "
                    "one result line per node, exit. TARGETS = 'all' or "
                    "comma-separated 1-based node indices (e.g. 1,2,3).")
    ap.add_argument("--config", help="config file (default: .data override, else template)")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", nargs="?", const="all", metavar="TARGETS",
                      help="occupy node(s) NOW — idle nodes only (default: all)")
    # const=False (not None): a bare '--kill' must be distinguishable from an
    # absent one. Unlike train/release, kill is the blunt instrument (it
    # SIGKILLs everything nvidia-smi lists), so '--kill' without a value is
    # an error, never a silent 'all'.
    mode.add_argument("--kill", nargs="?", const=False, metavar="TARGETS",
                      help="kill every GPU process on the node(s) — explicit target required")
    mode.add_argument("--release", nargs="?", const="all", metavar="TARGETS",
                      help="kill only our tagged trainer processes (default: all)")
    args = ap.parse_args(argv)
    if args.kill is False:
        ap.error("kill needs an explicit target (all | 1,2,3)")
    return ap, args


def resolve_targets(ap, spec):
    """Map TARGETS to an ordered [(1-based idx, ip), ...] list. A bad index
    is a usage error (exit 2 via ap.error, same as argparse): a one-shot CLI
    must fail loudly rather than touch the wrong node. 'all' means every
    configured node, in list order."""
    if spec == "all":
        return list(enumerate(manager.IPS, 1))
    targets = []
    for part in spec.split(","):
        part = part.strip()
        if not part.isdigit():
            ap.error(f"invalid target '{part}' (expect: all | 1,2,3)")
        idx = int(part)
        if not (1 <= idx <= len(manager.IPS)):
            ap.error(f"index out of range: {part} (1..{len(manager.IPS)})")
        targets.append((idx, manager.IPS[idx - 1]))
    return targets


# ---------------------------------------------------------------------------
# Parallel plumbing / reporting
# ---------------------------------------------------------------------------
def run_parallel(ips, fn):
    """fn(ip) for every ip concurrently; results come back in the SAME order
    as ips (executor.map preserves submission order), so the report lines
    stay deterministic. One worker thread per node — they are pure I/O wait,
    and a hung node must never serialize the run."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(ips))) as ex:
        return list(ex.map(fn, ips))


def print_report(targets, outcomes):
    """Print one '<idx> <ip>: <result>' line per target, in target order;
    return True iff every outcome succeeded — that boolean IS the exit code."""
    ok = True
    for (idx, ip), (success, text) in zip(targets, outcomes):
        ok = ok and success
        print(f"{idx} {ip}: {text}")
    return ok


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def _safe_state(ip):
    """get_node_state with an exception guard: one wedged node must not lose
    the other nodes' results in a one-shot run. An error surfaces INSIDE the
    FAIL(...) wording so the line format stays uniform."""
    try:
        return manager.get_node_state(ip)[2]
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"


def _safe_check(ip):
    """check_trainer with an exception guard — unconfirmed is the safe
    verdict when the check itself blows up."""
    try:
        return bool(manager.check_trainer(ip))
    except Exception:  # noqa: BLE001
        return False


def do_train(targets):
    """Occupy the target nodes NOW. Launch happens ONLY on IDLE nodes —
    exactly the GUI's rule (an UNVERIFIED/USED/TRAIN/BROKEN/... node is never
    touched) — and refusals reuse the GUI's FAIL(<state>) wording."""
    ips = [ip for _, ip in targets]
    # State fetch is the slow part (nvidia-smi + CUDA probe over ssh): one
    # parallel pass for ALL targets before anything is launched, so a launch
    # decision is never made on a stale per-node read.
    states = run_parallel(ips, _safe_state)

    outcomes = [None] * len(targets)
    launch_positions = []
    for pos, ((idx, ip), state) in enumerate(zip(targets, states)):
        if state == "IDLE":
            launch_positions.append(pos)
        else:
            outcomes[pos] = (False, f"FAIL({state})")

    launched = []
    for pos in launch_positions:
        _, ip = targets[pos]
        try:
            # Fire-and-forget; the trainer must SURVIVE this process — see
            # the module docstring for why no cleanup is ever run here.
            manager.launch_trainer(ip)
            launched.append(pos)
        except Exception as exc:  # noqa: BLE001 — report, don't crash the run
            outcomes[pos] = (False, f"FAIL(launch error: {exc})")

    if launched:
        # ssh -f / docker exec -d return before the remote trainer process
        # exists (the GUI's exit cleanup works around the same race), so give
        # the launcher a moment, then confirm with pgrep — in parallel,
        # because sequential confirms would stack timeouts on dead nodes.
        time.sleep(5)
        confirmed = run_parallel([targets[pos][1] for pos in launched], _safe_check)
        for pos, up in zip(launched, confirmed):
            outcomes[pos] = ((True, "started, confirmed") if up else
                             (False, "started, not confirmed (check view.sh)"))

    return print_report(targets, outcomes)


def do_kill(targets):
    """Kill EVERY GPU process on the targets by running gpu_kill.sh — the
    blunt counterpart of release. The result line is the script's last
    stdout line (its one-line summary, e.g. "SIGTERM'd: 2, SIGKILL'd: 0");
    any non-zero rc means the node could not be reached or the script
    failed — reported as 'unreachable', like the GUI."""
    script = manager.node_path("UTILS", "nodes_monitor", "utils", "gpu_kill.sh")

    def kill_one(ip):
        try:
            rc, out = manager.run_node(ip, f"bash {script}", timeout=15)
        except Exception:  # noqa: BLE001 — same bucket as rc != 0
            return (False, "unreachable")
        if rc != 0:
            return (False, "unreachable")
        lines = out.strip().splitlines()
        # gpu_kill.sh always prints a summary line; "done" only guards a
        # hypothetical empty stdout so the report line is never blank.
        return (True, lines[-1] if lines else "done")

    return print_report(targets, run_parallel([ip for _, ip in targets], kill_one))


def do_release(targets):
    """Kill only OUR tagged trainer on the targets (pkill on the marker).
    rc 1 ('no tagged trainer') counts as SUCCESS: the post-condition this
    command exists to establish — no tagged trainer on the node — already
    holds. Only rc > 1 (transport/command failure) is a failure."""

    def release_one(ip):
        try:
            rc = manager.kill_trainer(ip)
        except Exception:  # noqa: BLE001 — same bucket as rc > 1
            return (False, "unreachable")
        if rc == 0:
            return (True, "killed tagged trainer")
        if rc == 1:
            return (True, "no tagged trainer")
        return (False, "unreachable")

    return print_report(targets, run_parallel([ip for _, ip in targets], release_one))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv=None):
    ap, args = parse_args(argv)

    # Deferred import — deliberately AFTER all argument validation: --help
    # and usage errors must print instantly, keep working even when the
    # backend module is mid-refactor, and never touch the network (importing
    # manager runs `hostname -I` at module scope). Bound as a module global
    # so the mode helpers above can use it.
    global manager
    import manager

    manager.load_config(args.config)
    manager.finalize_config()
    manager.load_nodes()

    if args.train is not None:
        mode, spec = do_train, args.train
    elif args.kill is not None:
        # Never False here: parse_args already errored on a bare --kill.
        mode, spec = do_kill, args.kill
    else:
        mode, spec = do_release, args.release

    targets = resolve_targets(ap, spec)
    sys.exit(0 if mode(targets) else 1)


if __name__ == "__main__":
    main()
