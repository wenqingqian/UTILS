#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nodes monitor workbench — GUI backend for the multi-node GPU monitor.

Started via `manager.sh --gui` (or directly: `python3 utils/manager/manager.py`).
Also importable as a module: manager_cli.py builds its one-shot CLI on the
config/node helpers below (load_config, finalize_config, load_nodes,
run_node, get_node_state, launch_trainer, check_trainer, kill_trainer,
is_excluded) — keep module import side-effect-free so that stays cheap.

A persistent control panel for the multi-node GPU monitor:

    python3 utils/manager/manager.py [--config FILE] [--exclude 1,3]

Commands at the `cmd` prompt:
    kill N | kill all | kill 1,2,3   kill every GPU process on the node(s)
    train [N|1,2]                    occupy node(s) NOW — only when idle
    try_train [N|1,2]                arm background occupation (auto-launch
                                     as soon as a node goes idle)
    release N | release all          stop try_train and kill only our
                                     tagged trainer processes
    theme                            open the interactive theme picker
    theme NAME                       table | icons | dashboard | status | neon |
                                     matrix | graph | cards | waves | weather | mono
    help                             show this help
    quit | exit | q                  leave the workbench

Architecture (why this is not bash):
  * A background thread collects node states; the main thread only reads the
    shared dict — keystrokes are never blocked by ssh/docker.
  * Input is a select() on stdin in cbreak mode; terminal restoration on exit
    is an explicit termios.tcsetattr (no hidden bash `read` restore).
  * Column alignment uses unicodedata.east_asian_width — exact terminal
    display width regardless of the shell's locale.
"""

import os
import re
import sys
import time
import signal
import shlex
import select
import argparse
import subprocess
import threading
import unicodedata
import termios
import tty
from collections import deque, defaultdict

try:
    import tomllib  # python >= 3.11
except ModuleNotFoundError:  # pragma: no cover
    sys.exit("manager.py needs python3 >= 3.11 (tomllib)")

# This file lives two levels deeper than the other tools (utils/manager/manager.py):
# SCRIPT_DIR must climb two levels to point at nodes_monitor/ and UTILS_ROOT at
# the repo root so the config and nodes.conf candidates below keep resolving
# to the same paths.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = os.path.dirname(os.path.dirname(_PKG_DIR))
UTILS_ROOT = os.path.dirname(SCRIPT_DIR)

# ---------------------------------------------------------------------------
# ANSI colors / styles
# ---------------------------------------------------------------------------
R = "\033[0m"
BOLD = "\033[1m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
GRAY = "\033[90m"
BRIGHT_GREEN = "\033[92m"

# ---------------------------------------------------------------------------
# Config / state
# ---------------------------------------------------------------------------
CONF = {}
IPS = []
EXCLUDE = set()          # 0-based indices to skip
STATS = {}               # idx -> (total, compute, state)
STATS_LOCK = threading.Lock()
PENDING_TRY = set()      # ips armed for try_train
LAST_TRY_LAUNCH = {}     # ip -> epoch of last launch attempt
TRY_FAILS = {}           # ip -> consecutive try_train launches never seen TRAIN
MESSAGE = ""
INPUT = ""
THEME = "table"
COLS = 80
FRAME_W = 0
OLD_TERM = None
QUIT = False

MARKER = "__UTILS_train_job__"
PGREP_PATTERN = None  # derived from the marker in finalize_config()
SSH_IDENTITY_PATH = ""  # ssh key path as THIS machine sees it (finalize_config)

# This repo's path RELATIVE TO the workspace, derived in finalize_config from
# UTILS_ROOT's real location — never assumed to be a bare "UTILS" (the repo
# may sit in a subdirectory of the workspace, e.g. <ws>/sga_framework/UTILS).
# The node-side path of anything in the repo is then NODE_ROOT/UTILS_WS_REL/...
# The "UTILS" default preserves the flat-layout answer for any node_path call
# made before finalize_config runs.
UTILS_WS_REL = "UTILS"

# Where the workspace is mounted for commands running ON A NODE: the container
# of the local node (docker exec) and the remote nodes (plain ssh) both see the
# directory as /workspace, while THIS host — where the tools run, and hence
# where the ssh client reads the key — sees it as [paths] workspace_root. The
# node-side mount point is the docker deployment's convention, not a config
# value; only the host view is configurable. Mirrors NODE_ROOT in monitor.sh.
NODE_ROOT = "/workspace"


def node_path(*parts):
    """Absolute path as seen by a command executed on a node (docker exec on
    the local node, ssh on the remote ones)."""
    return os.path.join(NODE_ROOT, *parts)


# Re-launch throttle for try_sweep: after an attempt on an ip, wait this long
# before arming another launch (a dying trainer must not be re-fired every
# INTERVAL). Deliberately NOT tied to the probe cache window (cooldown).
LAUNCH_COOLDOWN = 60
# Strike limit for try_sweep: this many consecutive launches on an ip without
# the trainer ever coming up disarm it with a warning. Guards the frozen
# healthy verdict against a node whose nvidia-smi works but whose CUDA
# contexts are wedged — the trainer would crash at init and be re-fired
# forever (the probe that would flag it BROKEN only runs at startup).
TRY_FAIL_LIMIT = 3

# Trainers launched by THIS session; cleanup() kills them on exit so no
# process is ever left behind on the nodes, however we end.
LAUNCHED = set()        # ip -> trainer launched (kill on exit)
LAUNCH_THREADS = []     # in-flight launch threads (join before the kill)
EDGE = ""               # ANSI color for the frame borders (per theme)
RESET = R               # color reset used by the frame code ("" for mono)
FRAME_GLYPHS = ("╭", "╮", "╰", "╯", "│", "─")   # tl tr bl br vl hl
ASCII_GLYPHS = ("+", "+", "+", "+", "|", "-")

# ---------------------------------------------------------------------------
# Width / alignment helpers (locale-independent)
# ---------------------------------------------------------------------------
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

# Box-drawing/symbol glyphs (─│╭╮●◆▲↻⊘…, EAW 'A') are assumed to render at
# 1 column, which virtually all terminals do. If a terminal renders them at
# 2 columns, borders will drift — that is a terminal-width assumption, not a
# bug in the layout math.
AMBIG_WIDTH = 1


def char_width(c):
    eaw = unicodedata.east_asian_width(c)
    if eaw in "WF":
        return 2
    if eaw == "A":
        return AMBIG_WIDTH
    return 1


def vis_width(s):
    """Terminal display width of s with ANSI codes stripped. Wide (CJK etc.)
    characters count 2 columns via unicodedata — independent of locale."""
    s = _ANSI_RE.sub("", s)
    return sum(char_width(c) for c in s)


def pad_to(s, width):
    """Right-pad (possibly colored) s to an exact visible width; over-wide
    content is truncated (with …) so borders can never be pushed apart."""
    if vis_width(s) > width:
        s = trunc_vis(s, width)
    return s + " " * max(0, width - vis_width(s))


def trunc_vis(s, width):
    """Truncate (possibly colored) s to `width` visible columns, keeping ANSI
    codes intact; when cut, one column is reserved for the trailing '…'."""
    if vis_width(s) <= width:
        return s
    out = []
    n = 0
    i = 0
    while i < len(s):
        m = _ANSI_RE.match(s, i)
        if m:
            out.append(m.group(0))
            i = m.end()
            continue
        c = s[i]
        w = char_width(c)
        if n + 1 + w > width:      # leave one column for the ellipsis
            break
        out.append(c)
        n += w
        i += 1
    return "".join(out) + RESET + "…"


def tail_vis(s, width):
    """Return the TAIL of plain-text s that fits into `width` visible columns
    — what a typist needs to see of an over-long command line."""
    w = 0
    i = len(s)
    while i > 0:
        cw = char_width(s[i - 1])
        if w + cw > width:
            break
        w += cw
        i -= 1
    return s[i:]


# ---------------------------------------------------------------------------
# Config loading (same resolution order as the shell version)
# ---------------------------------------------------------------------------
# Keys this program understands, per section. Anything else is warned about
# and ignored (a typo like `pgrep_patern` under [trainer] would otherwise
# silently keep the default — the operator must notice). Keep in sync with
# the case-statement in monitor.sh's load_config.
_KNOWN_KEYS = {
    "ssh": {"identity", "port"},
    "paths": {"workspace_root", "launcher_host"},
    "env": {"container"},
    "trainer": {"marker", "pgrep_pattern", "command", "probe_command"},
    "monitor": {"interval", "cooldown", "compute_threshold",
                "mem_used_threshold", "log_file"},
    # "cooldown" is accepted for schema parity with monitor.sh and controls
    # the view.sh BROKEN-cache trust window. The manager retains only its
    # confirmed BROKEN verdict in process memory and re-probes other outcomes.
    "ui": {"theme"},
    "nodes": {"list", "file"},
}


def load_config(cfg_file=None):
    global CONF, INTERVAL, COMPUTE_THRESHOLD, MEM_THRESHOLD, LOG_FILE
    candidates = []
    if cfg_file:
        candidates.append(cfg_file)
    else:
        candidates = [
            os.path.join(UTILS_ROOT, ".data", "nodes_monitor", "config.toml"),
            os.path.join(SCRIPT_DIR, "config.toml"),
        ]
    path = None
    for c in candidates:
        if os.path.isfile(c):
            path = c
            break
    if path is None:
        sys.exit(f"Error: no config.toml found (looked in {candidates})")
    with open(path, "rb") as fh:
        CONF = tomllib.load(fh)

    for section, table in CONF.items():
        known = _KNOWN_KEYS.get(section, set())
        if isinstance(table, dict):
            for key in table:
                if key not in known:
                    print(f"Warning: ignoring unknown config key: {section}.{key}",
                          file=sys.stderr)
        else:
            # A bare top-level key (no section) is not part of the schema.
            print(f"Warning: ignoring unknown config key: {section}",
                  file=sys.stderr)

    mon = CONF.get("monitor", {})
    ui = CONF.get("ui", {})
    INTERVAL = int(mon.get("interval", 5))
    COMPUTE_THRESHOLD = int(mon.get("compute_threshold", 0))
    MEM_THRESHOLD = int(mon.get("mem_used_threshold", 100))
    LOG_FILE = mon.get("log_file", "/tmp/utils_train.log")
    if "cooldown" in mon:
        # Accepted for schema parity but consumed only by monitor.sh's probe
        # cache — say so instead of letting an operator assume it throttles
        # try_sweep (it did in old versions).
        print("Note: monitor.cooldown is a view.sh probe-cache knob; the "
              "manager ignores it", file=sys.stderr)


# Called after load_config (by main() and by manager_cli.py): derive
# dependent values, enforce the required settings, and auto-fix the SSH key
# permissions (OpenSSH refuses keys readable by others). Mirrors
# finalize_config in monitor.sh.
def finalize_config():
    global PGREP_PATTERN, SSH_IDENTITY_PATH, UTILS_WS_REL
    ws = CONF.get("paths", {}).get("workspace_root", "")
    identity = CONF.get("ssh", {}).get("identity", "")
    container = CONF.get("env", {}).get("container", "")
    if not ws:
        sys.exit("Error: [paths] workspace_root is required")
    if not identity:
        sys.exit("Error: [ssh] identity is required")
    if not container:
        sys.exit("Error: [env] container is required")

    # ---- workspace path model ----
    # The same workspace directory has two names:
    #   host view — [paths] workspace_root: where this program (and hence the
    #               ssh client reading the key) runs;
    #   node view — NODE_ROOT: every command sent to a node runs in this view
    #               (docker exec on the local node, plain ssh on the remote
    #               ones).
    # [ssh] identity is stored RELATIVE to the workspace, so the key file is
    # <workspace_root>/<identity> here and <NODE_ROOT>/<identity> on a node.
    ws = ws.rstrip("/") or "/"
    if os.path.isabs(identity):
        sys.exit(f"Error: [ssh] identity must be relative to workspace_root "
                 f"(got '{identity}'; use e.g. identity = \"./cluster_ssh_key\")")
    host_identity = os.path.normpath(os.path.join(ws, identity))
    node_identity = os.path.normpath(os.path.join(NODE_ROOT, identity))
    # Resolve the key against THIS machine's view: normally the host view
    # exists; when the tools themselves run inside the container, only the node
    # view does.
    if os.path.isfile(host_identity):
        SSH_IDENTITY_PATH = host_identity
    elif os.path.isfile(node_identity):
        SSH_IDENTITY_PATH = node_identity
    else:
        sys.exit(f"Error: [ssh] identity file not found: tried {host_identity} "
                 f"and {node_identity}")

    # Node-side location of THIS repo: NODE_ROOT + the repo's path relative to
    # the workspace. Derived, never hardcoded — the repo may sit in a
    # subdirectory of the workspace (e.g. <ws>/sga_framework/UTILS), and a
    # fixed "UTILS" segment then resolves to a path that does not exist on any
    # node (probe/launcher fail with rc=2 and free nodes show UNVERIFIED).
    # Like the identity above, the derivation tries the host view first and
    # falls back to the node view for tools running inside the container.
    # realpath collapses symlinks on both sides so relpath compares the same
    # physical directories.
    utils_root = os.path.realpath(UTILS_ROOT)
    for root in (ws, NODE_ROOT):
        cand = os.path.relpath(utils_root, os.path.realpath(root))
        if cand != ".." and not cand.startswith("../"):
            UTILS_WS_REL = cand
            break
    else:
        sys.exit(f"Error: the UTILS repo ({utils_root}) is outside both "
                 f"[paths] workspace_root ({ws}) and the node workspace "
                 f"({NODE_ROOT}) — move it under the workspace")

    # Auto-fix SSH private key permissions: OpenSSH ignores keys that are
    # group/world-readable. Tighten to 600 if too open so auth does not fail.
    if os.stat(SSH_IDENTITY_PATH).st_mode & 0o077:
        try:
            os.chmod(SSH_IDENTITY_PATH, 0o600)
        except OSError:
            # Warn but do not die: the fs may be read-only / root-squashed.
            print(f"Warning: could not chmod 600 {SSH_IDENTITY_PATH} (read-only fs / "
                  f"root-squash?); OpenSSH may refuse the key", file=sys.stderr)

    # Keep the pgrep/pkill pattern in sync with the marker: derive it from
    # the marker when the config does not set it. The [X]first-char bracket
    # trick stops pgrep/pkill from matching their own command line.
    trainer = CONF.get("trainer", {})
    marker = trainer.get("marker", MARKER)
    if not marker:
        sys.exit("Error: [trainer] marker must not be empty")
    PGREP_PATTERN = trainer.get("pgrep_pattern") or ("[%s]%s" % (marker[0], marker[1:]))


# Same regex as monitor.sh's IP_RE — keep the two in sync.
_IP_RE = re.compile(
    r"^(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])"
    r"(\.(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])){3}$")
_NODE_RE = re.compile(
    r"^((?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])"
    r"(?:\.(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])){3})"
    r"(?::([1-9][0-9]{0,4}))?$")


def parse_node(spec):
    if not isinstance(spec, str):
        return None
    m = _NODE_RE.fullmatch(spec)
    if not m:
        return None
    port = int(m.group(2)) if m.group(2) else int(CONF.get("ssh", {}).get("port", 2222))
    if port > 65535:
        return None
    return m.group(1), port


def node_host(spec):
    parsed = parse_node(spec)
    return parsed[0] if parsed else spec.split(":", 1)[0]


def node_port(spec):
    parsed = parse_node(spec)
    return parsed[1] if parsed else int(CONF.get("ssh", {}).get("port", 2222))


def load_nodes():
    global IPS
    nodes = CONF.get("nodes", {})
    if nodes.get("list"):
        IPS = []
        for ip in nodes["list"]:
            # A typo (or a non-string TOML value) must die loudly here, not
            # as a cryptic ssh failure against a bogus address later.
            parsed = parse_node(ip)
            if parsed is None:
                sys.exit(f"Error: invalid node '{ip}' in [nodes] list (expected IP or IP:port)")
            IPS.append(ip)
    else:
        f = nodes.get("file") or os.path.join(SCRIPT_DIR, "nodes.conf")
        if not f.startswith("/"):
            f = os.path.join(UTILS_ROOT, f)
        try:
            with open(f) as fh:
                IPS = [ln.split("#")[0].strip() for ln in fh]
                IPS = [ln for ln in IPS if ln]
        except OSError as exc:
            sys.exit(f"Error: node list file not readable: {f} ({exc})")
        for ip in IPS:
            if parse_node(ip) is None:
                sys.exit(f"Error: invalid node '{ip}' in {f} (expected IP or IP:port)")
    if not IPS:
        sys.exit("Error: no nodes configured ([nodes] list or file)")


# ---------------------------------------------------------------------------
# Remote execution
# ---------------------------------------------------------------------------
def local_ips():
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True,
                             timeout=5).stdout.split()
        return set(out)
    except Exception:
        return set()


LOCAL_IPS = local_ips()


def is_local(ip):
    return node_host(ip) in LOCAL_IPS


def ssh_opts(ip=None):
    ssh = CONF.get("ssh", {})
    opts = [
        "-p", str(node_port(ip) if ip else ssh.get("port", 2222)),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=5",
        "-o", "ServerAliveInterval=5",
        "-o", "ServerAliveCountMax=3",
        "-o", "BatchMode=yes",
        "-o", "LogLevel=ERROR",
    ]
    if SSH_IDENTITY_PATH:
        opts += ["-i", SSH_IDENTITY_PATH]
    return opts


def run_node(ip, cmd, timeout=30):
    """Run cmd in the target environment; return (rc, stdout_text)."""
    if is_local(ip):
        try:
            p = subprocess.run(["docker", "exec", CONF.get("env", {}).get("container", ""),
                                "bash", "-c", cmd],
                               capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return (124, "")
        except (FileNotFoundError, OSError):
            return (127, "")
        return (p.returncode, p.stdout.decode("utf-8", "replace"))
    try:
        p = subprocess.run(["ssh", *ssh_opts(ip), node_host(ip), "bash -c " + shlex.quote(cmd)],
                           capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return (124, "")
    except (FileNotFoundError, OSError):
        return (127, "")
    return (p.returncode, p.stdout.decode("utf-8", "replace"))


# ---------------------------------------------------------------------------
# Node state
# ---------------------------------------------------------------------------
def fetch_gpu_data(ip):
    return run_node(
        ip,
        "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total "
        "--format=csv,noheader,nounits",
    )


def node_reachable(ip):
    return run_node(ip, "true")[0] == 0


def check_trainer(ip):
    return run_node(ip, f"pgrep -f '{PGREP_PATTERN}'")[0] == 0


def probe_cuda(ip):
    prefix = CONF.get("trainer", {}).get("probe_command", "")
    probe = node_path(UTILS_WS_REL, "nodes_monitor", "utils", "cuda_probe.py")
    # shlex.quote: the node-side path may contain spaces.
    return run_node(ip, f"{prefix}python3 {shlex.quote(probe)}")[0]


PROBE_TS = {}      # ip -> (epoch, verdict 0/1/2)
PROBE_FAILS = {}   # ip -> consecutive rc=1 failures
PROBE_FAIL_THRESHOLD = 3
PROBE_LOCKS = defaultdict(threading.Lock)   # one probe in flight per ip


def probe_cuda_cached(ip):
    """0 = healthy, 1 = confirmed BROKEN (PROBE_FAIL_THRESHOLD consecutive
    rc=1 failures), 2 = inconclusive (transport/environment failure — NOT a
    GPU verdict; callers must not treat it as BROKEN).

    The probe occupies GPU memory while it runs and is performed once per
    state observation, including on occupied nodes, so a wedged CUDA context
    cannot be hidden by USED. Only a confirmed BROKEN verdict is retained for
    the process lifetime; healthy and inconclusive results are refreshed on
    the next observation. The per-ip lock keeps concurrent callers from
    probing the same node twice."""
    if ip in PROBE_TS:
        ts, verdict = PROBE_TS[ip]
        # Only confirmed BROKEN is reusable. Healthy and inconclusive results
        # are per-observation values and must be re-probed on the next state
        # collection so recovery and newly wedged contexts are detectable.
        if verdict == 1:
            return verdict
    with PROBE_LOCKS[ip]:
        # Another thread may have probed while we waited for the lock.
        if ip in PROBE_TS:
            ts, verdict = PROBE_TS[ip]
            if verdict == 1:
                return verdict
        rc = probe_cuda(ip)
        if rc == 0:
            PROBE_FAILS[ip] = 0
            verdict = 0
        elif rc == 1:
            PROBE_FAILS[ip] = PROBE_FAILS.get(ip, 0) + 1
            verdict = 1 if PROBE_FAILS[ip] >= PROBE_FAIL_THRESHOLD else 2
        else:
            verdict = 2
        PROBE_TS[ip] = (time.time(), verdict)
    return verdict


def get_node_state(ip):
    """Return (total, compute, state)."""
    rc, data = fetch_gpu_data(ip)
    if rc != 0:
        return (-1, 0, "BROKEN" if node_reachable(ip) else "OFFLINE")
    if not data.strip():
        return (-1, 0, "OFFLINE")

    total = compute = mem = 0
    for line in data.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        util = int(parts[1]) if parts[1].isdigit() else 0
        used = int(parts[2]) if parts[2].isdigit() else 0
        total += 1
        if util > COMPUTE_THRESHOLD:
            compute += 1
        if used > MEM_THRESHOLD:
            mem += 1
    if total == 0:
        return (0, 0, "NO_GPU")
    if check_trainer(ip):
        trainer = True
    else:
        trainer = False

    # Probe health before classifying occupancy so a wedged CUDA context cannot
    # be hidden behind USED. A confirmed BROKEN verdict overrides occupancy;
    # inconclusive results preserve TRAIN/USED, while an otherwise-free node is
    # kept out of the IDLE pool as UNVERIFIED.
    verdict = probe_cuda_cached(ip)
    if verdict == 1:
        return (total, compute, "BROKEN")
    if trainer:
        return (total, compute, "TRAIN")
    if compute > 0 or mem > 0:
        return (total, compute, "USED")
    if verdict == 2:
        return (total, compute, "UNVERIFIED")
    return (total, compute, "IDLE")


# ---------------------------------------------------------------------------
# Trainer launch / kill (launch is fully async — never blocks the UI)
# ---------------------------------------------------------------------------
POPENS = []   # launched Popen objects (reaped opportunistically)


def _reap_popens():
    for p in POPENS[:]:
        if p.poll() is not None:
            POPENS.remove(p)


def _safe_thread(fn):
    """Wrap a background-thread target: exceptions are logged to a file and
    never printed to the terminal (a traceback would corrupt the full-screen
    UI and look like input problems)."""
    def wrapper(*args, **kwargs):
        try:
            fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            try:
                with open(LOG_FILE + ".err", "a") as fh:
                    fh.write(f"[{time.strftime('%H:%M:%S')}] {fn.__name__}: {exc}\n")
            except Exception:
                pass
    return wrapper


@_safe_thread
def launch_trainer(ip):
    trainer = CONF.get("trainer", {})
    prefix = trainer.get("command", "")
    marker = trainer.get("marker", MARKER)
    # [paths] launcher_host, when set, is an absolute NODE-side path (as seen
    # inside the container / on the compute nodes); the default is derived from
    # the node-side workspace mount point.
    launcher = CONF.get("paths", {}).get("launcher_host") or node_path(
        UTILS_WS_REL, "nodes_monitor", "utils", "run_train.sh")
    if is_local(ip):
        # shlex.quote the paths: the launcher path and LOG_FILE may contain
        # spaces, which would otherwise split into separate words on the
        # node-side shell.
        cmd = "{ %sTRAIN_TAG='%s' bash %s; } > %s 2>&1" % (
            prefix, marker, shlex.quote(launcher), shlex.quote(LOG_FILE))
        p = subprocess.Popen(["docker", "exec", "-d",
                              CONF.get("env", {}).get("container", ""),
                              "bash", "-c", cmd])
    else:
        # Same quoting as above; the quotes shlex.quote adds are single
        # quotes, which the `escaped` rewrite below already survives.
        inner = "%sTRAIN_TAG='%s' bash %s" % (prefix, marker, shlex.quote(launcher))
        escaped = inner.replace("'", "'\\''")
        cmd = "setsid bash -c '%s' > %s 2>&1 < /dev/null &" % (escaped, shlex.quote(LOG_FILE))
        p = subprocess.Popen(["ssh", "-f", *ssh_opts(ip), node_host(ip),
                              "bash -c " + shlex.quote(cmd)])
    POPENS.append(p)
    _reap_popens()
    # Tracked for exit-time cleanup. NOTE: ssh -f / docker exec -d return
    # before the remote trainer actually exists, so _kill_launched_trainers
    # re-checks a moment later (see there).
    LAUNCHED.add(ip)


def spawn_launch(ip):
    """Launch a trainer off the UI thread and remember the thread so exit
    cleanup can join it first (a kill racing an in-flight launch would miss
    the trainer)."""
    t = threading.Thread(target=launch_trainer, args=(ip,), daemon=True)
    t.start()
    LAUNCH_THREADS.append(t)


def kill_trainer(ip, timeout=15):
    return run_node(ip, f"pkill -f '{PGREP_PATTERN}'", timeout=timeout)[0]


def _kill_launched_trainers():
    """Best-effort, bounded: kill every trainer THIS session launched, so no
    process is ever left on the nodes regardless of how we exit. Runs in
    parallel threads with a global cap; stragglers die with the process."""
    for t in list(LAUNCH_THREADS):
        t.join(10)                     # in-flight launches finish first
    # A launch thread that hit the join timeout may still have an ssh -f /
    # docker exec wrapper in its auth/spawn phase; terminate it so nothing
    # of this session outlives the process. A wrapper killed here never
    # spawns the trainer, so the marker pkill below just finds nothing.
    for p in list(POPENS):
        if p.poll() is None:
            try:
                p.terminate()
            except Exception:          # never let cleanup itself fail
                pass
    ips = list(LAUNCHED)
    if not ips:
        return

    def round_kill():
        def kill_one(ip):
            try:
                kill_trainer(ip, timeout=6)
            except Exception:          # never let cleanup itself fail
                pass
        ts = [threading.Thread(target=kill_one, args=(ip,), daemon=True)
              for ip in ips]
        for t in ts:
            t.start()
        for t in ts:
            t.join(8)

    round_kill()
    # ssh -f / docker exec -d may still be spawning the trainer when the first
    # round ran; give it a moment and kill again so none slips through.
    time.sleep(3)
    round_kill()


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------
def get_cols():
    global COLS
    try:
        COLS = int(subprocess.run(["tput", "cols"],
                                  capture_output=True, text=True).stdout) or 80
    except Exception:
        COLS = 80


def frame_line(s=""):
    # Overlong content is truncated (with a trailing …) so the border can
    # never be pushed out of alignment.
    if vis_width(s) > FRAME_W - 4:
        s = trunc_vis(s, FRAME_W - 4)
    pad = FRAME_W - vis_width(s) - 2
    if pad < 0:
        pad = 0
    vl = FRAME_GLYPHS[4]
    sys.stdout.write(f"{EDGE}{vl}{RESET} {s}{' ' * pad} {EDGE}{vl}{RESET}\n")


def dash_fill(n):
    return FRAME_GLYPHS[5] * max(0, n)


def frame_title(s):
    tl, tr, hl = FRAME_GLYPHS[0], FRAME_GLYPHS[1], FRAME_GLYPHS[5]
    fill = FRAME_W - vis_width(s) - 3
    if fill < 1:
        fill = 1
    sys.stdout.write(f"{EDGE}{tl}{RESET}{hl} {s} {dash_fill(fill)}{EDGE}{tr}{RESET}\n")


def frame_bottom():
    bl, br = FRAME_GLYPHS[2], FRAME_GLYPHS[3]
    sys.stdout.write(f"{EDGE}{bl}{RESET}{dash_fill(FRAME_W)}{EDGE}{br}{RESET}\n")


def frame_prompt():
    global PROMPT_COL
    tl, tr, bl, br, vl, hl = FRAME_GLYPHS
    inner = FRAME_W - 8
    maxlen = inner - 2
    disp = tail_vis(INPUT, maxlen)
    pad = inner - 2 - vis_width(disp)
    top = dash_fill(inner - 7)
    bot = dash_fill(inner)
    cmd_label = f"{BOLD}cmd{RESET}" if THEME != "mono" else "cmd"
    frame_line(f"  {EDGE}{tl}{RESET}{hl} {cmd_label} {hl}{top}{EDGE}{tr}{RESET}")
    frame_line(f"  {EDGE}{vl}{RESET} {disp}{' ' * pad} {EDGE}{vl}{RESET}")
    frame_line(f"  {EDGE}{bl}{RESET}{bot}{EDGE}{br}{RESET}")
    # Caret: frame border "│ " (2) + inset "  " (2) + box border "│ " (2)
    # precede the text, so the text starts at column 7 (1-based).
    PROMPT_COL = 7 + vis_width(disp)


PROMPT_COL = 0


def refresh_prompt():
    """Redraw only the prompt's middle row (the caret's line)."""
    vl = FRAME_GLYPHS[4]
    inner = FRAME_W - 8
    maxlen = inner - 2
    disp = tail_vis(INPUT, maxlen)
    pad = inner - 2 - vis_width(disp)
    content = f"  {EDGE}{vl}{RESET} {disp}{' ' * pad} {EDGE}{vl}{RESET}"
    fpad = FRAME_W - vis_width(content) - 2
    if fpad < 0:
        fpad = 0
    sys.stdout.write("\033[G\033[2K")
    sys.stdout.write(f"{EDGE}{vl}{RESET} {content}{' ' * fpad} {EDGE}{vl}{RESET}")
    sys.stdout.write(f"\033[{7 + vis_width(disp)}G")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Theme system
# ---------------------------------------------------------------------------
def state_color(state):
    return {
        "IDLE": GREEN, "TRAIN": BLUE, "USED": YELLOW,
        "BROKEN": RED, "OFFLINE": RED, "NO_GPU": RED,
        "UNVERIFIED": GRAY,
    }.get(state, GRAY)


def state_icon(state):
    return {"IDLE": "●", "TRAIN": "▲", "USED": "◆",
            "OFFLINE": "✖", "BROKEN": "✖", "NO_GPU": "○"}.get(state, "?")


def node_markers(ip, state):
    m = []
    if state == "TRAIN":
        m.append(f"{BOLD}{BLUE}◆ owned{R}")
    if ip in PENDING_TRY:
        m.append(f"{BOLD}{YELLOW}↻ try{R}")
    if is_excluded(ip):
        m.append(f"{GRAY}⊘ excl{R}")
    return " ".join(m)


def compute_cell(total, compute, state):
    """(colored compute text, visible str) for the table cell."""
    if state in ("checking", "UNVERIFIED", "OFFLINE", "NO_GPU", "BROKEN"):
        return ("-", "-")
    text = f"{compute}/{total}"
    if state == "USED" and compute == 0:
        return (f"{RED}{text}{R}", text)
    return (text, text)


def status_common():
    pending = sorted(i + 1 for i, ip in enumerate(IPS) if ip in PENDING_TRY)
    owned = sorted(i + 1 for i, ip in enumerate(IPS)
                   if STATS.get(i, (0, 0, ""))[2] == "TRAIN")
    s = f"{YELLOW}try: [{','.join(map(str, pending))}]   owned: [{','.join(map(str, owned))}]{R}"
    if MESSAGE:
        s += f"  {GRAY}|{R}  {MESSAGE}"
    frame_line("  " + s)


# ---- theme: table ----
# Column widths: idx=3, ip=12, state=8, compute=8, marks=24. Every border
# segment is (cell width + 2 spaces); header/data rows are produced by the
# same pad_to calls, so the borders can never drift.
TABLE_TOP = "╭─────┬──────────────┬──────────┬──────────┬──────────────────────────╮"
TABLE_SEP = "├─────┼──────────────┼──────────┼──────────┼──────────────────────────┤"
TABLE_BOT = "╰─────┴──────────────┴──────────┴──────────┴──────────────────────────╯"


def theme_table_header():
    frame_line(f"  {GRAY}{TABLE_TOP}{R}")
    row = (f"  {GRAY}│{R} {pad_to(f'{BOLD}idx{R}', 3)} {GRAY}│{R} "
           f"{pad_to(f'{BOLD}ip{R}', 12)} {GRAY}│{R} "
           f"{pad_to(f'{BOLD}state{R}', 8)} {GRAY}│{R} "
           f"{pad_to(f'{BOLD}compute{R}', 8)} {GRAY}│{R} "
           f"{pad_to(f'{BOLD}marks{R}', 24)} {GRAY}│{R}")
    frame_line(row)
    frame_line(f"  {GRAY}{TABLE_SEP}{R}")


def theme_table_node(idx, ip):
    total, compute, state = STATS.get(idx - 1, (0, 0, "checking"))
    sc = state_color(state)
    ccell, _ = compute_cell(total, compute, state)
    marks = node_markers(ip, state)
    row = (f"  {GRAY}│{R} {pad_to(str(idx), 3)} {GRAY}│{R} "
           f"{pad_to(ip, 12)} {GRAY}│{R} "
           f"{pad_to(f'{sc}{state}{R}', 8)} {GRAY}│{R} "
           f"{pad_to(ccell, 8)} {GRAY}│{R} "
           f"{pad_to(marks, 24)} {GRAY}│{R}")
    frame_line(row)


def theme_table_status():
    frame_line(f"  {GRAY}{TABLE_BOT}{R}")
    frame_line("")
    status_common()


# ---- theme: icons ----
def theme_icons_header():
    frame_line(f"  {GRAY}{'─' * 62}{R}")


def theme_icons_node(idx, ip):
    total, compute, state = STATS.get(idx - 1, (0, 0, "checking"))
    sc = state_color(state)
    icon = state_icon(state)
    marks = node_markers(ip, state)
    ccell, _ = compute_cell(total, compute, state)
    frame_line(f"  {sc}{icon}{R} {GRAY}[{idx}]{R} {BOLD}{pad_to(ip, 12)}{R}  "
               f"{pad_to(f'{sc}{state}{R}', 10)}  {pad_to(ccell, 5)} gpu  {marks}")


def theme_icons_status():
    frame_line(f"  {GRAY}{'─' * 62}{R}")
    frame_line("")
    status_common()


# ---- theme: dashboard ----
def theme_dashboard_header():
    frame_line(f"  {GRAY}{'─' * 62}{R}")


def theme_dashboard_node(idx, ip):
    total, compute, state = STATS.get(idx - 1, (0, 0, "checking"))
    sc = state_color(state)
    icon = state_icon(state)
    marks = node_markers(ip, state)
    ccell, _ = compute_cell(total, compute, state)
    frame_line(f"  {sc}{icon}{R} {BOLD}[{idx}]{R} {BOLD}{pad_to(ip, 12)}{R}  "
               f"{pad_to(f'{sc}{state}{R}', 10)}  {pad_to(ccell, 5)}  {marks}")


def theme_dashboard_status():
    frame_line(f"  {GRAY}{'─' * 62}{R}")
    frame_line("")
    status_common()


# ---- theme: status ----
def theme_status_header():
    pass


def theme_status_node(idx, ip):
    total, compute, state = STATS.get(idx - 1, (0, 0, "checking"))
    sc = state_color(state)
    marks = node_markers(ip, state)
    ccell, _ = compute_cell(total, compute, state)
    frame_line(f"  {GRAY}[{idx}]{R} {BOLD}{pad_to(ip, 12)}{R}  {pad_to(f'{sc}{state}{R}', 10)}  {pad_to(ccell, 5)}  {marks}")


def theme_status_status():
    frame_line(f"  {GRAY}{'─' * 62}{R}")
    buckets = {"TRAIN": [], "IDLE": [], "USED": [], "DOWN": []}
    for i, ip in enumerate(IPS):
        state = STATS.get(i, (0, 0, ""))[2]
        if state in ("TRAIN", "IDLE", "USED"):
            buckets[state].append(i + 1)
        elif state in ("OFFLINE", "BROKEN", "NO_GPU", "UNVERIFIED"):
            buckets["DOWN"].append(i + 1)
    pend = [i + 1 for i, ip in enumerate(IPS) if ip in PENDING_TRY]
    frame_line(f"  {GREEN}idle:[{buckets['IDLE']}]{R} {YELLOW}used:[{buckets['USED']}]{R} "
               f"{BLUE}owned:[{buckets['TRAIN']}]{R} {RED}down:[{buckets['DOWN']}]{R} "
               f"{YELLOW}try:[{pend}]{R}")
    if MESSAGE:
        frame_line("  " + MESSAGE)


# ---- theme: neon ----
# Colored double-line inner separators, a per-node utilization bar, and a
# live GPU-occupancy sparkline (per-node history kept by node_worker).
HISTORY = {}          # ip -> deque of compute/total ratios (last N fetches)
SPARK = "▁▂▃▄▅▆▇█"     # 8-level sparkline blocks (oldest left)
HISTORY_LEN = 24


def _record_history(ip, total, compute):
    if total > 0:
        HISTORY.setdefault(ip, deque(maxlen=HISTORY_LEN)).append(compute / total)


def util_bar(total, compute, color):
    """8-cell occupancy bar: ██░░░░░░ — filled cells in the state color."""
    if total <= 0:
        return GRAY + "░" * 8 + R
    filled = round(8 * compute / total)
    return (color + "█" * filled + GRAY + "░" * (8 - filled) + R)


def sparkline(ip, width):
    # Snapshot under the lock: node_worker appends to HISTORY under
    # STATS_LOCK, and iterating a deque that is concurrently mutated
    # raises RuntimeError — which would crash render() and kill the TUI.
    with STATS_LOCK:
        h = list(HISTORY.get(ip) or ())
    if not h:
        return GRAY + "·" * width + R
    s = "".join(SPARK[min(7, int(r * 7.999))] for r in h)
    if len(s) > width:
        s = s[-width:]
    return CYAN + s.rjust(width, "·") + R


def theme_neon_header():
    frame_line(f"  {EDGE}╞{R}{'═' * 62}{EDGE}╡{R}")


def theme_neon_node(idx, ip):
    total, compute, state = STATS.get(idx - 1, (0, 0, "checking"))
    sc = state_color(state)
    icon = state_icon(state)
    marks = node_markers(ip, state)
    ccell, _ = compute_cell(total, compute, state)
    frame_line(f"  {CYAN}◈{R} {BOLD}{idx:02d}{R} {BOLD}{pad_to(ip, 12)}{R}  "
               f"{sc}{icon}{R} {pad_to(f'{sc}{state}{R}', 10)}  "
               f"{pad_to(ccell, 5)}  "
               f"{util_bar(total, compute, sc)}  {sparkline(ip, 12)}  {marks}")


def theme_neon_status():
    frame_line(f"  {EDGE}╞{R}{'═' * 62}{EDGE}╡{R}")
    frame_line("")
    status_common()
    # Cluster-wide occupancy summary.
    tot = comp = 0
    counts = {}
    for i, ip in enumerate(IPS):
        total, compute, state = STATS.get(i, (0, 0, "checking"))
        tot += total
        comp += compute
        counts[state] = counts.get(state, 0) + 1
    pct = (100 * comp // tot) if tot else 0
    frame_line(f"  {GREEN}● idle {counts.get('IDLE', 0)}{R}   "
               f"{YELLOW}◆ used {counts.get('USED', 0)}{R}   "
               f"{BLUE}▲ owned {counts.get('TRAIN', 0)}{R}   "
               f"{RED}✖ down {counts.get('OFFLINE', 0) + counts.get('BROKEN', 0) + counts.get('NO_GPU', 0) + counts.get('UNVERIFIED', 0)}{R}   "
               f"{MAGENTA}▊ GPU busy {pct}%{R}")


# ---- theme: matrix ----
# Green-phosphor CRT look: monochrome greens, block separators, sparkline.
def theme_matrix_header():
    frame_line(f"  {GREEN}{'▚' * 62}{R}")


def theme_matrix_node(idx, ip):
    total, compute, state = STATS.get(idx - 1, (0, 0, "checking"))
    sc = state_color(state)
    marks = node_markers(ip, state)
    ccell, _ = compute_cell(total, compute, state)
    frame_line(f"  {GREEN}▮{R} {BRIGHT_GREEN}[{idx:02d}]{R} {BOLD}{BRIGHT_GREEN}{pad_to(ip, 12)}{R}  "
               f"{pad_to(f'{sc}{state}{R}', 10)}  "
               f"{GREEN}{pad_to(ccell, 5)}{R}  {sparkline(ip, 14)}  {marks}")


def theme_matrix_status():
    frame_line(f"  {GREEN}{'▚' * 62}{R}")
    frame_line("")
    status_common()
    counts = _state_counts()
    frame_line(f"  {BRIGHT_GREEN}> idle:{counts['IDLE']} used:{counts['USED']} "
               f"owned:{counts['TRAIN']} down:{counts['DOWN']}{R}")


# ---- theme: graph ----
# htop-style utilization bars: one bar per node, plus a cluster stacked bar.
def _graph_bar(total, compute, color, width=30):
    if total <= 0:
        return GRAY + "░" * width + R
    filled = round(width * compute / total)
    return color + "█" * filled + GRAY + "░" * (width - filled) + R


def theme_graph_header():
    frame_line(f"  {GRAY}{'─' * 62}{R}")


def theme_graph_node(idx, ip):
    total, compute, state = STATS.get(idx - 1, (0, 0, "checking"))
    sc = state_color(state)
    marks = node_markers(ip, state)
    pct = f"{100 * compute // total:3d}%" if total > 0 else "   -"
    frame_line(f"  {GRAY}[{idx:02d}]{R} {BOLD}{pad_to(ip, 12)}{R}  {_graph_bar(total, compute, sc)} "
               f"{pct}  {pad_to(f'{sc}{state}{R}', 10)}  {marks}")


def theme_graph_status():
    frame_line(f"  {GRAY}{'─' * 62}{R}")
    frame_line("")
    status_common()
    # Cluster stacked bar: one colored cell per node.
    segs = []
    for i, _ip in enumerate(IPS):
        state = STATS.get(i, (0, 0, ""))[2]
        if state in ("TRAIN", "IDLE", "USED"):
            segs.append(state_color(state) + "█" + R)
        elif state in ("OFFLINE", "BROKEN", "NO_GPU"):
            segs.append(RED + "█" + R)
        else:
            segs.append(GRAY + "░" + R)
    frame_line("  cluster: " + "".join(segs) +
               f"   {GREEN}█=idle{R} {YELLOW}█=used{R} {BLUE}█=owned{R} "
               f"{RED}█=down{R} {GRAY}░=unver{R}")


# ---- theme: cards ----
# Dashboard widgets: each node in its own rounded card, two per row.
CARD_W = 46
CARD_BUF = []


def _chip(text):
    """Inverse-video chip (fg/bg swapped) — renders like a filled pill."""
    return f"\033[7m{text}\033[0m"


def _build_card(idx, ip):
    total, compute, state = STATS.get(idx - 1, (0, 0, "checking"))
    sc = state_color(state)
    icon = state_icon(state)
    marks = node_markers(ip, state)
    ccell, _ = compute_cell(total, compute, state)
    title = f"{BOLD}{idx:02d}{R} {BOLD}{ip}{R}"
    w = CARD_W
    top = f"╭─ {title} " + "─" * max(0, w - 5 - vis_width(title)) + "╮"
    inner = f"{sc}{icon}{R} {_chip(f'{sc}{pad_to(state, 10)}{R}')}  {pad_to(ccell, 5)}  {marks}"
    body = f"│ {pad_to(inner, w - 4)} │"
    bottom = f"╰{'─' * (w - 2)}╯"
    return (top, body, bottom)


def _emit_cards():
    per = 2 if FRAME_W >= 106 else 1
    while len(CARD_BUF) >= per:
        for r in range(3):
            frame_line("  " + "  ".join(c[r] for c in CARD_BUF[:per]))
        del CARD_BUF[:per]


def theme_cards_header():
    pass


def theme_cards_node(idx, ip):
    CARD_BUF.append(_build_card(idx, ip))
    _emit_cards()


def theme_cards_status():
    if CARD_BUF:                   # odd remainder: flush it alone
        for r in range(3):
            frame_line("  " + "  ".join(c[r] for c in CARD_BUF))
        CARD_BUF.clear()
    frame_line("")
    status_common()


# ---- theme: waves ----
# Oscilloscope: block-letter banner, the wide sparkline as the star, and a
# cluster-wide waveform in the status line.
FONT = {
    "N": ("█  █", "██ █", "█ ██"),
    "O": ("█████", "█   █", "█████"),
    "D": ("████ ", "█   █", "████ "),
    "E": ("█████", "████ ", "█████"),
    "S": ("█████", "█    ", "█████"),
    "M": ("█   █", "██ ██", "█ █ █"),
    "I": ("  █  ", "  █  ", "  █  "),
    "T": ("█████", "  █  ", "  █  "),
    "R": ("████ ", "█   █", "███  "),
    " ": ("     ", "     ", "     "),
}
BANNER_PALETTE = [CYAN, MAGENTA, YELLOW, GREEN, BLUE]


def _block_banner(text):
    rows = ["", "", ""]
    for i, ch in enumerate(text):
        col = BANNER_PALETTE[i % len(BANNER_PALETTE)]
        for r in range(3):
            rows[r] += col + FONT[ch][r] + R + " "
    return rows


def theme_waves_header():
    for row in _block_banner("NODES MONITOR"):
        frame_line("  " + row)


def theme_waves_node(idx, ip):
    total, compute, state = STATS.get(idx - 1, (0, 0, "checking"))
    sc = state_color(state)
    icon = state_icon(state)
    marks = node_markers(ip, state)
    ccell, _ = compute_cell(total, compute, state)
    frame_line(f"  {sc}{icon}{R} {BOLD}{pad_to(ip, 12)}{R}  {sparkline(ip, 24)}  "
               f"{pad_to(f'{sc}{state}{R}', 10)}  {pad_to(ccell, 5)}  {marks}")


def _cluster_wave(width=24):
    """Cluster-busy waveform: per-sample averages of all per-node histories,
    aligned from the tail (all workers sample on the same INTERVAL)."""
    with STATS_LOCK:
        seqs = [list(h) for h in HISTORY.values()]
    if not seqs:
        return GRAY + "·" * width + R
    n = max(len(s) for s in seqs)
    out = []
    for k in range(n):
        i = n - 1 - k
        vals = [s[len(s) - 1 - i] for s in seqs if i < len(s)]
        out.append(sum(vals) / len(vals))
    s = "".join(SPARK[min(7, int(r * 7.999))] for r in out)
    if len(s) > width:
        s = s[-width:]
    return MAGENTA + s.rjust(width, "·") + R


def theme_waves_status():
    frame_line(f"  {CYAN}{'~' * 62}{R}")
    frame_line("")
    status_common()
    frame_line(f"  {CYAN}cluster busy{R}  {_cluster_wave()}")


# ---- theme: weather ----
# Emoji weather forecast. Every icon is EAW 'W' (guaranteed 2 columns), so
# the columns never drift even if the terminal's emoji font differs.
W_ICON = {"IDLE": "🔆", "USED": "💤", "TRAIN": "⚡", "BROKEN": "🔥",
          "OFFLINE": "🌑", "NO_GPU": "🪫", "UNVERIFIED": "⏳", "checking": "🌐"}


def theme_weather_header():
    frame_line(f"  {GRAY}{'·' * 62}{R}")


def theme_weather_node(idx, ip):
    total, compute, state = STATS.get(idx - 1, (0, 0, "checking"))
    sc = state_color(state)
    marks = node_markers(ip, state)
    icon = W_ICON.get(state, W_ICON["checking"])
    ccell, _ = compute_cell(total, compute, state)
    frame_line(f"  {icon} {GRAY}[{idx:02d}]{R} {BOLD}{pad_to(ip, 12)}{R}  {pad_to(f'{sc}{state}{R}', 10)}  {pad_to(ccell, 5)}  {marks}")


def theme_weather_status():
    frame_line(f"  {GRAY}{'·' * 62}{R}")
    frame_line("")
    status_common()
    counts = _state_counts()
    frame_line(f"  {W_ICON['IDLE']} {counts['IDLE']} idle   "
               f"{W_ICON['USED']} {counts['USED']} busy   "
               f"{W_ICON['TRAIN']} {counts['TRAIN']} owned   "
               f"{W_ICON['BROKEN']} {counts['DOWN']} down")


# ---- theme: mono ----
# Pure ASCII, zero ANSI colors: renders correctly on any terminal, in logs,
# or over `script` replay. Control sequences (clear screen, cursor) remain —
# they are unavoidable for any full-screen TUI.
def theme_mono_header():
    frame_line("  +" + "-" * 62 + "+")


def theme_mono_node(idx, ip):
    total, compute, state = STATS.get(idx - 1, (0, 0, "checking"))
    marks = []
    if state == "TRAIN":
        marks.append("[owned]")
    if ip in PENDING_TRY:
        marks.append("[try]")
    if is_excluded(ip):
        marks.append("[excl]")
    m = " ".join(marks)
    if state in ("checking", "UNVERIFIED", "OFFLINE", "NO_GPU", "BROKEN"):
        ccell = "-"
    else:
        ccell = f"{compute}/{total}"
    frame_line(f"  | {idx:>2}  {ip:<12} {state:<10} {ccell:<5} {m} |")


def theme_mono_status():
    frame_line("  +" + "-" * 62 + "+")
    frame_line("")
    pending = [i + 1 for i, ip in enumerate(IPS) if ip in PENDING_TRY]
    owned = [i + 1 for i, ip in enumerate(IPS) if STATS.get(i, (0, 0, ""))[2] == "TRAIN"]
    line = f"  try: [{','.join(map(str, pending))}]   owned: [{','.join(map(str, owned))}]"
    if MESSAGE:
        line += "  |  " + _ANSI_RE.sub("", MESSAGE)
    frame_line(line)


def _state_counts():
    """Cluster-wide state buckets for the summary lines."""
    counts = {"IDLE": 0, "USED": 0, "TRAIN": 0, "DOWN": 0}
    for i, _ip in enumerate(IPS):
        state = STATS.get(i, (0, 0, ""))[2]
        if state in ("TRAIN", "IDLE", "USED"):
            counts[state] += 1
        elif state in ("OFFLINE", "BROKEN", "NO_GPU", "UNVERIFIED"):
            counts["DOWN"] += 1
    return counts


THEMES = {
    "table": (theme_table_header, theme_table_node, theme_table_status, ""),
    "icons": (theme_icons_header, theme_icons_node, theme_icons_status, ""),
    "dashboard": (theme_dashboard_header, theme_dashboard_node, theme_dashboard_status, ""),
    "status": (theme_status_header, theme_status_node, theme_status_status, ""),
    "neon": (theme_neon_header, theme_neon_node, theme_neon_status, CYAN),
    "matrix": (theme_matrix_header, theme_matrix_node, theme_matrix_status, GREEN),
    "graph": (theme_graph_header, theme_graph_node, theme_graph_status, ""),
    "cards": (theme_cards_header, theme_cards_node, theme_cards_status, ""),
    "waves": (theme_waves_header, theme_waves_node, theme_waves_status, CYAN),
    "weather": (theme_weather_header, theme_weather_node, theme_weather_status, ""),
    "mono": (theme_mono_header, theme_mono_node, theme_mono_status, ""),
}

THEME_DESC = {
    "table": "classic bordered table",
    "icons": "compact icon rows",
    "dashboard": "dense one-line nodes",
    "status": "minimal + bucket summary",
    "neon": "cyan frame, bars & sparkline",
    "matrix": "green phosphor CRT",
    "graph": "htop-style utilization bars",
    "cards": "rounded dashboard cards",
    "waves": "block banner + big waveform",
    "weather": "emoji weather forecast",
    "mono": "pure ASCII, no colors",
}


def theme_picker():
    """Full-screen interactive theme selection (like help): ↑↓/j k to move,
    Enter to apply, Esc/q to cancel. The selected theme's node row is shown
    live below the list. The screen is painted once at entry and repainted
    only when the selection actually changes (or after a consumed escape
    sequence such as a focus event) — idle select timeouts just re-poll
    stdin, so an untouched picker emits zero traffic over slow SSH links
    (same conditional-render discipline as the main loop)."""
    global THEME, FRAME_W
    get_cols()
    FRAME_W = COLS - 2
    names = list(THEMES)
    sel = names.index(THEME) if THEME in names else 0
    current = THEME
    redraw = True
    while True:
        # A signal (SIGHUP etc.) sets QUIT with nobody to consume a keystroke:
        # hand control back to the main loop so its teardown path runs.
        if QUIT:
            return
        if redraw:
            sys.stdout.write("\033[2J\033[H\033[?25l")
            framed_title("● nodes monitor workbench — theme")
            frame_line("")
            for i, name in enumerate(names):
                if i == sel:
                    row = f"  {CYAN}▸{R} " + pad_to(f"\033[7m{BOLD}{name}{R}\033[0m", 11)
                else:
                    row = "    " + pad_to(f"{BOLD}{name}{R}", 11)
                if name == current:
                    row += f"{GREEN}(current){R} "
                row += f"{GRAY}{THEME_DESC[name]}{R}"
                frame_line(row)
            frame_line("")
            # live preview: the selected theme's first node row, current data
            idx = 1
            ip = IPS[0] if IPS else "10.0.0.1"
            if names[sel] == "cards":
                CARD_BUF.clear()
                for row in _build_card(idx, ip):
                    frame_line("  " + row)
            else:
                THEMES[names[sel]][1](idx, ip)
            frame_line("")
            frame_line(f"{GRAY}↑↓ / j k  move    Enter  apply    Esc / q  cancel{R}")
            frame_bottom()
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()
            redraw = False

        r, _, _ = select.select([sys.stdin], [], [], 0.2)
        if not r:
            continue
        ch = read_char()
        if ch is None:               # stdin EOF
            cleanup()
            return
        if ch == "\x1b":
            k = read_escape(timeout=0.15)   # SSH can split ESC and [B across packets
            if k == "UP":
                sel = (sel - 1) % len(names)
                redraw = True
            elif k == "DOWN":
                sel = (sel + 1) % len(names)
                redraw = True
            elif k is None:
                return               # lone ESC cancels
            else:
                # Recognized but unmapped sequence (focus event etc.): the
                # terminal may have changed under us (e.g. re-focused) — keep
                # the picker open and repaint, like the old code did.
                redraw = True
            continue
        if ch in ("\r", "\n"):
            THEME = names[sel]
            set_message(f"theme → {THEME}")
            return
        if ch in ("q", "Q", "\x7f"):
            return
        if ch in ("k", "K"):
            sel = (sel - 1) % len(names)
            redraw = True
        elif ch in ("j", "J"):
            sel = (sel + 1) % len(names)
            redraw = True
        # any other key is ignored: no repaint


def framed_title(text):
    """Title for the current theme — plain (no ANSI) for mono."""
    if THEME == "mono":
        frame_title(f"{text}  {len(IPS)} nodes  {time.strftime('%H:%M')}  theme:{THEME}")
    else:
        title_color = {"neon": MAGENTA, "matrix": BRIGHT_GREEN, "waves": MAGENTA}.get(THEME, CYAN)
        frame_title(f"{BOLD}{title_color}{text}{R}  "
                    f"{GRAY}{len(IPS)} nodes  {time.strftime('%H:%M')}  theme:{THEME}{R}")


def render():
    global FRAME_W, LAST_RENDER, RENDER_SIG, RENDER_MIN, EDGE, RESET, FRAME_GLYPHS
    FRAME_W = COLS - 2
    EDGE = THEMES[THEME][3]
    RESET = "" if THEME == "mono" else R
    FRAME_GLYPHS = ASCII_GLYPHS if THEME == "mono" else ("╭", "╮", "╰", "╯", "│", "─")
    sys.stdout.write("\033[2J\033[H\033[?25l")
    framed_title("● nodes monitor workbench")
    frame_line("")
    header, node, status, _ = THEMES[THEME]
    header()
    for i, ip in enumerate(IPS):
        node(i + 1, ip)
    status()
    frame_line("")
    frame_prompt()
    frame_bottom()
    # caret onto the prompt's middle row, right after the input text
    sys.stdout.write(f"\033[3A\033[{PROMPT_COL}G")
    sys.stdout.write("\033[?25h")
    sys.stdout.flush()
    LAST_RENDER = time.time()
    RENDER_SIG = stats_sig()
    RENDER_MIN = time.localtime().tm_min


def show_help():
    sys.stdout.write("\033[2J\033[H")
    framed_title("● nodes monitor workbench — commands")
    frame_line("")
    for line in [
        "  kill N | kill all | kill 1,2,3   kill every GPU process on the node(s)",
        "                                   (everything nvidia-smi lists)",
        "  train [N|1,2]                    occupy node(s) now — only when idle;",
        "                                   non-idle nodes report FAIL(state)",
        "  try_train [N|1,2]                arm background occupation: node(s)",
        "                                   occupied as soon as they go idle",
        "  release N | release all          stop try_train and kill only our",
        "                                   tagged trainer processes",
        "  theme                             open the interactive theme picker",
        "  theme NAME                       switch directly: table, icons, dashboard,",
        "                                   status, neon, matrix, graph, cards,",
        "                                   waves, weather, mono",
        "  help                             show this help",
        "  quit | exit | q                  leave the workbench (Ctrl-C / Ctrl-D too)",
    ]:
        frame_line(line)
    frame_line("")
    frame_line(f"{GRAY}node indices are 1-based positions in the node list{R}")
    frame_line("")
    frame_line(f"{GRAY}press any key to return{R}")
    frame_bottom()
    sys.stdout.flush()
    _wait_key(60)


def _wait_key(timeout):
    """Block until one key (or timeout); drain all immediately-available
    bytes so a multi-byte key sequence never leaks into the main loop.
    Polls in small slices so a Ctrl-C / quit during help is honored within
    ~0.2s instead of holding the main loop for the whole timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline and not QUIT:
        r, _, _ = select.select([sys.stdin], [], [], 0.2)
        if r:
            while True:
                r2, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not r2:
                    break
                b = os.read(sys.stdin.fileno(), 1)
                if not b:
                    return          # stdin EOF: hand back to the main loop's EOF handling
            return


# ---------------------------------------------------------------------------
# Command handling
# ---------------------------------------------------------------------------
def set_message(msg):
    global MESSAGE
    # len() counts ANSI escapes, so a colored message would be cut far short
    # of 70 READABLE chars — measure and cut visible chars only, never
    # slicing through an escape sequence.
    if len(_ANSI_RE.sub("", msg)) <= 70:
        MESSAGE = msg
        return
    out = []
    n = 0
    i = 0
    while i < len(msg) and n < 67:
        m = _ANSI_RE.match(msg, i)
        if m:
            out.append(m.group(0))      # ANSI codes ride along for free
            i = m.end()
        else:
            out.append(msg[i])
            n += 1
            i += 1
    MESSAGE = "".join(out) + R + "..."  # R: don't bleed a cut color onward


def ip_index(ip):
    return IPS.index(ip) + 1 if ip in IPS else 0


def is_excluded(ip):
    return ip in IPS and IPS.index(ip) in EXCLUDE


def parse_targets(spec):
    if not spec:
        return [ip for ip in IPS if not is_excluded(ip)]
    if spec == "all":
        return [ip for ip in IPS if not is_excluded(ip)]
    targets = []
    for part in spec.split(","):
        part = part.strip()
        if not part.isdigit():
            set_message(f"{RED}invalid index '{part}'{R} (expect: N | all | 1,2,3)")
            return None
        idx = int(part) - 1
        if not (0 <= idx < len(IPS)):
            set_message(f"{RED}index out of range: {part}{R} (1..{len(IPS)})")
            return None
        targets.append(IPS[idx])
    return targets


def cmd_kill(spec):
    if not spec:
        set_message(f"{YELLOW}usage: kill N | kill all | kill 1,2,3{R}")
        return
    targets = parse_targets(spec)
    if targets is None:
        return
    set_message(f"{YELLOW}kill: working…{R}")
    render()
    script = node_path(UTILS_WS_REL, "nodes_monitor", "utils", "gpu_kill.sh")

    def work():
        try:
            acc = []
            for ip in targets:
                rc, out = run_node(ip, f"bash {shlex.quote(script)}", timeout=10)
                last = "unreachable" if rc != 0 else (out.strip().splitlines() or [""])[-1]
                acc.append(f"{ip_index(ip)}:{last}")
            set_message(f"kill → {' '.join(acc)}")
        except Exception as exc:  # noqa: BLE001
            set_message(f"{RED}kill failed: {exc}{R}")
        global RENDER_NOW
        RENDER_NOW = True

    # Async: slow/unreachable nodes must never freeze the UI.
    threading.Thread(target=work, daemon=True).start()


def cmd_train(spec):
    targets = parse_targets(spec) if spec else [ip for ip in IPS if not is_excluded(ip)]
    if targets is None:
        return
    acc = []
    for ip in targets:
        idx = ip_index(ip)
        state = STATS.get(idx - 1, (0, 0, ""))[2]
        if state == "IDLE":
            spawn_launch(ip)
            PENDING_TRY.discard(ip)
            acc.append(f"{idx}:started")
        else:
            acc.append(f"{idx}:FAIL({state})")
    set_message(f"train → {' '.join(acc)}")


def cmd_try_train(spec):
    targets = parse_targets(spec) if spec else [ip for ip in IPS if not is_excluded(ip)]
    if targets is None:
        return
    acc = []
    for ip in targets:
        idx = ip_index(ip)
        state = STATS.get(idx - 1, (0, 0, ""))[2]
        if ip in PENDING_TRY:
            acc.append(f"{idx}:already-armed")
        elif state == "TRAIN":
            acc.append(f"{idx}:already-owned")
        else:
            PENDING_TRY.add(ip)
            TRY_FAILS.pop(ip, None)   # re-arming starts a fresh strike budget
            acc.append(f"{idx}:armed({'idle' if state == 'IDLE' else 'waiting ' + state})")
    set_message(f"try_train → {' '.join(acc)}")


def cmd_release(spec):
    if not spec:
        set_message(f"{YELLOW}usage: release N | release all{R}")
        return
    targets = parse_targets(spec)
    if targets is None:
        return
    set_message(f"{YELLOW}release: working…{R}")
    render()

    def work():
        try:
            acc = []
            for ip in targets:
                idx = ip_index(ip)
                state = STATS.get(idx - 1, (0, 0, ""))[2]
                rel = ""
                if ip in PENDING_TRY:
                    PENDING_TRY.discard(ip)
                    rel = "disarmed"
                if state == "OFFLINE":
                    rel += (" " if rel else "") + "offline"
                else:
                    rc = kill_trainer(ip)
                    if rc == 0:
                        rel += (" " if rel else "") + "killed tagged trainer"
                    elif rc == 1:
                        rel += (" " if rel else "") + "no tagged trainer"
                    else:
                        rel += (" " if rel else "") + "unreachable"
                acc.append(f"{idx}:{rel}")
            set_message(f"release → {' '.join(acc)}")
        except Exception as exc:  # noqa: BLE001
            set_message(f"{RED}release failed: {exc}{R}")
        global RENDER_NOW
        RENDER_NOW = True

    threading.Thread(target=work, daemon=True).start()


def cmd_theme(spec):
    global THEME
    spec = spec.strip()
    if not spec:
        theme_picker()
    elif spec in THEMES:
        THEME = spec
        set_message(f"theme → {THEME}")
    else:
        set_message(f"{RED}unknown theme '{spec}'{R} — see help")


def handle_command(input_text):
    cmd, _, rest = input_text.partition(" ")
    rest = rest.strip()
    if cmd in ("h", "help"):
        show_help()
    elif cmd in ("quit", "exit", "q"):
        cleanup()
    elif cmd == "kill":
        cmd_kill(rest)
    elif cmd == "train":
        cmd_train(rest)
    elif cmd == "try_train":
        cmd_try_train(rest)
    elif cmd == "release":
        cmd_release(rest)
    elif cmd == "theme":
        cmd_theme(rest)
    elif cmd:
        set_message(f"{RED}unknown command '{cmd}'{R} — type 'help'")


# ---------------------------------------------------------------------------
# Keyboard input (cbreak; read UTF-8 chars with a pending-byte buffer)
# ---------------------------------------------------------------------------
# Bytes read ahead but not yet consumed as a character (e.g. an ESC sequence
# that turned out to be a lone ESC followed by a normal key) are parked here
# so they are never lost — this is what prevents "swallowed" keystrokes.
PENDING_IN = b""


def read_char():
    """Read one UTF-8 character from stdin. Never blocks on continuation
    bytes (a partial sequence is dropped after a short deadline instead of
    freezing the UI), and never loses bytes: lookahead stays in PENDING_IN.
    Returns None on EOF."""
    global PENDING_IN
    if not PENDING_IN:
        b0 = os.read(sys.stdin.fileno(), 1)
        if not b0:
            return None
        PENDING_IN = b0
    c = PENDING_IN[0]
    need = 1
    if (c & 0xE0) == 0xC0:
        need = 2
    elif (c & 0xF0) == 0xE0:
        need = 3
    elif (c & 0xF8) == 0xF0:
        need = 4
    if need == 1:
        PENDING_IN = PENDING_IN[1:]
        return chr(c)
    deadline = time.time() + 0.05
    while len(PENDING_IN) < need and time.time() < deadline:
        r, _, _ = select.select([sys.stdin], [], [], 0.02)
        if not r:
            continue
        try:
            PENDING_IN += os.read(sys.stdin.fileno(), need - len(PENDING_IN))
        except OSError:
            break
    if len(PENDING_IN) < need:
        # Incomplete multi-byte sequence: drop just the lead byte; whatever
        # arrives later is a fresh character, never swallowed.
        PENDING_IN = PENDING_IN[1:]
        return ""
    buf = PENDING_IN[:need]
    PENDING_IN = PENDING_IN[need:]
    try:
        return buf.decode("utf-8")
    except UnicodeDecodeError:
        return ""


def read_escape(timeout=0.15):
    """After an ESC char: consume the rest of a key sequence and return its
    name ("UP"/"DOWN"/"LEFT"/"RIGHT") or "OTHER" for any other recognized
    sequence (focus/mouse events etc.). Returns None ONLY for a lone ESC
    (no introducer within `timeout`) — callers may cancel on that.

    The byte right after ESC is an INTRODUCER, not a final byte: '[' (CSI)
    is drained until a final byte (0x40-0x7E); 'O' (SS3, e.g. F-keys) takes
    exactly one more byte. Any other byte is NOT part of a sequence (lone
    ESC followed by a fast keystroke) — it goes back into PENDING_IN so it
    is not lost. `timeout` is how long to wait for the introducer: the
    input line now tolerates split sequences like the theme picker does —
    over a slow SSH link ESC and the rest of the sequence can arrive in
    separate packets, and the old 0.05s input-line timeout leaked the
    leftover bytes (e.g. the "[A" of an Up arrow) into the command line."""
    global PENDING_IN
    deadline = time.time() + timeout
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    if not r:
        return None                     # lone ESC
    b = os.read(sys.stdin.fileno(), 1)
    if not b:
        return None
    if b == b"[":
        while time.time() < deadline:
            r, _, _ = select.select([sys.stdin], [], [], 0.02)
            if not r:
                continue
            b = os.read(sys.stdin.fileno(), 1)
            if not b:
                return None
            if 0x40 <= b[0] <= 0x7E:   # CSI final byte
                return {"A": "UP", "B": "DOWN", "C": "RIGHT", "D": "LEFT"}.get(chr(b[0]), "OTHER")
    elif b == b"O":
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        if r:
            os.read(sys.stdin.fileno(), 1)
        return "OTHER"
    else:
        PENDING_IN = b + PENDING_IN   # give the byte back to read_char
        return None
    return None


def handle_key(ch):
    global INPUT, QUIT
    if ch in ("\r", "\n"):
        handle_command(INPUT)
        INPUT = ""
        render()
    elif ch in ("\x7f", "\b"):
        if INPUT:
            INPUT = INPUT[:-1]
            refresh_prompt()          # backspace: redraw (multi-byte safe)
    elif ch in ("\x03", "\x04"):       # Ctrl-C / Ctrl-D
        cleanup()
    elif ch == "\x15":                 # Ctrl-U
        INPUT = ""
        refresh_prompt()
    elif ch == "\x09":                 # Tab
        pass
    elif ch == "\x1b":                 # escape sequence (arrow keys etc.)
        read_escape()                  # drained; the input line has no arrows
    elif 32 <= ord(ch) <= 126 or ord(ch) >= 160:
        INPUT += ch
        inner = FRAME_W - 8
        maxlen = inner - 2
        if vis_width(INPUT) <= maxlen:
            # Not overflowing: the caret is already right after the input
            # text, so just echo the character — zero redraw, no flicker,
            # no per-keystroke terminal traffic (the fix for input lag on
            # slow/SSH terminals).
            sys.stdout.write(ch)
            sys.stdout.flush()
        else:
            refresh_prompt()          # overflow: redraw the visible tail


# ---------------------------------------------------------------------------
# Background state collection (one thread per node — the main thread never
# blocks on ssh/docker; a slow or failing node never stalls the others)
# ---------------------------------------------------------------------------
def node_worker(i, ip):
    while True:
        try:
            st = get_node_state(ip)
        except Exception:
            st = None          # keep last known state; never print to the UI
        if st is not None:
            with STATS_LOCK:
                STATS[i] = st
                _record_history(ip, st[0], st[1])
        time.sleep(INTERVAL)


def try_sweep():
    """Auto-launch armed (try_train) nodes that went IDLE."""
    now = time.time()
    for ip in list(PENDING_TRY):
        idx = ip_index(ip)
        if idx == 0:
            continue
        state = STATS.get(idx - 1, (0, 0, ""))[2]
        if state == "TRAIN":
            PENDING_TRY.discard(ip)
            TRY_FAILS.pop(ip, None)
        elif state == "IDLE":
            if now - LAST_TRY_LAUNCH.get(ip, 0) > LAUNCH_COOLDOWN:
                if TRY_FAILS.get(ip, 0) >= TRY_FAIL_LIMIT:
                    PENDING_TRY.discard(ip)
                    TRY_FAILS.pop(ip, None)
                    set_message(f"{RED}try_train: {ip} failed {TRY_FAIL_LIMIT}x"
                                f" without coming up — disarmed{R}")
                    continue
                spawn_launch(ip)
                LAST_TRY_LAUNCH[ip] = now
                TRY_FAILS[ip] = TRY_FAILS.get(ip, 0) + 1


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
RESIZED = False      # SIGWINCH arrived: re-read COLS and redraw in main loop
RENDER_NOW = False   # a worker finished a command: redraw in main loop
LAST_RENDER = 0.0    # epoch of the last full render (set by render())
RENDER_SIG = None    # STATS signature at the last render (conditional redraw)
RENDER_MIN = -1      # minute shown in the title at the last render


def stats_sig():
    """Signature of the current STATS — the periodic render is skipped when
    it is unchanged (keeps terminal traffic to a minimum on slow/remote
    terminals like VSCode's SSH terminal, where a 2KB full redraw visibly
    delays keystrokes)."""
    return tuple(STATS.get(i) for i in range(len(IPS)))


def restore_terminal():
    global OLD_TERM
    try:
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()
        if OLD_TERM is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, OLD_TERM)
    except Exception:
        pass


def cleanup():
    """Request a clean shutdown. The main loop notices QUIT within ~0.2s and
    runs the teardown (kill our trainers, restore the terminal, clear the
    screen) in its own thread — no network I/O ever happens in a signal
    handler."""
    global QUIT
    QUIT = True


def on_signal(signum, frame):
    cleanup()


def on_winch(signum, frame):
    # Just flag it; get_cols() (subprocess) and render() run in the main loop.
    global RESIZED
    RESIZED = True


def on_tstp(signum, frame):
    restore_terminal()
    os.kill(os.getpid(), signal.SIGSTOP)


def on_cont(signum, frame):
    tty.setcbreak(sys.stdin.fileno())
    render()


def main(argv=None):
    global THEME, OLD_TERM, RESIZED, RENDER_NOW, LAST_RENDER
    ap = argparse.ArgumentParser(description="nodes monitor workbench")
    ap.add_argument("--config", help="config file (default: .data override, else template)")
    ap.add_argument("--exclude", help="skip nodes by 1-based workbench index, e.g. 1,3")
    args = ap.parse_args(argv)

    load_config(args.config)
    finalize_config()
    load_nodes()

    if args.exclude:
        for part in args.exclude.split(","):
            part = part.strip()
            if not part.isdigit() or not (1 <= int(part) <= len(IPS)):
                sys.exit(f"invalid --exclude index: {part} (1..{len(IPS)})")
            EXCLUDE.add(int(part) - 1)

    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        sys.exit("manager.py needs an interactive terminal")

    THEME = CONF.get("ui", {}).get("theme", "table")
    if THEME not in THEMES:
        sys.exit(f"unknown theme '{THEME}' in config (see 'theme' in the workbench)")

    OLD_TERM = termios.tcgetattr(sys.stdin.fileno())
    tty.setcbreak(sys.stdin.fileno())

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    # SIGHUP (ssh session drop) and SIGQUIT (Ctrl-\) keep their default
    # dispositions only until here — default means instant death with NO
    # cleanup: trainers left on the nodes, terminal left in cbreak mode.
    signal.signal(signal.SIGHUP, on_signal)
    signal.signal(signal.SIGQUIT, on_signal)
    signal.signal(signal.SIGWINCH, on_winch)
    signal.signal(signal.SIGTSTP, on_tstp)
    signal.signal(signal.SIGCONT, on_cont)

    set_message(f"{GRAY}ready — type 'help'{R}")
    get_cols()

    # Startup probe: one health snapshot per node, in parallel — the ONLY
    # probe of the session (the probe occupies GPU memory while it runs).
    # Verdicts land in the cache before the state workers need it and are
    # then frozen for the process lifetime (see probe_cuda_cached).
    for ip in IPS:
        threading.Thread(target=probe_cuda_cached, args=(ip,), daemon=True).start()

    # One collection thread per node: a slow or failing node never blocks
    # the UI or the other nodes (see node_worker).
    for i, ip in enumerate(IPS):
        threading.Thread(target=node_worker, args=(i, ip), daemon=True).start()

    try:
        render()
        while not QUIT:
            # PENDING_IN holds bytes that were read ahead (e.g. an ESC
            # sequence that turned out to be a lone ESC): consume them
            # immediately — waiting on select() would swallow the keystroke.
            if PENDING_IN:
                ch = read_char()
            else:
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
                ch = read_char() if r else ""   # "" = no input; None = EOF
            if ch is None:      # stdin EOF (detached terminal)
                cleanup()
            elif ch:
                handle_key(ch)
            now = time.time()
            if RESIZED:
                RESIZED = False
                get_cols()
                render()
            elif RENDER_NOW:
                RENDER_NOW = False
                render()
            elif now - LAST_RENDER >= INTERVAL:
                try_sweep()
                # Conditional redraw: skip the full-screen repaint when
                # nothing changed (state stable + same minute). On VSCode's
                # SSH terminal a 2KB redraw costs real time and makes
                # keystrokes feel laggy — so we only pay for it on change.
                if stats_sig() != RENDER_SIG or time.localtime().tm_min != RENDER_MIN:
                    render()
                LAST_RENDER = now
    finally:
        # Whatever happens (exception, signal, EOF): never leave OUR trainers
        # on the nodes, and never leave the terminal broken or cluttered —
        # otherwise the user's shell swallows typed input and the last frame
        # stays on screen.
        try:
            _kill_launched_trainers()
        finally:
            restore_terminal()
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
