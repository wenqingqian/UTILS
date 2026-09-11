# nodes_monitor — AI agent guide

## view.sh — read-only query: which nodes are in a given state

stdout = ONE line of space-separated IPs (empty line if none). All diagnostics go to stderr. Always safe to run: it never changes node state.

```bash
./view.sh [--idle|--used|--train|--broken|--offline|--no_gpu|--unverified] [--num N|all] [--hostfile PATH] [--config FILE]
```

Defaults: `--idle`, `--num all`. At most one state flag; `--num N` keeps the first N matches in node-list order.

```bash
./view.sh                      # all idle nodes  -> "10.8.1.17 10.8.1.23 ..."
./view.sh --used --num 2       # first 2 used nodes
FREE=$(./view.sh --num 4)      # capture into a var for scripting
./view.sh --num 2 --hostfile hf  # also writes an MPI hostfile:
                                 #   #hostfile
                                 #   10.8.1.17 slots=8
                                 #   10.8.1.23 slots=8
```

States: `IDLE` free & probe-verified · `USED` occupied by others · `TRAIN` occupied by this tool's own trainer · `BROKEN` GPU/driver confirmed bad · `OFFLINE` unreachable · `NO_GPU` · `UNVERIFIED` probe inconclusive — never treat as free.

## Node-side path model (read before moving the repo or editing paths)

Any repo file addressed ON a node (probe / launcher / gpu_kill.sh) resolves as `/workspace/<repo's path relative to [paths] workspace_root>/...` — derived at startup from the tools' own real location, never configurable and never a hardcoded `UTILS` segment. Consequences:

- The repo may sit in any workspace subdirectory (e.g. `<ws>/sga_framework/UTILS`), but MUST live under the workspace (or under `/workspace` itself when the tools run inside the container); `finalize_config` aborts loudly otherwise.
- Symptom of a broken node-side path is NOT an error message: the probe exits rc=2 (file not found), every free node shows `UNVERIFIED`, and `--train`/auto-occupancy silently selects nothing (UNVERIFIED nodes are never launched on). If MANY nodes suddenly read UNVERIFIED, suspect the path derivation before suspecting GPUs.
- `[ssh] identity` follows the same relative-to-workspace rule, so a repo move only ever needs the `workspace_root` value to stay correct.

The CUDA health probe runs once per GPU-bearing node on each `view.sh` call, including nodes with current compute or memory usage. A confirmed `BROKEN` probe result takes precedence over `USED`; `TRAIN`/`USED` remain the state when the probe is healthy or inconclusive. Probe cache is cross-call only for confirmed `BROKEN` verdicts (default 12 h); healthy and inconclusive results are re-probed on the next call. The CUDA probe briefly occupies GPU memory. Config/node list: `../.data/nodes_monitor/config.toml` if present, else the bundled template. Node entries may be `IP` or `IP:port`; an entry-level port overrides `[ssh].port` for that node. Management actions (occupy/kill/release) live in `manager.sh`, not here.
