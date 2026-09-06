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

Notes: the CUDA probe — which briefly occupies GPU memory (a CUDA context per GPU) — runs only at monitor startup: once per session in manager.sh (verdicts frozen for the process lifetime), at most once per node per view.sh call with a 12 h cross-call cache (`cooldown`). Config/node list: `../.data/nodes_monitor/config.toml` if present, else the bundled template. Management actions (occupy/kill/release) live in `manager.sh`, not here.
