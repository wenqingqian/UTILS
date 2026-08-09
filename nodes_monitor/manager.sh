#!/usr/bin/env bash
# Workbench-style persistent control panel for the multi-node GPU monitor.
#
#   ./manager.sh                    start the workbench (nodes from config.toml)
#   ./manager.sh --config FILE      use a specific config file
#   ./manager.sh --exclude 1,3      skip nodes by workbench index (1-based)
#
# Commands at the `cmd>` prompt:
#   kill N | kill all | kill 1,2,3   kill every GPU process on the node(s)
#                                    (everything nvidia-smi lists)
#   train [N|1,2]                    occupy node(s) NOW — only works when idle,
#                                    non-idle nodes report FAIL(state);
#                                    no target = all nodes (idle ones only)
#   try_train [N|1,2]                arm background occupation: node(s) are
#                                    occupied as soon as they go idle; shown in
#                                    the status bar / node markers;
#                                    no target = all nodes
#   release N | release all          stop try_train on node(s) and kill only our
#                                    tagged trainer processes — every other
#                                    process is left untouched (that is the
#                                    difference to `kill`)
#   theme NAME                       switch UI theme (table|icons|dashboard|status)
#   help                             show this help (any key to return)
#   quit | exit | q                  leave the workbench (Ctrl-C / Ctrl-D too, clean)
#
# Node indices are 1-based positions in the node list.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Shared plumbing: config loading, run_node, get_node_state, launch_trainer,
# kill_trainer, is_excluded, join_csv, colors, INTERVAL/COOLDOWN, THEME, ...
source "${SCRIPT_DIR}/monitor.sh"

# ---------------------------------------------------------------------------
# Workbench options (manager-specific)
# ---------------------------------------------------------------------------

MGR_EXCLUDE_INDICES=()

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --config)
                if [[ -z "${2:-}" ]]; then
                    echo "usage: $0 [--config FILE] [--exclude 1,3]" >&2
                    exit 1
                fi
                CONFIG_FILE="$2"
                shift 2
                ;;
            --exclude)
                if [[ -z "${2:-}" ]]; then
                    echo "usage: $0 [--config FILE] [--exclude 1,3]" >&2
                    exit 1
                fi
                IFS=', ' read -r -a MGR_EXCLUDE_INDICES <<< "$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1" >&2
                echo "usage: $0 [--config FILE] [--exclude 1,3]" >&2
                exit 1
                ;;
        esac
    done
}

# 1-based workbench indices -> 0-based monitor.sh EXCLUDE_INDICES.
mgr_apply_exclude() {
    EXCLUDE_INDICES=()
    local idx zero
    for idx in "${MGR_EXCLUDE_INDICES[@]}"; do
        if [[ ! "${idx}" =~ ^[0-9]+$ ]]; then
            echo "invalid --exclude index: ${idx}" >&2
            exit 1
        fi
        zero=$((idx - 1))
        if (( zero < 0 || zero >= ${#IPS[@]} )); then
            echo "index out of range: ${idx} (1..${#IPS[@]})" >&2
            exit 1
        fi
        EXCLUDE_INDICES+=("${zero}")
    done
}

# ---------------------------------------------------------------------------
# Workbench state
# ---------------------------------------------------------------------------

declare -A STATS          # node index -> "total|compute|state"
declare -A PENDING_TRY    # ip -> 1 : try_train armed, wait for idle
declare -A LAST_TRY_LAUNCH  # ip -> epoch of last launch attempt

INPUT=""                  # current command being typed
MESSAGE=""                # status bar message (last action)
STATE_TMP=""              # per-tick state files

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

set_message() {
    MESSAGE="$1"
    if (( ${#MESSAGE} > 70 )); then
        MESSAGE="${MESSAGE:0:67}..."
    fi
}

# 1-based workbench index of an ip (0 if not found).
ip_index() {
    local ip="$1" i
    for i in "${!IPS[@]}"; do
        if [[ "${IPS[$i]}" == "${ip}" ]]; then
            echo "$((i + 1))"
            return 0
        fi
    done
    echo 0
}

# Fetch one node's state and print it in a single parseable line.  Runs inside
# a background subshell so all nodes are probed in parallel.
mgr_get_state() {
    local ip="$1"
    local total=0 compute=0 mem_used=0 trainer=0 state=""
    get_node_state "${ip}" total compute mem_used trainer state
    printf '%s|%s|%s\n' "${total}" "${compute}" "${state}"
}

# One parallel state-fetch round; results go to per-node files (atomic mv so
# the foreground loop never reads a half-written file).
collect_stats() {
    local -a pids=()
    local i p
    for i in "${!IPS[@]}"; do
        ( mgr_get_state "${IPS[$i]}" > "${STATE_TMP}/${i}.tmp" && mv "${STATE_TMP}/${i}.tmp" "${STATE_TMP}/${i}" ) &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "${p}" 2>/dev/null || true; done
}

# Background loop: keep the state files fresh without blocking input/render.
refresh_loop() {
    while true; do
        collect_stats
        sleep "${INTERVAL}"
    done
}

# Foreground: read the latest state files into STATS (fast, no network).
load_stats() {
    local i total compute state
    for i in "${!IPS[@]}"; do
        total=0; compute=0; state=""
        IFS='|' read -r total compute state < "${STATE_TMP}/${i}" || true
        STATS["${i}"]="${total}|${compute}|${state}"
    done
}

# Foreground: try_train sweep — occupy armed nodes as soon as they go idle.
# Launches are fully async: launch_trainer can block on a slow ssh handshake,
# so waiting on it here would freeze the UI for seconds.
try_sweep() {
    local i ip now state
    now=$(date +%s)
    for ip in "${!PENDING_TRY[@]}"; do
        i="$(ip_index "${ip}")"
        state="${STATS[$((i - 1))]-}"
        state="${state##*|}"
        case "${state}" in
            TRAIN)
                unset "PENDING_TRY[${ip}]"
                ;;
            IDLE)
                if (( now - ${LAST_TRY_LAUNCH["${ip}"]:-0} > COOLDOWN )); then
                    LAUNCHED_IPS+=("${ip}")
                    ( launch_trainer "${ip}" ) &
                    LAST_TRY_LAUNCH["${ip}"]=${now}
                fi
                ;;
        esac
    done
}

# ---------------------------------------------------------------------------
# Command targets: "all" / "1,2,3" -> list of IPs (globals TARGETS)
# ---------------------------------------------------------------------------

TARGETS=()

parse_targets() {
    local spec="$1" part idx i parts
    TARGETS=()
    [[ -z "${spec}" ]] && return 0

    if [[ "${spec}" == "all" ]]; then
        for i in "${!IPS[@]}"; do
            is_excluded "${IPS[$i]}" || TARGETS+=("${IPS[$i]}")
        done
        return 0
    fi

    IFS=', ' read -r -a parts <<< "${spec}"
    for part in "${parts[@]}"; do
        if [[ ! "${part}" =~ ^[0-9]+$ ]]; then
            set_message "${C_RED}invalid index '${part}'${C_RESET} (expect: N | all | 1,2,3)"
            return 1
        fi
        idx=$((part - 1))
        if (( idx < 0 || idx >= ${#IPS[@]} )); then
            set_message "${C_RED}index out of range: ${part}${C_RESET} (1..${#IPS[@]})"
            return 1
        fi
        TARGETS+=("${IPS[$idx]}")
    done
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

cmd_kill() {
    local spec="$1" ip out last acc=""
    [[ -z "${spec}" ]] && { set_message "${C_YELLOW}usage: kill N | kill all | kill 1,2,3${C_RESET}"; return; }
    parse_targets "${spec}" || return

    set_message "${C_YELLOW}kill: working…${C_RESET}"
    render
    for ip in "${TARGETS[@]}"; do
        local rc=0
        out="$(run_node "${ip}" "bash ${KILL_GPU_SCRIPT}" 2>&1)" || rc=$?
        if (( rc != 0 )); then
            last="unreachable"
        else
            last="$(printf '%s' "${out}" | tail -n 1)"
        fi
        acc+="$(ip_index "${ip}"):${last} "
    done
    set_message "kill → ${acc}"
}

cmd_train() {
    local spec="$1" ip state idx i acc=""
    local -a targets=()
    if [[ -z "${spec}" ]]; then
        for i in "${!IPS[@]}"; do
            is_excluded "${IPS[$i]}" || targets+=("${IPS[$i]}")
        done
    else
        parse_targets "${spec}" || return
        targets=("${TARGETS[@]}")
    fi

    for ip in "${targets[@]}"; do
        idx="$(ip_index "${ip}")"
        state="${STATS[$((idx - 1))]-}"
        state="${state##*|}"
        if [[ "${state}" == "IDLE" ]]; then
            # Fully async (slow ssh handshakes must not freeze the UI).
            LAUNCHED_IPS+=("${ip}")
            ( launch_trainer "${ip}" ) &
            unset "PENDING_TRY[${ip}]"
            acc+="${idx}:started "
        else
            acc+="${idx}:FAIL(${state}) "
        fi
    done
    set_message "train → ${acc}"
}

cmd_try_train() {
    local spec="$1" ip state idx i acc=""
    local -a targets=()
    if [[ -z "${spec}" ]]; then
        for i in "${!IPS[@]}"; do
            is_excluded "${IPS[$i]}" || targets+=("${IPS[$i]}")
        done
    else
        parse_targets "${spec}" || return
        targets=("${TARGETS[@]}")
    fi

    for ip in "${targets[@]}"; do
        idx="$(ip_index "${ip}")"
        state="${STATS[$((idx - 1))]-}"
        state="${state##*|}"
        if [[ "${PENDING_TRY[${ip}]+x}" == "x" ]]; then
            acc+="${idx}:already-armed "
        elif [[ "${state}" == "TRAIN" ]]; then
            acc+="${idx}:already-owned "
        else
            PENDING_TRY["${ip}"]=1
            if [[ "${state}" == "IDLE" ]]; then
                acc+="${idx}:armed(idle) "
            else
                acc+="${idx}:armed(waiting ${state}) "
            fi
        fi
    done
    set_message "try_train → ${acc}"
}

cmd_release() {
    local spec="$1" ip idx state acc="" released=""
    [[ -z "${spec}" ]] && { set_message "${C_YELLOW}usage: release N | release all${C_RESET}"; return; }
    parse_targets "${spec}" || return

    for ip in "${TARGETS[@]}"; do
        idx="$(ip_index "${ip}")"
        state="${STATS[$((idx - 1))]-}"
        state="${state##*|}"
        released=""
        if [[ "${PENDING_TRY[${ip}]+x}" == "x" ]]; then
            unset "PENDING_TRY[${ip}]"
            released="disarmed "
        fi
        if [[ "${state}" == "OFFLINE" ]]; then
            released+="offline"
        elif kill_trainer "${ip}"; then
            released+="killed tagged trainer"
        else
            local rc=$?
            if (( rc > 1 )); then
                released+="unreachable"
            else
                released+="no tagged trainer"
            fi
        fi
        acc+="${idx}:${released} "
    done
    set_message "release → ${acc}"
}

cmd_theme() {
    local spec="$1"
    spec="$(trim "${spec}")"
    case "${spec}" in
        table|icons|dashboard|status)
            THEME="${spec}"
            set_message "theme → ${THEME}"
            ;;
        "")
            set_message "current theme: ${THEME} (table|icons|dashboard|status)"
            ;;
        *)
            set_message "${C_RED}unknown theme '${spec}'${C_RESET} (table|icons|dashboard|status)"
            ;;
    esac
}

handle_command() {
    local input="$1" cmd rest
    cmd="${input%% *}"
    rest="${input#"${cmd}"}"
    rest="${rest#"${rest%%[![:space:]]*}"}"
    case "${cmd}" in
        "")
            ;;
        h|help)
            show_help
            ;;
        quit|exit|q)
            cleanup
            ;;
        kill)
            cmd_kill "${rest}"
            ;;
        train)
            cmd_train "${rest}"
            ;;
        try_train)
            cmd_try_train "${rest}"
            ;;
        release)
            cmd_release "${rest}"
            ;;
        theme)
            cmd_theme "${rest}"
            ;;
        *)
            set_message "${C_RED}unknown command '${cmd}'${C_RESET} — type 'help'"
            ;;
    esac
}

show_help() {
    printf '\033[2J\033[H'
    FRAME_W=$(( COLS - 2 ))
    frame_title "${C_BOLD}${C_CYAN}● nodes monitor workbench — commands${C_RESET}"
    frame_line ""
    frame_line "  kill N | kill all | kill 1,2,3   kill every GPU process on the node(s)"
    frame_line "                                   (everything nvidia-smi lists)"
    frame_line "  train [N|1,2]                    occupy node(s) now — only when idle;"
    frame_line "                                   non-idle nodes report FAIL(state)"
    frame_line "  try_train [N|1,2]                arm background occupation: node(s)"
    frame_line "                                   occupied as soon as they go idle"
    frame_line "  release N | release all          stop try_train and kill only our"
    frame_line "                                   tagged trainer processes"
    frame_line "  theme NAME                       switch UI theme (table|icons|dashboard|status)"
    frame_line "  help                             show this help"
    frame_line "  quit | exit | q                  leave the workbench (Ctrl-C / Ctrl-D too)"
    frame_line ""
    frame_line "${C_GRAY}node indices are 1-based positions in the node list${C_RESET}"
    frame_line ""
    frame_line "${C_GRAY}press any key to return${C_RESET}"
    frame_bottom
    read -t 60 -N 1 -r || true
}

# ---------------------------------------------------------------------------
# Frame & alignment helpers (full-width rounded frame)
#
# Perf note: render runs on every refresh cycle and handle_key triggers the
# prompt redraw on EVERY keystroke, so these helpers avoid $(...) subshells.
# strip_ansi stores into the global STRIP_RESULT instead of printing; the
# remaining printf calls are bash builtins (no fork).
# ---------------------------------------------------------------------------

FRAME_W=0       # frame inner width (cols - 2), recomputed on every render
PROMPT_COL=0    # absolute column of the input caret, set by frame_prompt
COLS=80         # terminal width, refreshed via get_cols (SIGWINCH)
REFRESH_PID=""  # background stats-collection loop
STRIP_RESULT=""

# Terminal width cache (tput is a fork; SIGWINCH keeps it in sync).
get_cols() {
    COLS="$(tput cols 2>/dev/null || echo 80)"
}

# Remove the known ANSI color codes from a string; result in STRIP_RESULT.
strip_ansi() {
    local s="$1" c
    for c in C_RESET C_BOLD C_RED C_GREEN C_YELLOW C_BLUE C_CYAN C_GRAY; do
        s="${s//${!c}/}"
    done
    STRIP_RESULT="${s}"
}

# Right-pad a (possibly colored) string to an exact VISIBLE width.
pad_to() {
    local width="$1" pad
    shift
    local s="$*"
    strip_ansi "${s}"
    pad=$(( width - ${#STRIP_RESULT} ))
    (( pad < 0 )) && pad=0
    printf '%s%*s' "${s}" "${pad}" ""
}

# Print <count> box-drawing dashes. Uses bash parameter expansion instead of
# `tr ' ' '─'`, which mangles multibyte chars under a C locale.
dash_fill() {
    local n="$1" spaces
    if (( n <= 0 )); then
        return 0
    fi
    spaces="$(printf '%*s' "${n}" '')"
    printf '%s' "${spaces// /─}"
}

# One full-width frame row: │ <content><pad> │  (total visible width = FRAME_W + 2,
# matching frame_title/frame_bottom)
frame_line() {
    local s="$*" pad
    strip_ansi "${s}"
    pad=$(( FRAME_W - ${#STRIP_RESULT} - 2 ))
    (( pad < 0 )) && pad=0
    printf '│ %s%*s │\n' "${s}" "${pad}" ""
}

# Rounded top edge with a title: ╭─ <title> ──────╮
frame_title() {
    local s="$*" fill
    strip_ansi "${s}"
    fill=$(( FRAME_W - ${#STRIP_RESULT} - 3 ))
    (( fill < 1 )) && fill=1
    printf '╭─ %s %s╮\n' "${s}" "$(dash_fill "${fill}")"
}

# Rounded bottom edge: ╰──────╯
frame_bottom() {
    printf '╰%s╯\n' "$(dash_fill "${FRAME_W}")"
}

# Rounded input box (2-space inset); the caret is placed right after the
# input text by render (PROMPT_COL). Overlong input shows its tail.
# All three rows are exactly (inner+2) wide: top "╭─ cmd ─" + dashes + "╮",
# middle "│ " + text + " │", bottom "╰" + dashes + "╯".
frame_prompt() {
    local inner=$(( FRAME_W - 8 ))
    local maxlen=$(( inner - 2 ))
    local disp="${INPUT}" pad top_dash bot_dash
    if (( ${#disp} > maxlen )); then
        disp="${disp: -maxlen}"
    fi
    pad=$(( inner - 2 - ${#disp} ))
    top_dash="$(dash_fill $((inner - 7)))"
    bot_dash="$(dash_fill "${inner}")"
    frame_line "  ╭─ ${C_BOLD}cmd${C_RESET} ─${top_dash}╮"
    frame_line "  │ ${disp}$(printf '%*s' "${pad}" '') │"
    frame_line "  ╰${bot_dash}╯"
    # The caret sits right after the input: frame border "│ " (2) + inset
    # "  " (2) + box border "│ " (2) precede the text, so the column is 6+len.
    PROMPT_COL=$(( 6 + ${#disp} ))
}

# Redraw ONLY the prompt's middle row (the line the caret is on) after a
# keystroke changed INPUT — avoids a full-screen redraw per character. The
# redrawn row is byte-identical to what frame_prompt prints via frame_line.
refresh_prompt() {
    local inner=$(( FRAME_W - 8 ))
    local maxlen=$(( inner - 2 ))
    local disp="${INPUT}" pad content fpad
    if (( ${#disp} > maxlen )); then
        disp="${disp: -maxlen}"
    fi
    pad=$(( inner - 2 - ${#disp} ))
    content="  │ ${disp}$(printf '%*s' "${pad}" '') │"
    strip_ansi "${content}"
    fpad=$(( FRAME_W - ${#STRIP_RESULT} - 2 ))
    (( fpad < 0 )) && fpad=0
    printf '\033[G\033[2K'
    printf '│ %s%*s │' "${content}" "${fpad}" ""
    printf '\033[%dG' "$((6 + ${#disp}))"
}

# ---------------------------------------------------------------------------
# Theme system: every theme implements three functions
#   theme_<name>_header   -> prints the title line(s)
#   theme_<name>_node IDX IP -> prints one node row
#   theme_<name>_status   -> prints the status bar
# THEME decides which set is dispatched.
# ---------------------------------------------------------------------------

state_color_for() {
    case "$1" in
        IDLE)   printf '%s' "${C_GREEN}" ;;
        TRAIN)  printf '%s' "${C_BLUE}"  ;;
        USED)   printf '%s' "${C_YELLOW}";;
        BROKEN|OFFLINE|NO_GPU) printf '%s' "${C_RED}" ;;
        *)      printf '%s' "${C_GRAY}"  ;;
    esac
}

state_icon_for() {
    case "$1" in
        IDLE)    printf '%s' "●" ;;
        TRAIN)   printf '%s' "▲" ;;
        USED)    printf '%s' "◆" ;;
        OFFLINE) printf '%s' "✖" ;;
        BROKEN)  printf '%s' "✖" ;;
        NO_GPU)  printf '%s' "○" ;;
        *)       printf '%s' "?" ;;
    esac
}

node_markers() {
    local idx="$1" ip="$2" state="$3" m=""
    [[ "${state}" == "TRAIN" ]] && m+="${C_BOLD}${C_BLUE}◆ owned${C_RESET} "
    [[ "${PENDING_TRY[${ip}]+x}" == "x" ]] && m+="${C_BOLD}${C_YELLOW}↻ try${C_RESET} "
    is_excluded "${ip}" && m+="${C_GRAY}⊘ excl${C_RESET} "
    printf '%s' "${m% }"
}

# Read the STATS entry for workbench index $1 into total/compute/state.
# Always exits 0 (callers must not be aborted by set -e on a missing entry).
stats_for() {
    local idx="$1"
    local -n _total="$2" _compute="$3" _state="$4"
    _total=0; _compute=0; _state="checking"
    local key=$((idx - 1))
    if [[ -n "${STATS[${key}]+x}" ]]; then
        IFS='|' read -r _total _compute _state <<< "${STATS[${key}]}"
    fi
    return 0
}

# ---- theme: table (default, rounded aligned table) ----
# Column widths: idx=3, ip=12, state=8, compute=8, marks=24. Header and data
# rows are produced by the same pad_to calls, so the borders never drift.
theme_table_header() {
    frame_line "  ${C_GRAY}╭─────┬──────────────┬──────────┬──────────┬────────────────────────╮${C_RESET}"
    frame_line "  ${C_GRAY}│${C_RESET} $(pad_to 3 "${C_BOLD}idx${C_RESET}") ${C_GRAY}│${C_RESET} $(pad_to 12 "${C_BOLD}ip${C_RESET}") ${C_GRAY}│${C_RESET} $(pad_to 8 "${C_BOLD}state${C_RESET}") ${C_GRAY}│${C_RESET} $(pad_to 8 "${C_BOLD}compute${C_RESET}") ${C_GRAY}│${C_RESET} $(pad_to 24 "${C_BOLD}marks${C_RESET}") ${C_GRAY}│${C_RESET}"
    frame_line "  ${C_GRAY}├─────┼──────────────┼──────────┼──────────┼────────────────────────┤${C_RESET}"
}
theme_table_node() {
    local idx="$1" ip="$2"
    local total compute state
    stats_for "${idx}" total compute state
    local sc; sc="$(state_color_for "${state}")"
    local marks; marks="$(node_markers "${idx}" "${ip}" "${state}")"
    local compute_str="-" cc="${C_RESET}"
    if [[ "${state}" != "checking" && "${state}" != "OFFLINE" && "${state}" != "NO_GPU" && "${state}" != "BROKEN" ]]; then
        compute_str="${compute}/${total}"
        # USED with nothing computing = memory held but idle cores: highlight.
        if [[ "${state}" == "USED" && "${compute}" == "0" ]]; then
            cc="${C_RED}"
        fi
    fi
    frame_line "  ${C_GRAY}│${C_RESET} $(pad_to 3 "${idx}") ${C_GRAY}│${C_RESET} $(pad_to 12 "${ip}") ${C_GRAY}│${C_RESET} $(pad_to 8 "${sc}${state}${C_RESET}") ${C_GRAY}│${C_RESET} $(pad_to 8 "${cc}${compute_str}${C_RESET}") ${C_GRAY}│${C_RESET} $(pad_to 24 "${marks}") ${C_GRAY}│${C_RESET}"
}
theme_table_status() {
    frame_line "  ${C_GRAY}╰─────┴──────────────┴──────────┴──────────┴────────────────────────╯${C_RESET}"
    frame_line ""
    status_common
}

# ---- theme: icons ----
theme_icons_header() {
    frame_line "  ${C_GRAY}──────────────────────────────────────────────────────────────${C_RESET}"
}
theme_icons_node() {
    local idx="$1" ip="$2"
    local total compute state
    stats_for "${idx}" total compute state
    local sc icon marks
    sc="$(state_color_for "${state}")"
    icon="$(state_icon_for "${state}")"
    marks="$(node_markers "${idx}" "${ip}" "${state}")"
    case "${state}" in
        checking|OFFLINE|NO_GPU|BROKEN)
            frame_line "  ${sc}${icon}${C_RESET} ${C_GRAY}[${idx}]${C_RESET} ${C_BOLD}${ip}${C_RESET}  ${sc}${state}${C_RESET}  ${marks}"
            ;;
        *)
            local compute_str="${compute}/${total}" cc="${C_RESET}"
            [[ "${state}" == "USED" && "${compute}" == "0" ]] && cc="${C_RED}"
            frame_line "  ${sc}${icon}${C_RESET} ${C_GRAY}[${idx}]${C_RESET} ${C_BOLD}${ip}${C_RESET}  ${sc}${state}${C_RESET}  ${cc}${compute_str}${C_RESET} gpu  ${marks}"
            ;;
    esac
}
theme_icons_status() {
    frame_line "  ${C_GRAY}──────────────────────────────────────────────────────────────${C_RESET}"
    frame_line ""
    status_common
}

# ---- theme: dashboard (bold emphasis rows; the outer frame replaces the
# former double-line box) ----
theme_dashboard_header() {
    frame_line "  ${C_GRAY}──────────────────────────────────────────────────────────────${C_RESET}"
}
theme_dashboard_node() {
    local idx="$1" ip="$2"
    local total compute state
    stats_for "${idx}" total compute state
    local sc icon marks
    sc="$(state_color_for "${state}")"
    icon="$(state_icon_for "${state}")"
    marks="$(node_markers "${idx}" "${ip}" "${state}")"
    local compute_str="-" cc="${C_RESET}"
    if [[ "${state}" != "checking" && "${state}" != "OFFLINE" && "${state}" != "NO_GPU" && "${state}" != "BROKEN" ]]; then
        compute_str="${compute}/${total}"
        [[ "${state}" == "USED" && "${compute}" == "0" ]] && cc="${C_RED}"
    fi
    frame_line "  ${sc}${icon}${C_RESET} ${C_BOLD}[${idx}]${C_RESET} ${C_BOLD}${ip}${C_RESET}  ${sc}${state}${C_RESET}  ${cc}${compute_str}${C_RESET}  ${marks}"
}
theme_dashboard_status() {
    frame_line "  ${C_GRAY}──────────────────────────────────────────────────────────────${C_RESET}"
    frame_line ""
    status_common
}

# ---- theme: status (condensed rows + detailed status area) ----
theme_status_header() {
    :
}
theme_status_node() {
    local idx="$1" ip="$2"
    local total compute state
    stats_for "${idx}" total compute state
    local sc; sc="$(state_color_for "${state}")"
    local marks; marks="$(node_markers "${idx}" "${ip}" "${state}")"
    local compute_str="-" cc="${C_RESET}"
    if [[ "${state}" != "checking" && "${state}" != "OFFLINE" && "${state}" != "NO_GPU" && "${state}" != "BROKEN" ]]; then
        compute_str="${compute}/${total}"
        [[ "${state}" == "USED" && "${compute}" == "0" ]] && cc="${C_RED}"
    fi
    frame_line "  ${C_GRAY}[${idx}]${C_RESET} ${C_BOLD}${ip}${C_RESET}  ${sc}${state}${C_RESET}  ${cc}${compute_str}${C_RESET}  ${marks}"
}
theme_status_status() {
    local -a pending=() owned=() idle=() used=() down=()
    local i ip state
    for i in "${!IPS[@]}"; do
        ip="${IPS[$i]}"
        [[ "${PENDING_TRY[${ip}]+x}" == "x" ]] && pending+=("$((i + 1))")
        state="${STATS[${i}]-}"; state="${state##*|}"
        case "${state}" in
            TRAIN)   owned+=("$((i+1))") ;;
            IDLE)    idle+=("$((i+1))") ;;
            USED)    used+=("$((i+1))") ;;
            OFFLINE|BROKEN|NO_GPU) down+=("$((i+1))") ;;
        esac
    done
    frame_line "  ${C_GRAY}──────────────────────────────────────────────────────────────${C_RESET}"
    frame_line "  ${C_GREEN}idle:[${idle[*]}]${C_RESET} ${C_YELLOW}used:[${used[*]}]${C_RESET} ${C_BLUE}owned:[${owned[*]}]${C_RESET} ${C_RED}down:[${down[*]}]${C_RESET} ${C_YELLOW}try:[${pending[*]}]${C_RESET}"
    [[ -n "${MESSAGE}" ]] && frame_line "  ${MESSAGE}"
}

# Common status bar (shared by table/icons/dashboard).
status_common() {
    local -a pending=() owned=()
    local i ip state
    for i in "${!IPS[@]}"; do
        ip="${IPS[$i]}"
        [[ "${PENDING_TRY[${ip}]+x}" == "x" ]] && pending+=("$((i + 1))")
        state="${STATS[${i}]-}"; state="${state##*|}"
        [[ "${state}" == "TRAIN" ]] && owned+=("$((i + 1))")
    done
    local s="${C_YELLOW}try: [${pending[*]}]   owned: [${owned[*]}]${C_RESET}"
    [[ -n "${MESSAGE}" ]] && s+="  ${C_GRAY}|${C_RESET}  ${MESSAGE}"
    frame_line "  ${s}"
}

# ---------------------------------------------------------------------------
# Render (dispatch by theme, inside the full-width rounded frame)
# ---------------------------------------------------------------------------

render() {
    FRAME_W=$(( COLS - 2 ))
    printf '\033[2J\033[H\033[?25l'
    frame_title "${C_BOLD}${C_CYAN}● nodes monitor workbench${C_RESET}  ${C_GRAY}${#IPS[@]} nodes  $(date '+%H:%M:%S')  theme:${THEME}${C_RESET}"
    frame_line ""
    "theme_${THEME}_header"
    local i
    for i in "${!IPS[@]}"; do
        "theme_${THEME}_node" "$((i + 1))" "${IPS[$i]}"
    done
    "theme_${THEME}_status"
    frame_line ""
    frame_prompt
    frame_bottom
    # The prompt's middle row is three lines above the final cursor position
    # (bottom edge ends with a newline): up 3, then absolute column.
    printf '\033[3A\033[%dG' "${PROMPT_COL}"
    printf '\033[?25h'
}

# ---------------------------------------------------------------------------
# Keyboard handling (terminal runs in cbreak mode: -icanon -echo)
#
# Known limitation: input is ASCII-only. Multibyte characters are dropped
# (LC_ALL=C byte decoding) and escape sequences are handled by swallowing
# exactly two following bytes — long CSI sequences (Page Up/Down, OSC) can
# leak extra bytes as literal input.
# ---------------------------------------------------------------------------

handle_key() {
    local ch="$1" bv
    case "${ch}" in
        $'\r'|$'\n')            # Enter
            handle_command "${INPUT}"
            INPUT=""
            render
            ;;
        $'\x7f'|$'\b')          # Backspace
            INPUT="${INPUT%?}"
            refresh_prompt
            ;;
        $'\x03'|$'\x04')        # Ctrl-C / Ctrl-D -> clean quit
            cleanup
            ;;
        $'\x15')                # Ctrl-U -> clear line
            INPUT=""
            refresh_prompt
            ;;
        $'\x09')                # Tab -> ignore
            ;;
        $'\x1b')                # Escape sequences (arrows...): swallow
            read -t 0.1 -N 1 -r 2>/dev/null || true
            read -t 0.1 -N 1 -r 2>/dev/null || true
            ;;
        *)
            if [[ "${ch}" == "'" ]]; then
                bv=39
            else
                bv="$(LC_ALL=C printf '%d' "'${ch}")"
            fi
            if (( bv >= 32 && bv <= 126 )); then
                INPUT+="${ch}"
                refresh_prompt
            fi
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

cleanup() {
    trap - EXIT INT TERM TSTP CONT WINCH
    [[ -n "${REFRESH_PID}" ]] && kill "${REFRESH_PID}" 2>/dev/null || true
    # Never leave a trainer behind: kill every trainer THIS session launched
    # (best-effort — an unreachable node just times out).
    # 1. Stop in-flight launchers FIRST so no new spawn can be initiated
    #    after we start killing (a detached ssh -f / docker exec -d child
    #    keeps running even when its launcher subshell is killed).
    jobs -p 2>/dev/null | xargs -r kill 2>/dev/null || true
    wait 2>/dev/null || true
    # 2. ...then kill our trainers twice: a detached ssh -f / docker exec -d
    #    can still land the trainer after the first round (see monitor.sh).
    #    The +-guard form expands to zero words for an empty LAUNCHED_IPS
    #    (the ${arr[@]:-} idiom would iterate once with ip="" — and is_local
    #    "" is TRUE on this host, pkill'ing the LOCAL container).
    local round ip
    for round in 1 2; do
        for ip in ${LAUNCHED_IPS[@]+"${LAUNCHED_IPS[@]}"}; do
            kill_trainer "${ip}" >/dev/null 2>&1 || true
        done
        (( round == 1 )) && sleep 3
    done
    stty sane 2>/dev/null || true
    tput rmcup 2>/dev/null || true
    printf '\033[?25h' 2>/dev/null || true
    [[ -n "${STATE_TMP:-}" ]] && rm -rf "${STATE_TMP}" 2>/dev/null || true
    # bash's read(1) restores the terminal to its pre-read state during exit,
    # clobbering the `stty sane` above (leaving the shell with -icanon -echo
    # -> "swallowed" input). Exec away from bash so that exit-time restore
    # never runs; the final stty leaves the terminal usable.
    exec /bin/stty sane 2>/dev/null || exit 0
}

main() {
    parse_args "$@"
    load_config
    finalize_config
    load_nodes
    mgr_apply_exclude

    # Validate the theme name (whitelist; also guards the theme_ dispatch).
    case "${THEME}" in
        table|icons|dashboard|status) ;;
        *) echo "unknown theme '${THEME}' in config (table|icons|dashboard|status)" >&2; exit 1 ;;
    esac

    if [[ ! -t 0 || ! -t 1 ]]; then
        echo "manager.sh needs an interactive terminal" >&2
        exit 1
    fi

    STATE_TMP="$(mktemp -d /tmp/utils_mgr.XXXXXX)"

    tput smcup 2>/dev/null || true
    stty -icanon -echo 2>/dev/null || true

    trap cleanup EXIT INT TERM
    trap 'stty sane 2>/dev/null || true; kill -STOP $$' TSTP
    trap 'stty -icanon -echo 2>/dev/null || true' CONT
    trap 'get_cols' WINCH
    get_cols

    set_message "${C_GRAY}ready — type 'help'${C_RESET}"

    # The collection loop starts immediately but runs in the background, so
    # the UI appears at once (nodes show "checking") and slow nodes never
    # block startup, keystrokes or rendering.
    refresh_loop &
    REFRESH_PID=$!

    render
    while true; do
        if read -t "${INTERVAL}" -N 1 -r ch; then
            handle_key "${ch}"
        else
            # read returns 1 on stdin EOF (closed/detached terminal) but >128
            # on timeout — quit on EOF instead of probing all nodes in a loop.
            if (( $? == 1 )); then
                cleanup
            fi
            load_stats
            try_sweep
            render
        fi
    done
}

main "$@"
