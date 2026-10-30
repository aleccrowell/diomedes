#!/bin/bash
# Start a long job (e.g. a lap_fit.jl chain) detached from the shell that launches it, in
# its own session, and record its real PID.
#
#   scripts/run_detached.sh <log> <pidfile> <command> [args...]
#
# Output goes to <log> (overwritten: use a fresh log per launch, a dying process can still
# hold the old one open). The job is outside Claude Code's tracked background commands, so
# the harness's memory-pressure reaper doesn't kill it (#113: it killed a 2011–2026 lap fit
# twice) and the harness sends no notification when it ends: watch <pidfile>'s PID with a
# tracked watcher (`ps -p $(cat <pidfile>)`), never `pgrep -f` on a pattern your own shell
# also contains. `setsid` alone forks when the caller leads a process group, so its own PID
# isn't the job's; the inner `$!` is.
set -euo pipefail
[ $# -ge 3 ] || { echo "usage: $0 <log> <pidfile> <command> [args...]" >&2; exit 2; }
log=$1 pidfile=$2
shift 2
cmd=$(printf '%q ' "$@")
setsid bash -c "$cmd > $(printf '%q' "$log") 2>&1 < /dev/null & echo \$! > $(printf '%q' "$pidfile")"
for _ in 1 2 3 4 5 6 7 8 9 10; do [ -s "$pidfile" ] && break; sleep 0.2; done
pid=$(cat "$pidfile")
ps -p "$pid" > /dev/null || { echo "started but already gone: pid $pid, see $log" >&2; exit 1; }
echo "$pid"
