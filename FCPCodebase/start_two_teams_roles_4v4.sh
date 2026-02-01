#!/usr/bin/env bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

host=${1:-localhost}
port=${2:-3100}
team1=${3:-FCPortugal}
team2=${4:-Opponent}

# ✅ Windows RoboViz 的绘图地址（WSL 网关）
draw_host=${5:-172.23.64.1}

# ✅ 只给一个球员开 Debug（避免8个人全画导致爆炸）
debug_team=${6:-$team1}
debug_unum=${7:-9}

# 4 roles per team: 1 GK, 2 DEF, 9 ST_L, 10 ST_R
unums=(1 2 9 10)

mkdir -p ./runlogs

echo "Starting two teams (4v4 roles):"
echo "  Team1: $team1  unums=${unums[*]}"
echo "  Team2: $team2  unums=${unums[*]}"
echo "  Server: $host:$port"
echo "  DrawHost: $draw_host"
echo "  Debug: team=$debug_team unum=$debug_unum"
echo ""

start_team () {
  local team="$1"
  for u in "${unums[@]}"; do
    dbg=0
    if [[ "$team" == "$debug_team" && "$u" == "$debug_unum" ]]; then
      dbg=1
    fi

    python3 ./Run_Player.py -i "$host" -d "$draw_host" -p "$port" -u "$u" -t "$team" -P 0 -D "$dbg" \
      > "./runlogs/${team}_${u}.out" 2>&1 &
    sleep 0.15
  done
}

# Team 1 slow start
start_team "$team1"

sleep 1

# Team 2 slow start
start_team "$team2"

echo ""
echo "Done. (8 agents started)"
echo "Use ./kill.sh to stop all agents."

