#!/bin/bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

host=${1:-localhost}
port=${2:-3100}
team1=${3:-Barcelona}
team2=${4:-RealMadrid}

mkdir -p ./runlogs

echo "Starting two teams slowly:"
echo "  Team1: $team1 (1..11)"
echo "  Team2: $team2 (1..11)"
echo "  Server: $host:$port"
echo ""

# Team 1 (slow start)
for i in {1..11}; do
  python3 ./Run_Player.py -i "$host" -p "$port" -u "$i" -t "$team1" -P 0 -D 0 \
    > "./runlogs/${team1}_${i}.out" 2>&1 &
  sleep 0.15
done

sleep 2

# Team 2 (slow start)
for i in {1..11}; do
  python3 ./Run_Player.py -i "$host" -p "$port" -u "$i" -t "$team2" -P 0 -D 0 \
    > "./runlogs/${team2}_${i}.out" 2>&1 &
  sleep 0.15
done

echo ""
echo "Done. (22 agents started)"
echo "Use ./kill.sh to stop all agents."
