#!/bin/bash

echo "🚀 Iniciando equipo JulianaFC (11 jugadores)..."

for i in {1..11}
do
    echo "→ Lanzando jugador $i"
   python3 team_agent_rl.py $i &
    sleep 0.3
done

echo "✔ Todos los jugadores están ejecutándose en segundo plano."
