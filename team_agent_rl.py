# team_agent_rl.py
import socket
import time
import threading
import json
import os
import re
import math
import numpy as np
from stable_baselines3 import PPO

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 6000
NUM_PLAYERS = 11
TEAM_NAME = "MY_TEAM"
CONF_FILE = "conf_file.conf"

INIT_RE = re.compile(r"\(init\s+([lrLR])\s+(\d+)", re.IGNORECASE)
POS_RE = re.compile(r"\(mypos\s+([\-0-9.]+)\s+([\-0-9.]+)\)", re.IGNORECASE)
BALL_RE = re.compile(r"\(ball\s+([\-0-9.]+)\s+([\-0-9.]+)", re.IGNORECASE)

MODEL_PATH = "models/ppo_rcss_final.zip"  # modelo entrenado

def load_positions(conf_file):
    if not os.path.exists(conf_file):
        raise FileNotFoundError(f"No se encontró {conf_file}")
    with open(conf_file, "r") as f:
        data = json.load(f)
    positions = {}
    for i in range(1, NUM_PLAYERS + 1):
        entry = data["data"][0].get(str(i))
        if entry is None:
            raise KeyError(f"No hay posición para '{i}' en {conf_file}")
        positions[i] = (float(entry["x"]), float(entry["y"]))
    return positions

def safe_send(sock, text):
    try:
        sock.sendto(text.encode(), (SERVER_HOST, SERVER_PORT))
    except:
        pass

# Cargar modelo una sola vez (política compartida)
MODEL = None
if os.path.exists(MODEL_PATH):
    MODEL = PPO.load(MODEL_PATH)
    print("[INFO] Modelo RL cargado:", MODEL_PATH)
else:
    print("[WARN] Modelo no encontrado en", MODEL_PATH, "— ejecuta train_rl.py primero para generar uno.")

def player_thread(idx, positions):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", 0))
    sock.settimeout(0.5)

    safe_send(sock, f"(init {TEAM_NAME})")

    side = None
    unum = None
    init_buf = ""
    t0 = time.time()
    while time.time() - t0 < 5:
        try:
            data, _ = sock.recvfrom(8192)
            msg = data.decode(errors="ignore")
            init_buf += msg
            m = INIT_RE.search(init_buf)
            if m:
                side = m.group(1).lower()
                unum = int(m.group(2))
                break
        except socket.timeout:
            continue

    if unum is None:
        sock.close()
        return

    target_pos = positions.get(unum) or positions.get(idx) or (-40.0, 0.0)
    if side == "r":
        target_pos = (-target_pos[0], target_pos[1])
    home_x, home_y = float(target_pos[0]), float(target_pos[1])

    # mandar move inicial
    for _ in range(6):
        safe_send(sock, f"(move {home_x:.2f} {home_y:.2f})")
        time.sleep(0.08)

    px, py = home_x, home_y
    ballx, bally = 0.0, 0.0
    last_ball_dist = math.hypot(ballx-px, bally-py)

    while True:
        # recolectar mensajes
        try:
            data, _ = sock.recvfrom(8192)
            msg = data.decode(errors="ignore")
        except socket.timeout:
            msg = ""
        # actualizar pos y pelota si vienen
        mpos = POS_RE.search(msg)
        if mpos:
            px = float(mpos.group(1)); py = float(mpos.group(2))
        mball = BALL_RE.search(msg)
        if mball:
            ballx = float(mball.group(1)); bally = float(mball.group(2))

        dx = ballx - px; dy = bally - py
        dist_home = math.hypot(px-home_x, py-home_y)

        obs = np.array([px, py, ballx, bally, dx, dy, dist_home/60.0], dtype=np.float32)

        # si no hay modelo, fallback a comportamiento heurístico
        if MODEL is None:
            # heurística simple: si cerca del balón, ir; si lejos, mantener home
            if math.hypot(dx,dy) < 10.0:
                safe_send(sock, f"(move {ballx:.2f} {bally:.2f})")
                safe_send(sock, "(dash 60)")
            else:
                safe_send(sock, f"(move {home_x:.2f} {home_y:.2f})")
            time.sleep(0.12)
            continue

        # usar modelo para predecir acción
        action, _ = MODEL.predict(obs, deterministic=False)
        # mapear acción discreta a comandos
        if action == 0:
            safe_send(sock, f"(move {ballx:.2f} {bally:.2f})")
            safe_send(sock, "(dash 60)")
        elif action == 1:
            safe_send(sock, "(dash 75)")
        elif action == 2:
            safe_send(sock, "(turn -30)")
            safe_send(sock, "(dash 40)")
        elif action == 3:
            safe_send(sock, "(turn 30)")
            safe_send(sock, "(dash 40)")
        elif action == 4:
            safe_send(sock, f"(move {home_x:.2f} {home_y:.2f})")
            safe_send(sock, "(dash 55)")

        time.sleep(0.09)

def main():
    positions = load_positions(CONF_FILE)
    threads = []
    for i in range(1, NUM_PLAYERS+1):
        t = threading.Thread(target=player_thread, args=(i, positions), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(0.12)
    print("[INFO] Equipo RL arrancado. Ctrl-C para parar.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Detenido.")

if __name__ == "__main__":
    main()
