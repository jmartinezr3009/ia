# team_agent_433.py
# Equipo 4-3-3 híbrido: reglas tácticas + micro-acciones PPO
import socket, time, threading, json, os, re, math
import numpy as np
from stable_baselines3 import PPO

# -------- CONFIG --------
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 6000
NUM_PLAYERS = 11
TEAM_NAME = "MY_TEAM"
CONF_FILE = "conf_file.conf"
MODEL_PATH = "models/ppo_rcss_final.zip"  # tu modelo PPO entrenado (único)

# Field bounds (aprox RoboCup)
FIELD_X_MIN, FIELD_X_MAX = -52.5, 52.5
FIELD_Y_MIN, FIELD_Y_MAX = -34.0, 34.0

# Regex
INIT_RE = re.compile(r"\(init\s+([lrLR])\s+(\d+)", re.IGNORECASE)
POS_RE = re.compile(r"\(mypos\s+([\-0-9.]+)\s+([\-0-9.]+)\)", re.IGNORECASE)
BALL_RE = re.compile(r"\(ball\s+([\-0-9.]+)\s+([\-0-9.]+)", re.IGNORECASE)

# Carga modelo (si existe)
MODEL = None
if os.path.exists(MODEL_PATH):
    MODEL = PPO.load(MODEL_PATH)
    print("[INFO] Modelo PPO cargado:", MODEL_PATH)
else:
    print("[WARN] Modelo PPO no encontrado en", MODEL_PATH, "- se usará heurística.")

# Cargar posiciones "home" desde conf_file
def load_positions(conf_file):
    if not os.path.exists(conf_file):
        raise FileNotFoundError(f"No se encontró {conf_file}")
    with open(conf_file, "r") as f:
        data = json.load(f)
    positions = {}
    for i in range(1, NUM_PLAYERS+1):
        entry = data["data"][0].get(str(i))
        if entry is None:
            raise KeyError(f"No hay posición para '{i}' en {conf_file}")
        positions[i] = (float(entry["x"]), float(entry["y"]))
    return positions

# roles según numero (4-3-3)
def role_of(unum):
    if unum == 1:
        return "goalie"
    if 2 <= unum <= 5:
        return "defender"
    if 6 <= unum <= 8:
        return "midfielder"
    return "forward"

# safe send
def safe_send(sock, text):
    try:
        sock.sendto(text.encode(), (SERVER_HOST, SERVER_PORT))
    except Exception:
        pass

# clamp pos to field (small margin)
def clamp_to_field(x,y):
    mx = max(FIELD_X_MIN+1.0, min(FIELD_X_MAX-1.0, x))
    my = max(FIELD_Y_MIN+1.0, min(FIELD_Y_MAX-1.0, y))
    return mx, my

# Heurística táctica: retornar un objetivo (tx,ty) a mantener según rol y balón
def tactical_target(role, home_x, home_y, ballx, bally):
    # goalie: quedarse cerca del arco (home)
    if role == "goalie":
        return home_x, home_y
    # defender: línea entre home y el balón, mantenerse más cerca de home_x
    if role == "defender":
        # empujar un poco hacia la pelota pero no más adelante que la línea defensiva
        tx = (home_x + ballx) * 0.45  # mezcla
        ty = home_y + (ballx - home_x) * 0.05 + (bally - home_y) * 0.2
        return clamp_to_field(tx, ty)
    # midfielder: posicion de apoyo entre balón y delanteros (media cancha)
    if role == "midfielder":
        # colocarse entre home y el balón, pero más adelantado que el defensor
        tx = (ballx * 0.6 + home_x * 0.4)
        ty = (bally * 0.7 + home_y * 0.3)
        return clamp_to_field(tx, ty)
    # forward: presionar la zona rival / acercarse al balón si está avanzado
    if role == "forward":
        # si balón está en zona de ataque, ir al balón, si no, moverse a posiciones de presión
        tx = ballx * 0.8 + home_x * 0.2
        ty = bally * 0.9 + home_y * 0.1
        return clamp_to_field(tx, ty)
    return clamp_to_field(home_x, home_y)

# decide si usar PPO micro o regla alta
def should_use_model(role, px, py, ballx, bally, dist_to_ball):
    # Si estás muy cerca del balón -> usar el modelo (micro decisiones)
    if dist_to_ball < 7.0:
        return True
    # Forwards y midfielders usan modelo más lejos para presionar
    if role in ("forward","midfielder") and dist_to_ball < 20.0:
        return True
    # goalie only micro when very close
    if role == "goalie" and dist_to_ball < 6.0:
        return True
    return False

# Map action (0..4) -> commands (same mapping que RcssGymEnv)
def map_action_to_commands(action, sock, home_x, home_y, ballx, bally):
    try:
        a = int(action)
    except:
        a = 0
    if a == 0:
        safe_send(sock, f"(move {ballx:.2f} {bally:.2f})")
        safe_send(sock, "(dash 60)")
    elif a == 1:
        safe_send(sock, "(dash 75)")
    elif a == 2:
        safe_send(sock, "(turn -30)")
        safe_send(sock, "(dash 40)")
    elif a == 3:
        safe_send(sock, "(turn 30)")
        safe_send(sock, "(dash 40)")
    else:
        safe_send(sock, f"(move {home_x:.2f} {home_y:.2f})")
        safe_send(sock, "(dash 55)")

# hilo por jugador
def player_thread(unum, positions):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", 0))
    sock.settimeout(0.5)

    # init
    safe_send(sock, f"(init {TEAM_NAME})")

    side = None
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
                #unum = int(m.group(2))  # ya tenemos unum por argumento
                break
        except socket.timeout:
            continue

    # home position
    target_pos = positions.get(unum, (-40.0, 0.0))
    if side == "r":
        target_pos = (-target_pos[0], target_pos[1])
    home_x, home_y = float(target_pos[0]), float(target_pos[1])

    # mover a home
    for _ in range(6):
        safe_send(sock, f"(move {home_x:.2f} {home_y:.2f})")
        time.sleep(0.08)

    role = role_of(unum)
    px, py = home_x, home_y
    ballx, bally = 0.0, 0.0

    while True:
        try:
            data, _ = sock.recvfrom(8192)
            msg = data.decode(errors="ignore")
        except socket.timeout:
            msg = ""

        # parse positions
        mpos = POS_RE.search(msg)
        if mpos:
            try:
                px = float(mpos.group(1)); py = float(mpos.group(2))
            except:
                pass
        mball = BALL_RE.search(msg)
        if mball:
            try:
                ballx = float(mball.group(1)); bally = float(mball.group(2))
            except:
                pass

        dx = ballx - px; dy = bally - py
        dist_ball = math.hypot(dx, dy)

        # prevención de bordes: si estamos cerca del borde, girar y entrar
        margin = 2.0
        if px < FIELD_X_MIN + margin:
            safe_send(sock, "(turn 45)"); safe_send(sock, "(dash 50)")
            time.sleep(0.12); continue
        if px > FIELD_X_MAX - margin:
            safe_send(sock, "(turn -45)"); safe_send(sock, "(dash 50)")
            time.sleep(0.12); continue
        if py < FIELD_Y_MIN + margin:
            safe_send(sock, "(turn 90)"); safe_send(sock, "(dash 50)")
            time.sleep(0.12); continue
        if py > FIELD_Y_MAX - margin:
            safe_send(sock, "(turn -90)"); safe_send(sock, "(dash 50)")
            time.sleep(0.12); continue

        # Si el modelo debe manejar el micro-control
        if MODEL is not None and should_use_model(role, px, py, ballx, bally, dist_ball):
            obs = np.array([px, py, ballx, bally, dx, dy, math.hypot(px-home_x, py-home_y)/60.0], dtype=np.float32)
            try:
                action, _ = MODEL.predict(obs, deterministic=False)
                map_action_to_commands(action, sock, home_x, home_y, ballx, bally)
            except Exception:
                # fallback heurístico
                if dist_ball < 10.0:
                    safe_send(sock, f"(move {ballx:.2f} {bally:.2f})"); safe_send(sock, "(dash 60)")
                else:
                    safe_send(sock, f"(move {home_x:.2f} {home_y:.2f})")
            time.sleep(0.09)
            continue

        # Si no usamos modelo: comportamiento táctico/reglas
        tx, ty = tactical_target(role, home_x, home_y, ballx, bally)

        # Si estoy muy lejos de tx,ty me muevo; si estoy cerca intento presionar el balón
        dist_to_target = math.hypot(px-tx, py-ty)
        if dist_ball < 8.0 and role in ("forward","midfielder"):
            # presionar al balón
            safe_send(sock, f"(move {ballx:.2f} {bally:.2f})")
            safe_send(sock, "(dash 60)")
        elif dist_to_target > 3.0:
            # mover a target táctico
            safe_send(sock, f"(move {tx:.2f} {ty:.2f})")
            # dash modulado por distancia
            dash_power = max(30, min(80, 40 + dist_to_target))
            safe_send(sock, f"(dash {dash_power:.1f})")
        else:
            # pequeño ajuste en zona
            angle = (math.degrees(math.atan2(ty-py, tx-px))) if dist_to_target>0.5 else 0
            safe_send(sock, f"(turn {angle:.1f})")
            safe_send(sock, "(dash 20)")

        time.sleep(0.12)

# main
def main():
    positions = load_positions(CONF_FILE)
    threads = []
    for unum in range(1, NUM_PLAYERS+1):
        t = threading.Thread(target=player_thread, args=(unum, positions), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(0.12)
    print("[INFO] Equipo 4-3-3 híbrido arrancado. Ctrl-C para parar.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Detenido.")

if __name__ == "__main__":
    main()
