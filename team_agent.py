import socket
import time
import threading
import json
import os
import re
import random
import math

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 6000
NUM_PLAYERS = 11
TEAM_NAME = "MY_TEAM"
CONF_FILE = "conf_file.conf"

INIT_RE = re.compile(r"\(init\s+([lrLR])\s+(\d+)", re.IGNORECASE)
POS_RE = re.compile(r"\(mypos\s+([\-0-9.]+)\s+([\-0-9.]+)\)", re.IGNORECASE)

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
    except Exception:
        pass

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def random_move_with_bounds(sock, unum, home_x, home_y):
    """
    Movimiento con límites y control de distancia desde la posición 'home' asignada.
    """
    FIELD_X_MIN, FIELD_X_MAX = -52.5, 52.5
    FIELD_Y_MIN, FIELD_Y_MAX = -34.0, 34.0

    # Radios máximos permitidos desde home position (arquero = unum 1)
    if unum == 1:
        MAX_DIST_FROM_HOME = 18.0  # arquero se queda cerca
        GOALIE_DASH_MAX = 40
    else:
        MAX_DIST_FROM_HOME = 45.0
        GOALIE_DASH_MAX = 70

    # iniciar estimación de posición en la posición 'home' (evitamos 0,0)
    px, py = float(home_x), float(home_y)
    last_pos_time = time.time()

    while True:
        try:
            # Intentamos leer mensajes para actualizar posición si vienen
            try:
                data, _ = sock.recvfrom(4096)
                msg = data.decode(errors="ignore")
                m = POS_RE.search(msg)
                if m:
                    px = float(m.group(1))
                    py = float(m.group(2))
                    last_pos_time = time.time()
            except socket.timeout:
                pass
            except Exception:
                pass

            # Si no hemos recibido actualizaciones por mucho tiempo,
            # nos basamos en la posición estimada (home) y forzamos
            # movimientos controlados en vez de dashes grandes.
            stale = (time.time() - last_pos_time) > 2.0

            # Distancia desde home
            dist_home = math.hypot(px - home_x, py - home_y)

            # ===========================
            #   SI SE ALEJÓ DEMASIADO DE SU CASA, REGRESAR
            # ===========================
            if dist_home > MAX_DIST_FROM_HOME:
                # mover directamente hacia la home_pos varias veces
                for _ in range(3):
                    safe_send(sock, f"(move {home_x:.2f} {home_y:.2f})")
                    # dash moderado para volver más rápido
                    dash_power = 60 if unum != 1 else GOALIE_DASH_MAX
                    safe_send(sock, f"(dash {dash_power:.1f})")
                    time.sleep(0.12)
                # actualizar la estimación para evitar ciclos
                px, py = float(home_x), float(home_y)
                last_pos_time = time.time()
                time.sleep(0.15)
                continue

            # ===========================
            #   CONTROL DE BORDES
            # ===========================
            # Si estamos demasiado cerca de un borde, giramos hacia adentro
            danger = False

            # margen de seguridad desde el borde
            MARGIN = 3.0

            if px < FIELD_X_MIN + MARGIN:
                # si estamos en el borde izquierdo, girar hacia la derecha
                safe_send(sock, "(turn 45)")
                safe_send(sock, f"(dash {50 if not stale else 35})")
                danger = True
            elif px > FIELD_X_MAX - MARGIN:
                safe_send(sock, "(turn -45)")
                safe_send(sock, f"(dash {50 if not stale else 35})")
                danger = True

            if py < FIELD_Y_MIN + MARGIN:
                safe_send(sock, "(turn 90)")
                safe_send(sock, f"(dash {50 if not stale else 35})")
                danger = True
            elif py > FIELD_Y_MAX - MARGIN:
                safe_send(sock, "(turn -90)")
                safe_send(sock, f"(dash {50 if not stale else 35})")
                danger = True

            if danger:
                # damos tiempo a que el jugador vuelva dentro y actualice posición
                time.sleep(0.18)
                continue

            # ===========================
            #   MOVIMIENTO NORMAL CONTROLADO
            # ===========================
            # Para el arquero movemos con menos agresividad
            if unum == 1:
                angle = random.uniform(-20, 20)
                power = random.uniform(10, 45)
            else:
                angle = random.uniform(-40, 40)
                power = random.uniform(15, 70)

            # Si estamos muy cerca de home, mover un poco alrededor
            if dist_home < 6.0:
                # movimiento más suave alrededor de la zona
                angle = random.uniform(-60, 60)
                power = random.uniform(5, 40) if unum == 1 else random.uniform(10, 55)

            safe_send(sock, f"(turn {angle:.1f})")
            time.sleep(0.07)
            safe_send(sock, f"(dash {power:.1f})")

            # espera corta antes del siguiente ciclo
            time.sleep(random.uniform(0.35, 1.0))

        except Exception:
            break

def player_thread(idx, positions):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", 0))
    sock.settimeout(1.0)

    print(f"[Jugador {idx}] Enviando init...")
    safe_send(sock, f"(init {TEAM_NAME})")

    side = None
    unum = None
    init_buf = ""
    t0 = time.time()

    # Esperar init
    while time.time() - t0 < 5:
        try:
            data, _ = sock.recvfrom(8192)
            msg = data.decode(errors="ignore")
            init_buf += msg
            m = INIT_RE.search(init_buf)
            if m:
                side = m.group(1).lower()
                unum = int(m.group(2))
                print(f"[Jugador {idx}] Init detectado: side={side}, unum={unum}")
                break
        except socket.timeout:
            continue
        except Exception:
            break

    if unum is None:
        print(f"[Jugador {idx}] ❌ No se detectó init.")
        sock.close()
        return

    # Obtener posición inicial (home)
    target_pos = positions.get(unum) or positions.get(idx)
    if target_pos:
        x, y = target_pos
    else:
        x, y = (-40.0, 0.0)

    # reflejar si el equipo está en el lado derecho
    if side == "r":
        x = -x

    home_x, home_y = float(x), float(y)

    print(f"[Jugador {unum}] Moviéndose a posición inicial ({home_x:.2f}, {home_y:.2f})")

    # mandar move varias veces para asegurar posición inicial
    for _ in range(6):
        safe_send(sock, f"(move {home_x:.2f} {home_y:.2f})")
        time.sleep(0.10)

    # Iniciar movimiento con límites y respetando home_pos
    threading.Thread(target=random_move_with_bounds, args=(sock, unum, home_x, home_y), daemon=True).start()

    # Mantener socket vivo
    while True:
        try:
            sock.recvfrom(8192)
        except socket.timeout:
            continue
        except:
            break

def main():
    try:
        positions = load_positions(CONF_FILE)
    except Exception as e:
        print("ERROR cargando conf:", e)
        return

    threads = []
    for i in range(1, NUM_PLAYERS + 1):
        t = threading.Thread(target=player_thread, args=(i, positions), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(0.12)

    print("[INFO] Jugadores iniciados (con límites y home-pos).")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[INFO] Equipo detenido.")

if __name__ == "__main__":
    main()
