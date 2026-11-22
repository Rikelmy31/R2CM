import sys
import cv2
import time
import argparse
import itertools
from collections import Counter, deque
from predict import predicted_color
import numpy as np
from draw import draw_2d_cube_state
import helpers
import twophase.solver as sv
import calibrate


# ---------------------------------------------------------------------------
# Câmera (suporta índice local 0/1/2... ou URL de stream do celular)
# ---------------------------------------------------------------------------

# Se quiser deixar uma URL padrão do celular direto no código, coloque aqui;
# caso contrário, deixe None e passe via --cam (ex.: http://192.168.0.20:8080/video)
DEFAULT_CAM_SOURCE = None


def get_average_bgr(frame, cx, cy, radius=3):
    """Calcula a média BGR de uma janela (2*radius+1)x(2*radius+1) ao redor de (cx, cy)."""
    h, w, _ = frame.shape
    cx, cy = int(cx), int(cy)
    x1 = max(0, cx - radius)
    x2 = min(w, cx + radius + 1)
    y1 = max(0, cy - radius)
    y2 = min(h, cy + radius + 1)
    region = frame[y1:y2, x1:x2]
    return np.mean(region, axis=(0, 1))


def abrir_camera(cam_source):
    """Abre webcam local (int) ou stream de celular (URL/RTSP) com menor latência."""
    cap = cv2.VideoCapture(cam_source)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a câmera: {cam_source}")

    # Reduz buffer para evitar atraso em streams de rede/celular
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if isinstance(cam_source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def warmup_camera(cap, num_frames=30, delay=0.05, rotate=0, flip=False):
    """Descarta frames iniciais para a câmera estabilizar exposição, foco e brilho."""
    for _ in range(num_frames):
        ok, frame = cap.read()
        if ok and frame is not None:
            _ = aplicar_orientacao(frame, rotate=rotate, flip=flip)
        if delay > 0:
            time.sleep(delay)


def aplicar_orientacao(frame, rotate=0, flip=False):
    """Aplica rotação (0, 90, 180, 270) e espelhamento horizontal se necessário."""
    rot_map = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    if rotate in rot_map:
        frame = cv2.rotate(frame, rot_map[rotate])
    if flip:
        frame = cv2.flip(frame, 1)
    return frame


def ajustar_frame(frame):
    """Leva qualquer resolução para ~640x480 (altura 480, corta as laterais se for mais largo).
    O filtro de tamanho dos adesivos em find_contours (30 a 60 px) foi feito para 640x480."""
    h, w = frame.shape[:2]
    if h == 480 and w == 640:
        return frame
    escala = 480.0 / h
    frame = cv2.resize(frame, (int(w * escala), 480))
    if frame.shape[1] > 640:
        x0 = (frame.shape[1] - 640) // 2
        frame = frame[:, x0:x0 + 640]
    return frame


def ler_frame(cap, rotate=0, flip=False, tentativas=120):
    """Lê um frame; em stream de rede (celular) tolera falhas momentâneas."""
    for _ in range(tentativas):
        ok, frame = cap.read()
        if ok and frame is not None:
            frame = aplicar_orientacao(frame, rotate=rotate, flip=flip)
            return ajustar_frame(frame)
        time.sleep(0.02)
    raise RuntimeError("Não consegui ler a câmera do celular. Confira a URL/índice e a rede/USB.")


# ---------------------------------------------------------------------------
# Leitura das faces
# ---------------------------------------------------------------------------

class Face:
    def __init__(self, name, class_colors, cam_rotate=0, cam_flip=False, history_size=7, min_votes=5):
        self.class_colors = class_colors
        self.cam_rotate = cam_rotate
        self.cam_flip = cam_flip
        self.face = [[None, None, None],
                     [None, None, None],
                     [None, None, None]]
        self.name = name
        self.scanned = False

        self.history_size = history_size
        self.min_votes = min_votes
        self.samples = [deque(maxlen=self.history_size) for _ in range(9)]

        self.rotations = {
            "white": {"white": "", "yellow": "left-2", "blue": "up-1", "green": "down-1", "orange": "left-1", "red": "right-1"},
            "yellow": {"white": "left-2", "yellow": "", "blue": "left-2", "green": "left-2", "orange": "left-3", "red": "left-1"},
            "blue": {"white": "up-1", "yellow": "up-1", "blue": "", "green": "down-2", "orange": "up-1", "red": "up-1"},
            "green": {"white": "up-1", "yellow": "up-1", "blue": "up-2", "green": "", "orange": "up-1", "red": "up-1"},
            "orange": {"white": "left-1", "yellow": "left-1", "blue": "right-1", "green": "right-1", "orange": "", "red": "right-2"},
            "red": {"white": "left-1", "yellow": "right-1", "blue": "left-1", "green": "left-1", "orange": "left-2", "red": ""}
        }

        self.colors = {
            "White": (255, 255, 255),
            "Yellow": (0, 255, 255),
            "Orange": (0, 165, 255),
            "Red": (0, 0, 255),
            "Green": (0, 255, 0),
            "Blue": (255, 0, 0)
        }

    def find_contours(self, dilatedFrame):
        contours, hierarchy = cv2.findContours(dilatedFrame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        final_contours = []
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.1 * perimeter, True)
            if len(approx) == 4:
                area = cv2.contourArea(contour)
                (x, y, w, h) = cv2.boundingRect(approx)
                ratio = w / float(h)
                if ratio >= 0.8 and ratio <= 1.2 and w >= 30 and w <= 60 and area / (w * h) > 0.4:
                    final_contours.append((x, y, w, h))
        if len(final_contours) < 9:
            return final_contours
        found = False
        contour_neighbors = {}
        for index, contour in enumerate(final_contours):
            (x, y, w, h) = contour
            contour_neighbors[index] = []
            center_x = x + w / 2
            center_y = y + h / 2
            radius = 1.5
            neighbor_positions = [
                [(center_x - w * radius), (center_y - h * radius)],
                [center_x, (center_y - h * radius)],
                [(center_x + w * radius), (center_y - h * radius)],
                [(center_x - w * radius), center_y],
                [center_x, center_y],
                [(center_x + w * radius), center_y],
                [(center_x - w * radius), (center_y + h * radius)],
                [center_x, (center_y + h * radius)],
                [(center_x + w * radius), (center_y + h * radius)],
            ]
            for neighbor in final_contours:
                (x2, y2, w2, h2) = neighbor
                for (x3, y3) in neighbor_positions:
                    if (x2 < x3 and y2 < y3) and (x2 + w2 > x3 and y2 + h2 > y3):
                        contour_neighbors[index].append(neighbor)
        for (contour, neighbors) in contour_neighbors.items():
            if len(neighbors) == 9:
                found = True
                final_contours = neighbors
                break
        if not found:
            return []
        y_sorted = sorted(final_contours, key=lambda item: item[1])
        top_row = sorted(y_sorted[0:3], key=lambda item: item[0])
        middle_row = sorted(y_sorted[3:6], key=lambda item: item[0])
        bottom_row = sorted(y_sorted[6:9], key=lambda item: item[0])
        sorted_contours = top_row + middle_row + bottom_row
        return sorted_contours

    def draw_contours(self, frame, contours):
        for index, (x, y, w, h) in enumerate(contours):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
        return frame

    def get_arrow(self, x, y, w, h, center_x, center_y, middle_color, wanted_color):
        if wanted_color == middle_color:
            return (0, 0), (0, 0), 1
        rotation = self.rotations[middle_color][wanted_color]
        action, times = rotation.split("-")[0], int(rotation.split("-")[1])

        if action == "up":
            start_point = (int(center_x), int(center_y + 2 * h))
            end_point = (int(center_x), int(center_y - 2 * h))
        elif action == "down":
            start_point = (int(center_x), int(center_y - 2 * h))
            end_point = (int(center_x), int(center_y + 2 * h))
        elif action == "left":
            start_point = (int(center_x + 2 * w), int(center_y))
            end_point = (int(center_x - 2 * w), int(center_y))
        elif action == "right":
            start_point = (int(center_x - 2 * w), int(center_y))
            end_point = (int(center_x + 2 * w), int(center_y))
        return start_point, end_point, times

    def _stabilized_color(self, idx, prediction):
        self.samples[idx].append(prediction)
        if len(self.samples[idx]) < self.min_votes:
            return None

        most_common_color, votes = Counter(self.samples[idx]).most_common(1)[0]
        if votes >= self.min_votes:
            return most_common_color
        return None

    def scan(self, cap, wanted_color):
        while True:
            frame = ler_frame(cap, rotate=self.cam_rotate, flip=self.cam_flip)
            raw = frame.copy()

            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurredFrame = cv2.blur(grayFrame, (3, 3))
            cannyFrame = cv2.Canny(blurredFrame, 30, 60, 3)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            dilatedFrame = cv2.dilate(cannyFrame, kernel)

            contours = self.find_contours(dilatedFrame)
            frame = self.draw_contours(frame, contours)

            cv2.putText(frame, f"Show the {wanted_color} face", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[wanted_color.capitalize()], 2, cv2.LINE_AA)
            if len(contours) == 9:
                middle = contours[4]
                x, y, w, h = middle[0], middle[1], middle[2], middle[3]
                center_x = x + w / 2
                center_y = y + h / 2

                color = get_average_bgr(raw, center_x, center_y, radius=3)
                middle_color = predicted_color(color, self.class_colors)

                start_point, end_point, times = self.get_arrow(x, y, w, h, center_x, center_y, middle_color, wanted_color)
                frame = cv2.arrowedLine(frame, start_point, end_point, self.colors[wanted_color.capitalize()], 5)

                if middle_color == wanted_color:
                    for idx, i in enumerate(contours):
                        x, y, w, h = i[0], i[1], i[2], i[3]
                        center_x = x + w / 2
                        center_y = y + h / 2

                        color = get_average_bgr(raw, center_x, center_y, radius=3)
                        prediction = predicted_color(color, self.class_colors)
                        stable_color = self._stabilized_color(idx, prediction)

                        display_label = stable_color if stable_color is not None else f"{prediction}?"
                        if stable_color is not None:
                            self.update(idx, stable_color)

                        cv2.putText(frame, display_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 0, 0), -1)
                else:
                    cv2.putText(frame, 'Not the correct color', (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            image = draw_2d_cube_state(frame, faces)
            cv2.imshow('frame', image)
            key_pressed = cv2.waitKey(1) & 0xFF
            if key_pressed == 27 or key_pressed == ord('q'):
                sys.exit()
            elif key_pressed == 8 or key_pressed == 127:
                return True

            if self.scanned:
                return False

    def update(self, index, color):
        if not self.scanned:
            col = index % 3
            row = index // 3
            self.face[row][col] = color

        if None not in self.face[0] and None not in self.face[1] and None not in self.face[2]:
            self.scanned = True
            if self.name == 'white':
                self.face = np.rot90(self.face, 3)
            elif self.name == 'orange':
                self.face = np.rot90(self.face, 1)
            elif self.name == 'yellow':
                self.face = np.rot90(self.face, 3)
            elif self.name == 'red':
                self.face = np.rot90(self.face, 3)
            elif self.name == 'blue':
                self.face = np.rot90(self.face, 2)
            elif self.name == 'green':
                self.face = np.rot90(self.face, 0)

    def flatten(self):
        return [color for row in self.face for color in row]


def all_scanned(faces):
    """Check if all faces have been scanned."""
    for face in faces.values():
        if not face.scanned:
            return False
    return True


# ---------------------------------------------------------------------------
# String do cubo (Kociemba) + verificação das peças
# ---------------------------------------------------------------------------

ORDEM = ['yellow', 'green', 'red', 'white', 'blue', 'orange']     # U R F D L B
LETRA = dict(zip(ORDEM, "URFDLB"))
FACES_K = "URFDLB"


def _pos_facelet(face, i, j):
    """Posição 3D (peça) e normal do adesivo (linha i, coluna j) de uma face, vista de fora."""
    if face == 'U': return (j - 1, 1, i - 1), (0, 1, 0)
    if face == 'R': return (1, 1 - i, 1 - j), (1, 0, 0)
    if face == 'F': return (j - 1, 1 - i, 1), (0, 0, 1)
    if face == 'D': return (j - 1, -1, 1 - i), (0, -1, 0)
    if face == 'L': return (-1, 1 - i, j - 1), (-1, 0, 0)
    if face == 'B': return (1 - j, 1 - i, -1), (0, 0, -1)


_FACELETS = [_pos_facelet(f, i, j) for f in FACES_K for i in range(3) for j in range(3)]


def _agrupar_pecas():
    d = {}
    for idx, (pos, _) in enumerate(_FACELETS):
        d.setdefault(pos, []).append(idx)
    return [v for v in d.values() if len(v) >= 2]


_PECAS = _agrupar_pecas()
_ARESTAS = [p for p in _PECAS if len(p) == 2]
_CANTOS = [p for p in _PECAS if len(p) == 3]
_RESOLVIDO = "".join(f * 9 for f in FACES_K)
_ARESTAS_OK = {frozenset(_RESOLVIDO[i] for i in p) for p in _ARESTAS}
_CANTOS_OK = {frozenset(_RESOLVIDO[i] for i in p) for p in _CANTOS}


def problemas(s):
    """Descreve o que há de impossível na string (lista vazia = todas as peças existem)."""
    if len(s) != 54:
        return [f"tamanho {len(s)} (deveria ser 54)"]
    out = []
    for l in FACES_K:
        if s.count(l) != 9:
            out.append(f"letra {l} aparece {s.count(l)}x (deveria ser 9)")
    vistas = set()
    for tipo, pecas, validas in (("aresta", _ARESTAS, _ARESTAS_OK),
                                 ("canto", _CANTOS, _CANTOS_OK)):
        for p in pecas:
            cores = frozenset(s[i] for i in p)
            nome = "-".join(s[i] for i in p)
            onde = ", ".join(f"{FACES_K[i // 9]}{i % 9 + 1}" for i in p)
            if cores not in validas:
                out.append(f"{tipo} inexistente {nome} (adesivos {onde})")
            elif len(cores) == len(p):
                if (tipo, cores) in vistas:
                    out.append(f"{tipo} repetida {nome} (adesivos {onde})")
                vistas.add((tipo, cores))
    return out


def _girar9(face9, k):
    """Gira uma face (lista de 9) k quartos de volta no sentido horário."""
    g = [face9[0:3], face9[3:6], face9[6:9]]
    for _ in range(k % 4):
        g = [list(r) for r in zip(*g[::-1])]
    return [c for r in g for c in r]


def candidatos_rotacao(blocos):
    """Todas as combinações de rotação das 6 faces que dão um cubo com peças válidas,
    das que mexem em menos faces para as que mexem em mais."""
    achados = []
    for ks in itertools.product(range(4), repeat=6):
        s = "".join("".join(_girar9(blocos[f], ks[f])) for f in range(6))
        if not problemas(s):
            achados.append((ks, s))
    achados.sort(key=lambda c: sum(1 for k in c[0] if k))
    return achados


def blocos_das_faces(faces):
    """6 blocos de 9 letras (U R F D L B). Cada adesivo vira a letra da face cujo centro tem a cor dele."""
    blocos = []
    for nome in ORDEM:
        centro = faces[nome].face[1][1]
        if centro != nome:
            print(f"[AVISO] o centro da face {nome} foi lido como '{centro}'. Recalibre as cores.")
        blocos.append([LETRA[c] for c in faces[nome].flatten()])
    return blocos


def resolver_cubo(faces, max_len=24, timeout=5):
    """Monta a string, confere as peças, corrige orientação de face se preciso e chama o Kociemba."""
    blocos = blocos_das_faces(faces)
    original = "".join("".join(b) for b in blocos)

    print("\ncube_string:", original)
    for f in blocos:
        print("  ", "".join(f[0:3]), "".join(f[3:6]), "".join(f[6:9]))
    for l in FACES_K:
        print(f"  {l}: {original.count(l)}", end="")
    print()

    ps = problemas(original)
    if ps:
        print("\nProblemas na leitura:")
        for p in ps:
            print("  -", p)

    cands = candidatos_rotacao(blocos)
    if not cands:
        print("\nNenhuma rotação de face deixa o cubo válido: há adesivo com COR classificada errada "
              "(veja as peças acima). Melhore a luz, recalibre (-c) e escaneie de novo.")
        return None
    if len(cands) > 1:
        print(f"\n[AVISO] {len(cands)} combinações de rotação são válidas; testando em ordem.")

    for ks, s in cands[:20]:
        res = sv.solve(s, max_len, timeout)
        if isinstance(res, str) and res.startswith("Error"):
            print("  solver rejeitou:", res)
            continue
        if any(ks):
            ajustes = {FACES_K[i]: ks[i] for i in range(6) if ks[i]}
            print(f"\n[CORRIGIDO] faces giradas (quartos de volta horários): {ajustes}")
            print("Para não precisar disso, some esses giros ao np.rot90 da face em Face.update.")
        return res
    print("\nO solver rejeitou todas as combinações válidas (torção/paridade): provável adesivo trocado "
          "entre duas cores parecidas.")
    return None


# ---------------------------------------------------------------------------
# Solução -> comandos do robô (simulação no terminal ou BLE com o ESP32)
# ---------------------------------------------------------------------------

def parse_solution(sol):
    """'R1 U3 F2 (3f)' -> ['R', "U'", 'F2']"""
    out = []
    for m in sol.split('(')[0].split():
        face, n = m[0], m[1]
        out.append({'1': face, '2': face + '2', '3': face + "'"}[n])
    return out


PAUSA = 0.15
robo = None

MOTOR = {'F': 'R', 'R': 'G', 'B': 'O', 'L': 'B'}
GARRA = {'R': '1', 'O': '2', 'B': '3', 'G': '4'}
NEUTRO, HORARIO, ANTI = '9', '1', '0'


def enviar(cmd, comentario=""):
    """Sem --ble só imprime; com --ble manda ao ESP32 e espera o 'OK' dele."""
    print(f"  >> {cmd:<3} {comentario}")
    if robo is not None:
        robo.enviar(cmd)
    else:
        time.sleep(PAUSA)


def girar_face(m, anti=False, duplo=False):
    g = GARRA[m]
    if duplo:
        enviar(m + ANTI, "pré-posiciona em 0° (garra ainda aberta)")
        enviar(g + 'C', "fecha garra")
        enviar(m + HORARIO, "gira 180° (meia volta)")
    else:
        enviar(g + 'C', "fecha garra")
        enviar(m + (ANTI if anti else HORARIO),
               "gira 90° anti-horário" if anti else "gira 90° horário")
    enviar(g + 'O', "abre garra")
    enviar(m + NEUTRO, "volta ao neutro")


def tombar(volta=False):
    """Gira o cubo inteiro 90° segurando pelas garras 1 e 2 (vermelha/laranja)."""
    enviar('1C', "segura pela vermelha")
    enviar('2C', "segura pela laranja")
    enviar('3O', "solta azul")
    enviar('4O', "solta verde")
    if not volta:
        enviar('R1', "tomba o cubo (U vai para o lado verde)")
        enviar('O0', "     (laranja gira junto, em sentido oposto)")
    else:
        enviar('R0', "desfaz o tombamento")
        enviar('O1', "     (laranja gira junto)")
    enviar('3C', "fixa pela azul")
    enviar('4C', "fixa pela verde")
    enviar('1O', "solta vermelha")
    enviar('2O', "solta laranja")
    enviar('R9', "vermelha ao neutro")
    enviar('O9', "laranja ao neutro")


def launch_cube(solution):
    modo = "BLE" if robo is not None else "SIMULAÇÃO"
    print(f"\n=== {modo} ({len(solution)} movimentos) ===")
    print(solution)
    for i, passo in enumerate(solution, 1):
        face = passo[0]
        duplo = passo.endswith('2')
        anti = passo.endswith("'")
        print(f"\n[{i}/{len(solution)}] Movimento {passo}")

        if face in MOTOR:
            girar_face(MOTOR[face], anti, duplo)
        else:
            tombar()
            girar_face('G' if face == 'U' else 'B', anti, duplo)
            tombar(volta=True)
    print(f"\n=== FIM ({modo}) ===")


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--calibrate", action='store_true', default=False,
                    required=False, help="Calibrate the colors.")
parser.add_argument("--cam", default=DEFAULT_CAM_SOURCE if DEFAULT_CAM_SOURCE is not None else "0",
                    help="Índice da câmera (0, 1...) ou URL do celular, ex.: http://192.168.0.20:8080/video")
parser.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0,
                    help="Rotaciona o frame da câmera do celular (0, 90, 180 ou 270 graus).")
parser.add_argument("--flip", action='store_true', default=False,
                    help="Espelha horizontalmente o frame da câmera.")
parser.add_argument("--ble", nargs="?", const="CuboRobo", default=None, metavar="NOME",
                    help="Envia os comandos ao ESP32 por BLE (nome anunciado, padrão CuboRobo).")
parser.add_argument("--3d", dest="visual3d", action="store_true", default=False,
                    help="Mostra a solução no visualizador 3D (PyCube) em vez de executar o robô.")
args = parser.parse_args()

calibrate_colors = args.calibrate


if __name__ == '__main__':
    colors = helpers.Colors()
    cam_source = int(args.cam) if str(args.cam).isdigit() else args.cam

    if calibrate_colors:
        calibrate.run(colors, cam_source, args.rotate, args.flip)

    faces = {
        'white': Face('white', colors, cam_rotate=args.rotate, cam_flip=args.flip),
        'orange': Face('orange', colors, cam_rotate=args.rotate, cam_flip=args.flip),
        'yellow': Face('yellow', colors, cam_rotate=args.rotate, cam_flip=args.flip),
        'red': Face('red', colors, cam_rotate=args.rotate, cam_flip=args.flip),
        'blue': Face('blue', colors, cam_rotate=args.rotate, cam_flip=args.flip),
        'green': Face('green', colors, cam_rotate=args.rotate, cam_flip=args.flip),
    }

    faces_list = ["white", "orange", "yellow", "red", "blue", "green"]
    idx = 0

    cap = abrir_camera(cam_source)
    warmup_camera(cap, num_frames=30, delay=0.05, rotate=args.rotate, flip=args.flip)

    while not all_scanned(faces):
        key = faces_list[idx]
        redo = faces[key].scan(cap, key)
        if redo and idx - 1 >= 0:
            last_key = faces_list[idx - 1]
            faces[last_key] = Face(last_key, colors, cam_rotate=args.rotate, cam_flip=args.flip)
            idx -= 1
        elif not redo:
            idx += 1

    cap.release()
    cv2.destroyAllWindows()

    print("Resolvendo o cubo...")
    raw_solution = resolver_cubo(faces)

    if raw_solution is None:
        print("\n[ERRO DE LEITURA] O cubo não pôde ser resolvido (detalhes acima).")
        print("Dica: boa iluminação, sem reflexo, e escaneie as 6 faces novamente.")
        sys.exit(1)

    print("Solução calculada com sucesso:", raw_solution)
    movimentos = parse_solution(raw_solution)

    if args.visual3d:
        from PyCube import PyCube
        PyCube().run([x for m in movimentos for x in ([m[0], m[0]] if m.endswith('2') else [m])])
    else:
        if args.ble:
            from ble_robo import BLERobo
            robo = BLERobo(args.ble)
        try:
            launch_cube(movimentos)
        finally:
            if robo is not None:
                robo.fechar()