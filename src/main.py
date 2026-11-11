import sys
import cv2
import time
import argparse
from predict import predicted_color
import numpy as np
from draw import draw_2d_cube_state
import helpers
import twophase.solver as sv
import calibrate


def sample(img, cx, cy, r=6):
    """Mediana de um pequeno bloco em volta do centro do adesivo (menos ruído que 1 pixel)."""
    h, w = img.shape[:2]
    y0, y1 = max(int(cy) - r, 0), min(int(cy) + r, h)
    x0, x1 = max(int(cx) - r, 0), min(int(cx) + r, w)
    patch = img[y0:y1, x0:x1]
    return np.median(patch.reshape(-1, 3), axis=0)


class Face:
    def __init__(self, name, class_colors):
        self.class_colors = class_colors
        self.face = [[None, None, None],
                     [None, None, None],
                     [None, None, None]]
        self.name = name
        self.scanned = False

        self.rotations = {"white": {"white": "", "yellow": "left-2", "blue": "up-1", "green": "down-1", "orange": "left-1", "red": "right-1"},
                          "yellow": {"white": "left-2", "yellow": "", "blue": "left-2", "green": "left-2", "orange": "left-3", "red": "left-1"},
                          "blue": {"white": "up-1", "yellow": "up-1", "blue": "", "green": "down-2", "orange": "up-1", "red": "up-1"},
                          "green": {"white": "up-1", "yellow": "up-1", "blue": "up-2", "green": "", "orange": "up-1", "red": "up-1"},
                          "orange": {"white": "left-1", "yellow": "left-1", "blue": "right-1", "green": "right-1", "orange": "", "red": "right-2"},
                          "red": {"white": "left-1", "yellow": "right-1", "blue": "left-1", "green": "left-1", "orange": "left-2", "red": ""}
                          }

        self.colors = {"White": (255, 255, 255), "Yellow": (0, 255, 255), "Orange": (0, 165, 255), "Red": (0, 0, 255), "Green": (0, 255, 0), "Blue": (255, 0, 0)}

    def find_contours(self, dilatedFrame):
        contours, hierarchy = cv2.findContours(dilatedFrame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        final_contours = []
        # Filter contours to find squares
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.1 * perimeter, True)
            if len(approx) == 4:
                area = cv2.contourArea(contour)
                (x, y, w, h) = cv2.boundingRect(approx)
                # Find aspect ratio of the shape
                ratio = w / float(h)
                # Check if contour is close to a square
                if ratio >= 0.8 and ratio <= 1.2 and w >= 30 and w <= 60 and area / (w * h) > 0.4:
                    final_contours.append((x, y, w, h))
        # exit early if there aren't enough contours
        if len(final_contours) < 9:
            return final_contours
        # Find the contour that has 9 neighbors
        found = False
        contour_neighbors = {}
        for index, contour in enumerate(final_contours):
            (x, y, w, h) = contour
            contour_neighbors[index] = []
            center_x = x + w / 2
            center_y = y + h / 2
            radius = 1.5
            neighbor_positions = [
                [(center_x - w * radius), (center_y - h * radius)],  # top left
                [center_x, (center_y - h * radius)],                 # top middle
                [(center_x + w * radius), (center_y - h * radius)],  # top right
                [(center_x - w * radius), center_y],                 # middle left
                [center_x, center_y],                                # center
                [(center_x + w * radius), center_y],                 # middle right
                [(center_x - w * radius), (center_y + h * radius)],  # bottom left
                [center_x, (center_y + h * radius)],                 # bottom middle
                [(center_x + w * radius), (center_y + h * radius)],  # bottom right
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
        # Sort all the contours on their X and Y
        y_sorted = sorted(final_contours, key=lambda item: item[1])
        # Split into 3 rows and sort each row on the x-value.
        top_row = sorted(y_sorted[0:3], key=lambda item: item[0])
        middle_row = sorted(y_sorted[3:6], key=lambda item: item[0])
        bottom_row = sorted(y_sorted[6:9], key=lambda item: item[0])
        sorted_contours = top_row + middle_row + bottom_row
        return sorted_contours

    def draw_contours(self, frame, contours):
        """Draw contours onto the given frame."""
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

    def scan(self, cap, wanted_color):
        while True:

            # read frame
            ret, frame = cap.read()
            raw = frame.copy()   # cópia limpa (sem desenhos) usada para ler as cores

            # process frame
            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurredFrame = cv2.blur(grayFrame, (3, 3))
            cannyFrame = cv2.Canny(blurredFrame, 30, 60, 3)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            dilatedFrame = cv2.dilate(cannyFrame, kernel)

            # find contours and draw them
            contours = self.find_contours(dilatedFrame)
            frame = self.draw_contours(frame, contours)

            # write wanted color on frame bottom left
            cv2.putText(frame, f"Show the {wanted_color} face", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[wanted_color.capitalize()], 2, cv2.LINE_AA)
            if len(contours) == 9:
                middle = contours[4]
                x = middle[0]
                y = middle[1]
                w = middle[2]
                h = middle[3]
                center_x = x + w / 2
                center_y = y + h / 2
                color = sample(raw, center_x, center_y)
                middle_color = predicted_color(color, self.class_colors)

                start_point, end_point, times = self.get_arrow(x, y, w, h, center_x, center_y, middle_color, wanted_color)
                frame = cv2.arrowedLine(frame, start_point, end_point, self.colors[wanted_color.capitalize()], 5)

                if middle_color == wanted_color:
                    for idx, i in enumerate(contours):
                        x = i[0]
                        y = i[1]
                        w = i[2]
                        h = i[3]
                        center_x = x + w / 2
                        center_y = y + h / 2
                        # cor do adesivo (mediana de um bloco, na imagem limpa)
                        color = sample(raw, center_x, center_y)
                        prediction = predicted_color(color, self.class_colors)
                        self.update(idx, prediction)
                        # write the color name on the contour
                        cv2.putText(frame, prediction, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 0, 0), -1)
                else:
                    cv2.putText(frame, 'Not the correct color', (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            image = draw_2d_cube_state(frame, faces)
            cv2.imshow('frame', image)
            key_pressed = cv2.waitKey(1) & 0xFF
            if key_pressed == 27 or key_pressed == ord('q'):
                print("Exiting...")
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
            print('Face {} scanned!'.format(self.name))

    def flatten(self):
        return [color for row in self.face for color in row]


def all_scanned(faces):
    """Check if all faces have been scanned."""
    for face in faces.values():
        if not face.scanned:
            return False
    return True


# ---------------------------------------------------------------------------
# Solução -> comandos do Arduino (simulação por serial/terminal)
# ---------------------------------------------------------------------------

def parse_solution(sol):
    """'R1 U3 F2 (3f)' -> ['R', "U'", 'F2']"""
    out = []
    for m in sol.split('(')[0].split():
        face, n = m[0], m[1]
        out.append({'1': face, '2': face + '2', '3': face + "'"}[n])
    return out


MODO_SIMULACAO = True    # mude para False quando tiver o Arduino
PAUSA = 0.15             # pausa entre comandos, só para acompanhar a leitura

# posição no solver -> motor (cor) da face
MOTOR = {'F': 'R', 'R': 'G', 'B': 'O', 'L': 'B'}   # R=vermelha G=verde O=laranja B=azul
# motor -> número da garra (igual ao seu sketch)
GARRA = {'R': '1', 'O': '2', 'B': '3', 'G': '4'}

# Convenção do giro: 9 = neutro (90°), 1 = 180° (horário), 0 = 0° (anti-horário)
NEUTRO, HORARIO, ANTI = '9', '1', '0'


def enviar(cmd, comentario=""):
    """Hoje só imprime. Depois troque por: ser.write((cmd + "\n").encode())"""
    print(f"  >> {cmd:<3} {comentario}")
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
    print(f"\n=== SIMULAÇÃO SERIAL ({len(solution)} movimentos) ===")
    print(solution)
    for i, passo in enumerate(solution, 1):
        face = passo[0]
        duplo = passo.endswith('2')
        anti = passo.endswith("'")
        print(f"\n[{i}/{len(solution)}] Movimento {passo}")

        if face in MOTOR:
            girar_face(MOTOR[face], anti, duplo)
        else:  # U ou D
            tombar()
            girar_face('G' if face == 'U' else 'B', anti, duplo)
            tombar(volta=True)
    print("\n=== FIM DA SIMULAÇÃO ===")


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--calibrate", action='store_true', default=False,
                    required=False, help="Calibrate the colors.")
args = parser.parse_args()

calibrate_colors = args.calibrate


if __name__ == '__main__':
    colors = helpers.Colors()
    if calibrate_colors:
        calibrate.run(colors)

    faces = {
        'white': Face('white', colors),
        'orange': Face('orange', colors),
        'yellow': Face('yellow', colors),
        'red': Face('red', colors),
        'blue': Face('blue', colors),
        'green': Face('green', colors),
    }

    faces_list = ["white", "orange", "yellow", "red", "blue", "green"]
    idx = 0

    # open camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    time.sleep(0.5)

    while not all_scanned(faces):
        key = faces_list[idx]
        redo = faces[key].scan(cap, key)
        if redo and idx - 1 >= 0:
            last_key = faces_list[idx - 1]
            faces[last_key] = Face(last_key, colors)
            idx -= 1
        elif not redo:
            idx += 1

    cap.release()
    cv2.destroyAllWindows()

    # Formato do twophase (Kociemba): 54 letras na ordem U R F D L B
    ORDER = ['yellow', 'green', 'red', 'white', 'blue', 'orange']   # U R F D L B
    LETTER = {'yellow': 'U', 'green': 'R', 'red': 'F',
              'white': 'D', 'blue': 'L', 'orange': 'B'}

    cube_string = "".join(LETTER[c] for name in ORDER for c in faces[name].flatten())
    print("cube_string:", cube_string)

    # diagnóstico antes de chamar o solver
    for l in "URFDLB":
        print(l, cube_string.count(l))      # todos devem ser 9
    print("centros:", [cube_string[i] for i in (4, 13, 22, 31, 40, 49)])  # deve ser U R F D L B

    print("Solving the cube...")
    raw_solution = sv.solve(cube_string, 24, 5)
    print("Retorno do solver:", raw_solution)

    if raw_solution.startswith("Error"):
        print("O solver rejeitou o cubo. Veja o número do erro e a cube_string acima.")
    else:
        launch_cube(parse_solution(raw_solution))
