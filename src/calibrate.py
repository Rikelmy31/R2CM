#!/usr/bin/env python3

# Imports
import os
import sys
import time
import argparse
import cv2
import numpy as np
import json
from functools import partial
from numpy import mean
COLORS = {"White": (255, 255, 255), "Yellow": (0, 255, 255), "Orange": (0, 165, 255), "Red": (0, 0, 255), "Green": (0, 255, 0), "Blue": (255, 0, 0)}


# ---------------------------------------------------------------------------
# Câmera (webcam local ou stream do celular) -- mesmas funções do main.py
# ---------------------------------------------------------------------------
def aplicar_orientacao(frame, rotate=0, flip=False):
    rot_map = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    if rotate in rot_map:
        frame = cv2.rotate(frame, rot_map[rotate])
    if flip:
        frame = cv2.flip(frame, 1)
    return frame


def ajustar_frame(frame):
    """Altura 480; corta as laterais se passar de 640 (igual ao main.py)."""
    h, w = frame.shape[:2]
    if h == 480 and w == 640:
        return frame
    frame = cv2.resize(frame, (int(w * 480.0 / h), 480))
    if frame.shape[1] > 640:
        x0 = (frame.shape[1] - 640) // 2
        frame = frame[:, x0:x0 + 640]
    return frame


def abrir_camera(cam_source):
    cap = cv2.VideoCapture(cam_source)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a câmera: {cam_source}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if isinstance(cam_source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def ler_frame(cap, rotate=0, flip=False, tentativas=120):
    for _ in range(tentativas):
        ok, frame = cap.read()
        if ok and frame is not None:
            return ajustar_frame(aplicar_orientacao(frame, rotate, flip))
        time.sleep(0.02)
    raise RuntimeError("Não consegui ler a câmera. Confira a URL/índice e a rede.")


def getLimits(window_name):
    min_b = cv2.getTrackbarPos('min B', window_name)
    min_g = cv2.getTrackbarPos('min G', window_name)
    min_r = cv2.getTrackbarPos('min R', window_name)

    max_b = cv2.getTrackbarPos('max B', window_name)
    max_g = cv2.getTrackbarPos('max G', window_name)
    max_r = cv2.getTrackbarPos('max R', window_name)

    min = np.array([min_b, min_g, min_r], np.uint8)
    max = np.array([max_b, max_g, max_r], np.uint8)

    return min, max


def onTrackbar(val):    # replaced by getLimits() but
    pass                # required to create trackbars


def mouseClick(event, x, y, flags, param, window_name, img_dict):

    if event == cv2.EVENT_LBUTTONDOWN:
        
        image = img_dict['image']
        y0, x0 = max(y - 3, 0), max(x - 3, 0)
        b, g, r = [int(v) for v in np.mean(image[y0:y + 4, x0:x + 4], axis=(0, 1))]
    
        range = 30

        min_b = b - range if b > range else 0
        max_b = b + range if b < 255-range else 255
        min_g = g - range if g > range else 0
        max_g = g + range if g < 255-range else 255
        min_r = r - range if r > range else 0
        max_r = r + range if r < 255-range else 255

        limits = {'min B': min_b, 'max B': max_b, 'min G': min_g, 'max G': max_g, 'min R': min_r, 'max R': max_r}

        for limit in limits:
            cv2.setTrackbarPos(limit, window_name, limits[limit])


def run(colors_class, cam_source=0, rotate=0, flip=False):
    

    # Create windows
    name_segmented = 'Segmented'
    name_original = 'Original'
    cv2.namedWindow(name_segmented, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(name_original, cv2.WINDOW_AUTOSIZE)



    limits = {'B': {'max': 200, 'min': 100}, 'G': {'max': 200, 'min': 100}, 'R': {'max': 200, 'min': 100}}


    # Create trackbars {'name': count}
    trackbars = {'min B': limits['B']['min'],
                    'max B': limits['B']['max'],
                    'min G': limits['G']['min'],
                    'max G': limits['G']['max'],
                    'min R': limits['R']['min'],
                    'max R': limits['R']['max']}

    for tb in trackbars:
        cv2.createTrackbar(tb, name_segmented, trackbars[tb], 255, onTrackbar)


    # Select camera and get first frame
    capture = abrir_camera(cam_source)
    image = ler_frame(capture, rotate, flip)
    #image = cv2.flip(image, 1)
    img_dict = {'image': image}


    # Move windows to starting pos
    cv2.moveWindow(name_original, 100, 0)
    cv2.moveWindow(name_segmented, image.shape[1] + 120, 0)


    # Create mouse callback
    cv2.setMouseCallback(name_original, partial(mouseClick, window_name=name_segmented, img_dict=img_dict))



    dict = {}
    colors = {"W": "White", "B": "Blue", "G": "Green", "R": "Red", "Y": "Yellow", "O": "Orange"}
    color_names = ["W", "B", "G", "R", "Y", "O"]
    color_calibrated = False
    color = color_names.pop(0)
    while True:

        if color_calibrated:
            color = color_names.pop(0)
            color_calibrated = False
        
        # Update image from camera
        image = ler_frame(capture, rotate, flip)
        #image = cv2.flip(image, 1)
        img_dict['image'] = image
        cv2.putText(image, f"Let's Calibrate {colors[color]} color", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[colors[color]], 2)
        cv2.imshow(name_original, image)


        # Update segmented image
        min, max = getLimits(name_segmented)
        image_thresholded = cv2.inRange(image, min, max)
        cv2.imshow(name_segmented, image_thresholded)


        # Keyboard inputs
        key = cv2.waitKey(10) & 0xFF        # Only read last byte (prevent numlock)

        if key == ord('w') or key == 13:    # w or ENTER
            min = min.tolist()
            max = max.tolist()

            dict[color] = {'B': {'max': max[0], 'min': min[0]},
                                'G': {'max': max[1], 'min': min[1]},
                                'R': {'max': max[2], 'min': min[2]}}

            if color == "O":
                for i in dict:
                    color = colors[i]
                    values = dict[i]
                    max = (values['B']['max'], values['G']['max'], values['R']['max'])
                    min = (values['B']['min'], values['G']['min'], values['R']['min'])
                    print(f"{color} max: {max} min: {min}")
                    average = (mean([max[0], min[0]]), mean([max[1], min[1]]), mean([max[2], min[2]]))
                    colors_class.update_prominent_color(color.lower(), average)
                    #save to file
                os.makedirs('resources', exist_ok=True)
                with open('resources/colors.json', 'w') as f:
                    json.dump(colors_class.prominent_color_palette, f)
                capture.release()
                cv2.destroyAllWindows()
                return
            color_calibrated = True

        elif key == ord('q') or key == 27:  # q or ESC
            sys.exit(0)




if __name__ == '__main__':
    import helpers
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", default="0")
    ap.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0)
    ap.add_argument("--flip", action="store_true")
    a = ap.parse_args()
    colors = helpers.Colors()
    run(colors, int(a.cam) if a.cam.isdigit() else a.cam, a.rotate, a.flip)