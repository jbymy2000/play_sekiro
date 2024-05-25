import cv2

from pysekiro.img_tools.get_status import get_status
from pysekiro.img_tools.get_vertices import roi
from pysekiro.img_tools.grab_screen import get_screen
from pysekiro.key_tools.get_keys import key_check

x   = 250
x_w = 550
y   = 75
y_h = 375

def main():

    paused = True
    print("Ready!")

    while True:
        keys = key_check()
        if paused:
            if 'T' in keys:
                paused = False
                print('Starting!')
        else:

            screen = get_screen()

            status_info = get_status(screen)[4]
            print('\r' + status_info, end='')

            cv2.imshow('roi', roi(screen, x, x_w, y, y_h))

            # Calibration line
            screen[409:, [48, 49, 304, 305], :] = 255    # Self_HP

            # screen[389, 401:483, :] = 255    # Self_Posture
            screen[[384, 385, 392,393], 401:483, :] = 255    # Self_Posture
            screen[389:, 401, :] = 255    # Self_Posture Midline

            screen[:41, [48, 49, 215, 216], :] = 255    # Target_HP

            # screen[29, 401:544, :] = 255    # Target_Posture
            screen[[25, 26, 32, 33], 401:544, :] = 255    # Target_Posture
            screen[:29, 401, :] = 255    # Target_Posture Midline

            cv2.imshow('screen', screen)
            cv2.waitKey(1)

            if 'P' in keys:
                cv2.destroyAllWindows()
                break

    print('\nDone!')