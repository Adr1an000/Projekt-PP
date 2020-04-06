import cv2
from pyardrone import ARDrone
import logging
import time


def main():
    logging.basicConfig(level=logging.DEBUG)

    client = ARDrone()
    print("0")
    client.video_ready.wait()
    print("1")
    try:
        while True:
            cv2.imshow('im', client.video_client.frame)
            if cv2.waitKey(10) == ord(' '):
                break
    finally:
        client.close()

if __name__ == '__main__':
    main()