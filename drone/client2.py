
import cv2
from pyardrone.video import VideoClient

def main():
    client = VideoClient('192.168.1.1', 5555)
    client.connect()
    client.video_ready.wait()
    try:
        while True:
            cv2.imshow('im', client.frame)
            if cv2.waitKey(10) == ord(' '):
                break
    finally:
        client.close()


if __name__ == '__main__':
    main()