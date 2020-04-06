from pyardrone import ARDrone
from contextlib import suppress

def main():
    drone = ARDrone()

    drone.navdata_ready.wait()
    with suppress(KeyboardInterrupt):
        while True:
            print(drone.state)
    drone.close()

if __name__ == '__main__':
    main()