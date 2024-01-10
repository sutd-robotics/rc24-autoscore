import logging
import socket
from typing import List

import pygame


# Define logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Define pygame RGB colours
BLACK = ( 0, 0, 0)
GREEN = ( 0, 255, 0)
WHITE = ( 255, 255, 255)
RED = ( 255, 0, 0)
BLUE = ( 0, 0, 255)
ORANGE = ( 255, 115, 0)
YELLOW = ( 242, 255, 0)
BROWN = ( 115, 87, 39)
PURPLE = ( 298, 0, 247)
GRAY = ( 168, 168, 168)
PINK = ( 255, 0, 234)


# Define sizings
IMAGE_SIZE = 128
IMAGE_SEP = 140
IMAGE_START = 80
IMAGE_Y = 200


# Define socket host and ports
SERVER_IP = '192.168.1.2'
SERVER_PORT = 9998
LISTEN_IP = '0.0.0.0'
LISTEN_PORT = 9999


BLUE_ALLIANCE = True


class Button:
    def __init__(
            self,
            name: str,
            server_socket: socket.socket,
            enable_icon: str, disable_icon: str,
            x: int, y: int,
            dx: int, dy: int,
            message: str
    ):
        self.__name = name
        self.__server_socket, self.__message = server_socket, message
        self.__enable_icon, self.__disable_icon = enable_icon, disable_icon
        self.__x, self.__y = x, y
        self.__dx, self.__dy = dx, dy
        self.enable()

    def __load_icon(self, img: str) -> pygame.Surface:
        icon = pygame.image.load(img).convert()
        return pygame.transform.scale(icon, (self.__dx, self.__dy))

    @property
    def name(self) -> str:
        return self.__name

    @property
    def x(self) -> int:
        return self.__x

    @property
    def y(self) -> int:
        return self.__y

    @property
    def dx(self) -> int:
        return self.__dx

    @property
    def dy(self) -> int:
        return self.__dy

    @property
    def icon(self) -> pygame.Surface:
        return self.__icon

    def enable(self) -> None:
        self.__state = True
        self.__icon = self.__load_icon(self.__enable_icon)

    def disable(self) -> None:
        self.__state = False
        self.__icon = self.__load_icon(self.__disable_icon)

    def press(self) -> None:
        self.__server_socket.send(self.__message.encode())
        # self.disable()

    def is_active(self) -> bool:
        return self.__state


if __name__ == '__main__':
    pygame.init()

    # Configure display
    display = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    pygame.display.set_caption('Show Text')
    font = pygame.font.Font('freesansbold.ttf', 32)

    # Initialise sockets
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    logger.info('Sending out connections on %s:%i', SERVER_IP, SERVER_PORT)

    listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen_socket.bind((LISTEN_IP, LISTEN_PORT))
    listen_socket.setblocking(False)
    logger.info('Listening for incoming connections on %s:%i', LISTEN_IP, LISTEN_PORT)

    # Initialise buttons
    buttons = []
    for i, (name, enable_icon, disable_icon, message) in enumerate(zip(
            ('FREEZE', 'RANDOM FREEZE', 'ERUPTION', 'TIME EXTENSION', 'DEACTIVATE BEACON'),
            ('freeze.png', 'random_freeze.png', 'eruption.png', 'time_extension.png', 'deactivate_beacon.png'),
            ('freeze deactivate.png', 'random_freeze deactivate.png', 'eruption deactivate.png', 'time_extension deactivate.png', 'deactivate_beacon deactivate.png'),
            (
                f'pw_{"b" if BLUE_ALLIANCE else "r"}_freeze',
                f'pw_{"b" if BLUE_ALLIANCE else "r"}_rfreeze',
                f'pw_{"b" if BLUE_ALLIANCE else "r"}_erupt',
                f'pw_{"b" if BLUE_ALLIANCE else "r"}_extend',
                f'pw_{"b" if BLUE_ALLIANCE else "r"}_beacon'
            )
    )):
        button = Button(
            name,
            server_socket,
            enable_icon, disable_icon,
            IMAGE_START + i * IMAGE_SEP, IMAGE_Y,
            IMAGE_SIZE, IMAGE_SIZE,
            message
        )
        button.enable()
        buttons.append(button)

    while True:
        # Check for data to read
        try:
            data = listen_socket.recv(1024).decode().strip()
            if data == 'stop':
                # for button in buttons:
                #     button.disable()
                pass
            elif data == 'start':
                for button in buttons:
                    button.enable()
        except socket.error as error:
            # logger.warning('Socket threw error: %s', error)
            pass

        # Check for user input
        for event in pygame.event.get():
            if event.type != pygame.MOUSEBUTTONDOWN:
                continue
            x, y = event.pos
            for button in buttons:
                if button.x <= x <= button.x + button.dx and \
                        button.y <= y <= button.y + button.dy and \
                        button.is_active():
                    logger.info('%s pressed!', button.name)
                    button.press()

        # Configure display
        display.fill(BLUE if BLUE_ALLIANCE else RED)
        for button in buttons:
            display.blit(button.icon, (button.x, button.y))
        pygame.display.update()
