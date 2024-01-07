import logging
import socket
import time


PORT = 12345
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_server_socket(host: str) -> socket.socket:
    # Initialise socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, PORT))

    # Allow for socket to listen for connectionss
    server_socket.listen(1)
    logger.info('Listening for incoming connections on %s:%i', host, PORT)

    return server_socket


def process_client(client_socket: socket.socket) -> None:
    # Decode data
    data = client_socket.recv(1024).decode()
    logger.info('Decoded data: %s', data)

    #client_socket.close()


if __name__ == '__main__':
    # Initialise server
    server_socket = get_server_socket('10.12.226.38')

    # Accept client connection
    client_socket, client_address = server_socket.accept()
    logger.info('Connection establisted from %s', client_address)

    try:
        while True:
            process_client(client_socket)
            time.sleep(0.1)
    except KeyboardInterrupt:
        client_socket.close()
        server_socket.close()
        logger.info('Connection terminated successfully')

    logger.info('Terminating server')