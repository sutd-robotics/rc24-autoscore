import RPi.GPIO as GPIO
import socket
import _thread

GPIO.setmode(GPIO.BOARD)

black_button = 3
white_button = 5
green_button = 7
red_button = 11

GPIO.setup(black_button, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(white_button, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(green_button, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(red_button, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# set host and port (port stays at 12345, can be any number above 1024)
# 0.0.0.0 listens on all available interfaces
src_host = '192.168.1.120'
dst_host = '192.168.1.2'
src_port = 12345
dst_port = 9998

# creating socket object
SERVER_SOCKET = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
SERVER_SOCKET.bind((src_host, src_port))
SERVER_SOCKET.listen(3)
print(f"Listening for incoming connections on {src_host}:{src_port}")

CLIENT_SOCKET = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#CLIENT_SOCKET.connect((dst_host, dst_port))
print(f"Sending out connections on {dst_host}:{dst_port}")

lrc = lbc = lwc = rrc = rbc = rwc = 0  # CV result
lr = lb = lw = rr = rb = rw = 0        # offset
LEFT, RIGHT, RED, BLUE, WHITE = range(5)
toggle_side = LEFT
toggle_colour = RED

def black_callback(_):
    global toggle_side
    toggle_side = (RIGHT, LEFT)[toggle_side]
    print(f'Toggled side to: {("LEFT", "RIGHT")[toggle_side]}')

def white_callback(_):
    global toggle_colour
    toggle_colour = (BLUE, WHITE, RED)[toggle_colour - RED]
    print(f'Toggled colour to: {("RED", "BLUE", "WHITE")[toggle_colour - RED]}')

def green_callback(_):
    if toggle_side == LEFT:
        update_left(val=1)
    else:
        update_right(val=1)

def red_callback(_):
    if toggle_side == LEFT:
        update_left(val=-1)
    else:
        update_right(val=-1)

def update_left(nlrc=lrc, nlbc=lbc, nlwc=lwc, val=0):
    global lrc, lbc, lwc, lr, lb, lw
    if val != 0:
        if toggle_colour == RED: lr += val
        elif toggle_colour == BLUE: lb += val
        else: lw += val
        print(f'Offsets: lr={lr}, lb={lb}, lw={lw}')
    
    if nlrc == lrc and nlbc == lbc and nlwc == lwc and val == 0:
        return
    lrc = nlrc
    lbc = nlbc
    lwc = nlwc
    
    res = f'LR={lrc+lr};LB={lbc+lb};LW={lwc+lw}'
    print(f'Sending code {res}')
    CLIENT_SOCKET.sendto(
        bytes(res, 'utf-8'),
        (dst_host, dst_port)
    )

def update_right(nrrc=rrc, nrbc=rbc, nrwc=rwc, val=0):
    global rrc, rbc, rwc, rr, rb, rw
    if val != 0:
        if toggle_colour == RED: rr += val
        elif toggle_colour == BLUE: rb += val
        else: rw += val
        print(f'Offsets: rr={rr}, rb={rb}, rw={rw}')
    
    if nrrc == rrc and nrbc == rbc and nrwc == rwc and val == 0:
        return
    rrc = nrrc
    rbc = nrbc
    rwc = nrwc
    
    res = f'RR={rrc+rr};RB={rbc+rb};RW={rwc+rw}'
    print(f'Sending code {res}')
    CLIENT_SOCKET.sendto(
        bytes(res, 'utf-8'),
        (dst_host, dst_port)
    )
    
# bouncetime added to prevent double counting from button, but now we can't spam press it
GPIO.add_event_detect(black_button, GPIO.FALLING, callback=black_callback, bouncetime = 300)
GPIO.add_event_detect(white_button, GPIO.FALLING, callback=white_callback, bouncetime = 300)
GPIO.add_event_detect(green_button, GPIO.FALLING, callback=green_callback, bouncetime = 300)
GPIO.add_event_detect(red_button, GPIO.FALLING, callback=red_callback, bouncetime = 300)

def on_new_client(client_socket):
    while True:
        data = client_socket.recv(1024).decode('utf-8').strip()
        print(f'data: {data}')
        
        if not data:
            print('ERROR: CLIENT DOES NOT HAVE VALID DATA')
            continue
        data = data.split(';')
        if len(data) != 3 or any(len(d) <= 3 for d in data):
            print('ERROR: CLIENT DOES NOT HAVE VALID DATA')
            continue
        try:
            if data[0][0] == 'L':
                update_left(nlrc=int(data[0].split('=')[-1]),
                            nlbc=int(data[1].split('=')[-1]),
                            nlwc=int(data[2].split('=')[-1]))
            elif data[0][0] == 'R':
                update_right(nrrc=int(data[0].split('=')[-1]),
                             nrbc=int(data[1].split('=')[-1]),
                             nrwc=int(data[2].split('=')[-1]))
            else:
                raise ValueError
        except ValueError:
            print('ERROR: CLIENT DOES NOT HAVE VALID DATA')
            continue
    client_socket.close()

try:
    while True:
        # accepting connection
        client_socket, client_address = SERVER_SOCKET.accept()
        print(f"Connection established from {client_address}")
        _thread.start_new_thread(on_new_client, (client_socket,))
except KeyboardInterrupt:
    SERVER_SOCKET.close()
    print("Connection terminated successfully")

print('Terminating server')
GPIO.cleanup()