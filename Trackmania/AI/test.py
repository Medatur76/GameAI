import socket, json, time
from test2 import test

def run(t: test):
    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Define the host and port
    host = '127.0.0.1'  # Localhost
    port = 5000         # You can choose any available port

    # Bind the socket to the host and port
    server_socket.bind((host, port))

    # Start listening for incoming connections
    server_socket.listen(1)
    print(f"Server listening on {host}:{port}")



    while True:
        # Accept a connection from a client
        client_socket, addr = server_socket.accept()

        # Receive the request data
        request = client_socket.recv(1024).decode('utf-8').split("\r\n")[0].split(" ")[1].replace(f"/{host}:{port}", "")

        # Determine the response based on the requested path
        if "/display" == request:
            html_data = open("index.html", "r").read()
            http_response = f"HTTP/1.1 200 OK\nContent-type: text/html\n\n{html_data}"
        elif "/data" == request:
            data = {
                'position': [51, 271, 8],
                'speed': 123,
                'time_elapesd': time.time(),
                'num': t.num
            }

            json_data = json.dumps(data)

            # Create an HTTP response
            http_response = f"HTTP/1.1 200 OK\nContent-Type: application/json\n\n{json_data}"
        else:
            # Default JSON data for the root endpoint
            data = {
                'status': 'idle',
                'training': {'generation': 1, 'ai': 50}
            }
            # Convert the data to JSON
            json_data = json.dumps(data)

            # Create an HTTP response
            http_response = f"HTTP/1.1 200 OK\nContent-Type: application/json\n\n{json_data}"

        # Send the response to the client
        client_socket.sendall(http_response.encode('utf-8'))

        # Close the client socket
        client_socket.close()