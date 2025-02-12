import socket, json, time
from AI.Training.Training import Training

def run(training: Training):
    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Define the host and port
    host = '127.0.0.1'  # Localhost
    port = 6161         # You can choose any available port

    # Bind the socket to the host and port
    server_socket.bind((host, port))

    # Start listening for incoming connections
    server_socket.listen(1)
    print(f"Server listening on {host}:{port}")



    while True:
        # Accept a connection from a client
        client_socket, _ = server_socket.accept()

        # Receive the request data
        request = client_socket.recv(1024).decode('utf-8').split("\r\n")[0].split(" ")[1].replace(f"/{host}:{port}", "")

        # Determine the response based on the requested path
        if "/display" == request:
            html_data = open("index.html", "r").read()
            http_response = f"HTTP/1.1 200 OK\nContent-type: text/html\n\n{html_data}"
        elif "/data" == request:
            data = {
                'position': training.position,
                'speed': training.speed,
                'time_elapsed': time.time()-training.startTime
            }

            json_data = json.dumps(data)

            # Create an HTTP response
            http_response = f"HTTP/1.1 200 OK\nContent-Type: application/json\n\n{json_data}"
        else:
            # Default JSON data for the root endpoint
            if training.isGen:
                data = {
                    'status': 'idle',
                    'training': {'generation': training.currentGen, 'ai': training.currentAI}
                }
            else:
                data = {
                    'status': 'idle',
                    'training': {'run': training.currentAI}
                }
            # Convert the data to JSON
            json_data = json.dumps(data)

            # Create an HTTP response
            http_response = f"HTTP/1.1 200 OK\nContent-Type: application/json\n\n{json_data}"

        # Send the response to the client
        client_socket.sendall(http_response.encode('utf-8'))

        # Close the client socket
        client_socket.close()