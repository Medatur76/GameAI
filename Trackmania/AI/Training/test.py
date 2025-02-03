# client example

import socket, time, json
while True:
    time.sleep(5)
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('127.0.0.1', 9000))
    data = client_socket.recv(512)
    data = client_socket.recv(512).replace(data, b'')
    print("RECEIVED:", data)
    # Decode UTF-8 bytes to Unicode, and convert single quotes 
    # to double quotes to make it valid JSON
    my_json = data.decode('utf8')
    print(my_json)
    print('- ' * 20)

    # Load the JSON to a Python list & dump it back out as formatted JSON
    data = json.loads(my_json)
    print(data["speed"])
    s = json.dumps(data, indent=4, sort_keys=True)
    print(s)
    client_socket.close()