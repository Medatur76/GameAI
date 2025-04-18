from AI.GameInfoProcessing.FrameProcessor import takePhoto
import socket, json

def getInputs():
    output: list[float] = takePhoto("Trackmania")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('127.0.0.1', 9000))
    data = client_socket.recv(512)
    data = client_socket.recv(512).replace(data, b'').decode('utf8')
    try:
        jsonData = json.loads(data)
    except:
        return [0 for _ in range(16)], [[0.0, 0.0, 0.0], False, False]
    output.append(float(jsonData["speed"]))
    gameData: list = []
    gameData.append(jsonData["position"])
    if (jsonData["end"] == "true"): gameData.append(True)
    else: gameData.append(False)
    if (jsonData["running"] == "true"): gameData.append(True)
    else: gameData.append(False)
    return output, gameData