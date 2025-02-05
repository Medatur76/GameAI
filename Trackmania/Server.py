from flask import Flask
from AI.main import training

app = Flask(__name__)
@app.route("/")
def main():
    return "Currently on Generation " + str(gen[0]) + ". In this generation we are on AI " + str(gen[1])
def getNums():
    return training.getAIG()

app.run(debug=True, host="127.0.0.1", port=6161)