from flask import Flask, request, Response
import json

from penguins import penguins_classifier

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello. Welcome to my first web app."

@app.route("/penguins", methods=["POST"])
def penguins():
    
    request_data = request.json
    
    response = penguins_classifier(request_data)
    
    return json.dumps(response)



if __name__ == "__main__":
    app.run()