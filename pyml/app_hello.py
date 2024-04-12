from flask import Flask, request, Response

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello. Welcome to my first web app."

if __name__ == "__main__":
    app.run()