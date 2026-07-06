"""Web server: management console, MJPEG stream, and control endpoints.

Run:  python server.py     then open  http://<device-ip>:8000  on the LAN.
"""
import hmac
import json
import os
import threading
import time

from flask import Flask, Response, jsonify, request, send_file

import config
import face_lib as fl
import recognition

state = recognition.State()
app = Flask(__name__)


def authorized():
    given = request.headers.get("Authorization", "")
    return hmac.compare_digest(given, "Bearer " + config.TOKEN)


@app.route("/")
def index():
    return send_file("console.html")


@app.route("/stream")
def stream():
    def gen():
        while True:
            if state.frame_jpeg:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + state.frame_jpeg + b"\r\n")
            time.sleep(0.05)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/log")
def get_log():
    if not os.path.exists(config.LOG_PATH):
        return jsonify([])
    with open(config.LOG_PATH) as f:
        return jsonify([json.loads(line) for line in f if line.strip()])


@app.route("/users")
def list_users():
    if not authorized():
        return jsonify(error="bad token"), 401
    return jsonify(sorted(state.db))


@app.route("/users/<name>", methods=["DELETE"])
def delete_user(name):
    if not authorized():
        return jsonify(error="bad token"), 401
    if name not in state.db:
        return jsonify(error="no such user"), 404
    del state.db[name]
    fl.save_db(state.db)
    return jsonify(ok=True)


@app.route("/enroll", methods=["POST"])
def enroll():
    if not authorized():
        return jsonify(error="bad token"), 401
    name = (request.get_json(silent=True) or {}).get("name", "").strip()
    if not name:
        return jsonify(error="name required"), 400
    if state.mode == "enroll":
        return jsonify(error="already enrolling"), 409
    state.enroll_name = name
    state.mode = "enroll"
    return jsonify(ok=True)


if __name__ == "__main__":
    threading.Thread(target=recognition.run, args=(state,), daemon=True).start()
    app.run(host=config.HOST, port=config.PORT, threaded=True)
