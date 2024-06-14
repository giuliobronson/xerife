from flask import Flask, Response, render_template, jsonify
from controllers import classification_controller, streaming_controller
from data.db.students import students
from config import src_url
import cv2

app = Flask(__name__)
cap = cv2.VideoCapture(src_url)

@app.route("/")
def root():
    classification_controller.train()
    return render_template("index.html", students=students)

@app.route("/stream")
def stream():
    return Response(streaming_controller.stream(cap=cap), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/train")
def train():
    accuracy = classification_controller.train()
    return jsonify(accuracy)

@app.route('/table_data')
def get_table_data():
    return jsonify(students)