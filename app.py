from flask import Flask, request, jsonify
import emotional_welbeing as ew
import cognitive_welbeing as cw

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/api/emotional/predict", methods=['POST'])
def predict_emotional():
    data = request.get_json()
    question = data['question']
    answer = ew.get_answer(question)
    return jsonify({'answer': answer}), 200

@app.route("/api/cognitive/predict", methods=['POST'])
def predict_cognitive():
    data = request.get_json()
    question = data['question']
    answer = cw.get_answer(question)
    return jsonify({'answer': answer}), 200