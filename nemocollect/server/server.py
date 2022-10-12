from flask import Flask
from flask import render_template, redirect, send_from_directory, request
import vi_pro

app = Flask(__name__, static_url_path='/static/')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/start', methods=["POST"])
def start():
    try:
        print("START RECORDING")
        vi_pro.start_recording()
    except Exception as e:
        print(e)
        return "invalid ids list: {}".format(str(e))
    return 'OK'

@app.route('/stop', methods=["POST"])
def stop():
    try:
        print("STOP RECORDING")
        vi_pro.stop_recording()
    except Exception as e:
        print(e)
        return "invalid ids list: {}".format(str(e))
    return 'OK'

@app.route('/next', methods=["POST"])
def next():
    try:
        print("NEXT")
        vi_pro.start_next()
    except Exception as e:
        print(e)
        return "invalid ids list: {}".format(str(e))
    return 'OK'


@app.route('/static/<path:path>')
def send_img(path):
    return send_from_directory('static', path)
