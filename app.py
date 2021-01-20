from flask import Flask, request, redirect, url_for, session, flash
from flask import render_template
import json
import os
from dl.detec_tracking_db import  multiDetect
from werkzeug.utils import secure_filename
from flask_caching import Cache
from flask_sqlalchemy import SQLAlchemy

UPLOAD_FOLDER = './static/uploads'

app = Flask(__name__)
app.secret_key = b'kds$%fdfge^%$G'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DEBUG'] = True
app.config["CACHE_TYPE"] = "null"
app.config ['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cameras.sqlite3'
db = SQLAlchemy(app)

cache = Cache(config={'CACHE_TYPE': 'redis'})
cache.init_app(app)

class Camera(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80))
    room = db.Column(db.String(80))
    path = db.Column(db.String(250))

    def __repr__(self):
        return '<Camera {}>'.format(self.name)

    def __init__(self, name, room, path):
        self.name = name
        self.room = room
        self.path = path


class Video(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    comment = db.Column(db.String(100))
    path = db.Column(db.String(250))

    def __repr__(self):
        return '<Video {}>'.format(self.comment)

    def __init__(self, comment, path):
        self.comment = comment
        self.path = path

db.create_all()

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == "GET":
        return render_template(
            "home.html"
        )
    else:
        form_source = request.form["source"]
        video_flow = request.form["video"]
        print(video_flow)
        filename = ''
        # form_file = request.form["file"]
        if 'file' not in request.files and  video_flow== '':
            print('No file part')
            return redirect(request.url)
        if video_flow == 'OffLine':
            f = request.files["file"]
            filename = f.filename
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        messages = json.dumps({"form_source": form_source, 'filename':filename, 'video_flow':video_flow})
        session['messages'] = messages
        print('2')
        return redirect(url_for("start", messages=messages))

@app.route('/start', methods=['GET', 'POST'])
def start():
    if request.method == "GET":
        messages = request.args['messages']  # counterpart for url_for()
        # messages = session['messages']
        messages = json.loads(messages)
        return render_template(
            "start.html", source=messages['form_source'], file=messages['filename'], flow=messages['video_flow']
        )
    else:
        return redirect(url_for("detector"))

@app.route('/detector', methods=['GET', 'POST'])
def detector():
    if request.method == "GET":
        messages = session['messages']
        messages = json.loads(messages)

        filename = messages['filename']
        flow = messages['video_flow']
        type = messages['form_source']
        # print(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        if filename != '' and type == 'OffLine':
            multiDetect(os.path.join(app.config['UPLOAD_FOLDER'], filename), 1, '1')
        if flow != '' and type == 'OnLine':
            multiDetect(flow, 1, '1')

        return render_template(
            "detector.html"
        )
