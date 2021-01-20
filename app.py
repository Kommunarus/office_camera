from flask import Flask, request, redirect, url_for, session, flash
from flask import render_template
import json
import os
from dl.detec_tracking_db import  multiDetect
from werkzeug.utils import secure_filename
from flask_caching import Cache
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import SelectField
from wtforms.validators import InputRequired

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


class ChoiceSourc(FlaskForm):
    group_web = SelectField('Web')
    group_file = SelectField('File')


db.create_all()



@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == "GET":
        return render_template(
            "home.html"
        )
    else:
        if'Cameras' in  request.form:
            return redirect(url_for("cameras"))
        if 'Files' in  request.form:
            return redirect(url_for("files"))
        if 'Detection' in  request.form:
            return redirect(url_for("start"))


@app.route('/start', methods=['GET', 'POST'])
def start():
    if request.method == "GET":
        groups_web_list = [(i.path, i.path) for i in Camera.query.all()]
        groups_web_file = [(i.path, i.path) for i in Video.query.all()]
        form = ChoiceSourc()
        form.group_web.choices = groups_web_list
        form.group_file.choices = groups_web_file
        return render_template(
            "start.html", form=form
        )
    else:
        form = ChoiceSourc()

        if'Web' in  request.form:
            sourc = form.group_web.data.strip()
            messages = json.dumps({"form_source": 'OnLine', "filename": sourc})

        if 'File' in  request.form:
            sourc = form.group_file.data.strip()
            messages = json.dumps({"form_source": 'OffLine', "filename": sourc})

        session['messages'] = messages
        return redirect(url_for("detector", sourc = sourc))


@app.route('/detector', methods=['GET', 'POST'])
def detector():
    if request.method == "GET":
        messages = session['messages']
        messages = json.loads(messages)

        filename = messages['filename']
        type = messages['form_source']
        # print(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        if filename != '' and type == 'OffLine':
            multiDetect(os.path.join(app.config['UPLOAD_FOLDER'], filename), 1, '1')
        if filename != '' and type == 'OnLine':
            multiDetect(filename, 1, '1')

        return render_template(
            "detector.html"
        )


@app.route('/cameras', methods=['GET', 'POST'])
def cameras():
    if request.method == "GET":
        return render_template(
            "cameras.html"
        )
    else:
        if'All' in  request.form:
            return render_template('cameras_all.html', list=Camera.query.all())
        if 'Add' in  request.form:
            return redirect(url_for("new_cam"))


@app.route('/new_cam', methods=['GET', 'POST'])
def new_cam():
    if request.method == 'POST':
        if not request.form['name'] or not request.form['room'] or not request.form['path']:
            flash('Please enter all the fields', 'error')
        else:
            cam = Camera(request.form['name'], request.form['room'],
                               request.form['path'])

            db.session.add(cam)
            db.session.commit()
            flash('Record was successfully added')
            return redirect(url_for('cameras_all'))
    return render_template('cameras_new.html')


@app.route('/cameras_all')
def cameras_all():
   return render_template('cameras_all.html', list = Camera.query.all() )


@app.route('/files', methods=['GET', 'POST'])
def files():
    if request.method == "GET":
        return render_template(
            "file.html"
        )
    else:
        if'All' in  request.form:
            return render_template('files_all.html', list=Video.query.all())
        if 'Add' in  request.form:
            return redirect(url_for("file_add"))


@app.route('/file_add', methods=['GET', 'POST'])
def file_add():
    if request.method == 'POST':
        if not request.form['comment'] or not request.form['path']:
            flash('Please enter all the fields', 'error')
        else:
            cam = Video(request.form['comment'],    request.form['path'])

            db.session.add(cam)
            db.session.commit()
            flash('Record was successfully added')
            return redirect(url_for('files_all'))
    return render_template('files_new.html')


@app.route('/files_all')
def files_all():
   return render_template('files_all.html', list = Video.query.all() )
