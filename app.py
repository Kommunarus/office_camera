from flask import Flask, request, redirect, url_for, session, flash
from flask import render_template
import json
import os
import psutil
import pandas as pd
from dl.detec_tracking_db import  multiDetect
from werkzeug.utils import secure_filename
from flask_caching import Cache
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import SelectMultipleField, SubmitField, FieldList, FormField, Label
import uuid

UPLOAD_FOLDER = './static/uploads'

app = Flask(__name__)
app.secret_key = b'kds$%fdfge^%$G'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DEBUG'] = True
app.config["CACHE_TYPE"] = "null"
app.config ['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cameras.sqlite3'
db = SQLAlchemy(app)

db.create_all()
def deleteProc():
    rows = Process.query.all()
    for row in rows:
        try:
            proc = psutil.Process(int(row.pid))
            proc.terminate()
            print('del process {}'.format(row.pid))
        except:
            print('Process with pid {} dont terminate'.format(row.pid))

        Process.query.filter_by(pid=row.pid).delete()
        db.session.commit()

    open('n.txt', 'w').close()




cache = Cache(config={'CACHE_TYPE': 'redis'})
cache.init_app(app)

ListProc = {}

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


class Process(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    pid = db.Column(db.String(100))
    type = db.Column(db.String(10))
    path = db.Column(db.String(250))

    def __repr__(self):
        return '<Process {}>'.format(self.pid)

    def __init__(self, pid, type, path):
        self.pid = pid
        self.type = type
        self.path = path

deleteProc()


class ChoiceSourc(FlaskForm):
    group_web = SelectMultipleField('Web')
    group_file = SelectMultipleField('File')

class SubmitFieldForm(FlaskForm):
    Submit = SubmitField('SSS')

class SubmitFieldFormList(FlaskForm):
    Submit_entries = FieldList(FormField(SubmitFieldForm))


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

        if 'Web' in  request.form:
            sourc = form.group_web.data
            messages = json.dumps({"form_source": 'OnLine'})

        if 'File' in  request.form:
            sourc = form.group_file.data
            messages = json.dumps({"form_source": 'OffLine'})

        session['messages'] = messages

        for i, path in enumerate(sourc):
            newproc = Process(pid='', type='s', path=path)
            db.session.add(newproc)
            db.session.commit()

        return redirect(url_for("detector", sourc = sourc))


@app.route('/detector', methods=['GET', 'POST'])
def detector():


    if request.method == "GET":

        open('n.txt', 'w').close()

        allfile = Process.query.all()
        messages = session['messages']
        messages = json.loads(messages)

        type = messages['form_source']
        with open('n.txt', 'w') as f:

            for i, row in enumerate(allfile):
                if row.pid.strip() == '':
                    proc = None
                    if type == 'OffLine':
                        proc = multiDetect(os.path.join(app.config['UPLOAD_FOLDER'], row.path.srip()), 1, '1', False, True)
                    else:
                        proc = multiDetect(row.path.strip(), 1, '1', False, True)


                    if proc != None:
                        ListProc[proc.pid] = proc
                        a_proc = Process.query.filter(Process.path == row.path).one()
                        a_proc.pid = proc.pid
                        f.write(str(proc.pid)+ "\n")

                        db.session.commit()
                else:
                    f.write(str(row.pid) + "\n")

        # form = MultiForm()
        select_metadata_form_list = SubmitFieldFormList()
        select_metadata_form_list.Submit_entries = get_select_entries()

        all_images = ''
        with open('n.txt', 'r') as f:
            reader = f.readlines()
            for line in reader:
                all_images += ''


        context = {
            "Submit_form_list": select_metadata_form_list,
        }

        return render_template(
            "detector.html", **context
        )
    else:
        d = request.form.to_dict()
        for k, v in d.items():

            if k.find('Submit') > -1:
                pid = int(k.replace('Submit',''))
                print(pid)
                proc = psutil.Process(pid)
                proc.terminate()

                Process.query.filter_by(pid=pid).delete()
                db.session.commit()
                print('kill process {}'.format(pid))

        all_proc = Process.query.all()
        if len(all_proc)==0:
            return redirect(url_for("start"))
        else:
            return redirect(url_for("detector"))



def get_select_entries():
    all_select_items = []
    if os.path.isfile('n.txt'):
        with open('n.txt', 'r') as f:
            reader = f.readlines()
            for line in reader:
                print('build form {}'.format(line.strip()))
                select_id = uuid.uuid1()   # allows for multiple selects
                sub_entry = SubmitFieldForm()
                sub_entry.Submit.id = 'Submit{}'.format(line.strip())
                sub_entry.Submit.name = 'Submit{}'.format(line.strip())
                sub_entry.Submit.description = 'img/{}.jpg?'.format(line.strip())
                sub_entry.Submit.label = Label(uuid.uuid1(),line.strip())
                all_select_items.append(sub_entry)

    return all_select_items

@app.route('/cameras', methods=['GET', 'POST'])
def cameras():
    if request.method == "GET":
        return render_template(
            "cameras.html"
        )
    else:
        if 'All' in  request.form:
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


if __name__ == '__main__':
    app.run()
