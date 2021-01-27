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
from wtforms import SelectMultipleField, SubmitField, FieldList, FormField, Label, SelectField
import uuid
import subprocess

UPLOAD_FOLDER = './static/uploads'
N_gpu = 2

app = Flask(__name__)
app.secret_key = b'kds$%fdfge^%$G'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DEBUG'] = True
app.config["CACHE_TYPE"] = "null"
app.config ['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cameras2.sqlite3'
db = SQLAlchemy(app)


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

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])
    # Convert lines into a dictionary
    # print(result)
    gpu_memory = [int(x) for x in result.decode().strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def get_select_entries():
    all_select_items = []
    if os.path.isfile('n.txt'):
        with open('n.txt', 'r') as f:
            reader = f.readlines()
            for line in reader:
                print('build form {}'.format(line.strip()))
                sub_entry = SubmitFieldForm()
                sub_entry.Submit.id = 'Submit{}'.format(line.strip())
                sub_entry.Submit.name = 'Submit{}'.format(line.strip())
                sub_entry.Submit.description = 'img/{}.jpg?'.format(line.strip())
                sub_entry.Submit.label = Label(uuid.uuid1(),line.strip())
                all_select_items.append(sub_entry)

    return all_select_items

cache = Cache(config={'CACHE_TYPE': 'redis'})
cache.init_app(app)

ListProc = {}
usegpu = []

class Placement(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    camera = db.relationship('Camera', backref='placement',)
    def __init__(self, name):
        self.name = name


class Camera(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80))
    path = db.Column(db.String(250))
    homography = db.relationship('Homography', backref='camera',)
    placement_id = db.Column(db.Integer, db.ForeignKey('placement.id'))
    def __init__(self, name, path, placement_id):
        self.name = name
        self.path = path
        self.placement_id = placement_id


class Homography(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    camera_id = db.Column(db.Integer, db.ForeignKey('camera.id'))
    xloc = db.Column(db.FLOAT(2))
    yloc = db.Column(db.FLOAT(2))
    xglob = db.Column(db.FLOAT(2))
    yglob = db.Column(db.FLOAT(2))
    def __init__(self, camera_id, xloc, yloc, xglob, yglob):
        self.camera_id = camera_id
        self.xloc = xloc
        self.yloc = yloc
        self.xglob = xglob
        self.yglob = yglob


class Video(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    comment = db.Column(db.String(100))
    path = db.Column(db.String(250))
    def __init__(self, id, comment, path):
        self.id = id
        self.comment = comment
        self.path = path


class Process(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    pid = db.Column(db.String(100))
    type = db.Column(db.String(10))
    path = db.Column(db.String(250))
    gpu = db.Column(db.Integer)
    cam = db.Column(db.String(100))
    cam_id = db.Column(db.Integer)
    def __init__(self, pid, type, path, gpu, cam, cam_id):
        self.pid = pid
        self.type = type
        self.path = path
        self.gpu = gpu
        self.cam = cam
        self.cam_id = cam_id


db.create_all()
db.session.commit()
deleteProc()



class ChoiceSourc(FlaskForm):
    group_web = SelectMultipleField('Web')
    sub_web_show = SubmitField('Просмотр потока')
    sub_web_write = SubmitField('Запись')
    sub_web_show_box = SubmitField('Просмотр с боксами')
    group_file = SelectMultipleField('File')

class PlaceSourc(FlaskForm):
    group_web = SelectField('Комната')

class CameraSourc(FlaskForm):
    group_web = SelectField('Камера')

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
        if'Placement' in  request.form:
            return redirect(url_for("placement"))
        if 'Files' in  request.form:
            return redirect(url_for("files"))
        if 'Homo' in  request.form:
            return redirect(url_for("homography_all"))
        if 'Detection' in  request.form:
            return redirect(url_for("start"))


@app.route('/start', methods=['GET', 'POST'])
def start():
    if request.method == "GET":
        groups_web_list = [(i.path, i.name) for i in Camera.query.all()]
        groups_web_file = [(i.path, i.name) for i in Video.query.all()]
        form = ChoiceSourc()
        form.group_web.choices = groups_web_list
        form.group_file.choices = groups_web_file
        return render_template(
            "start.html", form=form
        )
    else:
        form = ChoiceSourc()
        # print( request.form)
        if          'sub_web_show' in  request.form \
                or 'sub_web_write' in  request.form \
                or 'sub_web_show_box' in request.form \
        :
            sourc = form.group_web.data
            messages = json.dumps({"form_source": 'OnLine'})

        if 'File' in  request.form:
            sourc = form.group_file.data
            messages = json.dumps({"form_source": 'OffLine'})

        session['messages'] = messages

        for i, path in enumerate(sourc):
            if 'sub_web_show_box' in  request.form:
                type = 'b'
            elif 'sub_web_write' in  request.form:
                type = 'w'
            else:
                type = 's'

            camera = Camera.query.filter(Camera.path == path).one()

            newproc = Process(pid='', type=type, path=path, gpu='cpu', cam=camera.name, cam_id=camera.id)
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
                    gpu = '0'
                    mem = get_gpu_memory_map()
                    for i in range(N_gpu):
                        print('gpu {} - {}'.format(i, mem[i]))
                        if mem[i] < 4000 and not i in usegpu:
                            gpu = str(i)
                            usegpu.append(i)

                    if type == 'OffLine':
                        proc = multiDetect(os.path.join(app.config['UPLOAD_FOLDER'], row.path.srip()), 'cpu', 'file',
                                           '1', False, False, True)
                    else:
                        if row.type == 's':
                            proc = multiDetect(row.path.strip(), 'cpu', row.cam, row.cam_id, False, False, True)
                        if row.type == 'b':
                            proc = multiDetect(row.path.strip(), gpu, row.cam, row.cam_id, False, True, False)
                        if row.type == 'w':
                            proc = multiDetect(row.path.strip(), gpu, row.cam, row.cam_id, True, False, True)


                    if proc != None:
                        ListProc[proc.pid] = proc
                        a_proc = Process.query.filter(Process.path == row.path).one()
                        a_proc.pid = proc.pid
                        a_proc.gpu = gpu
                        f.write(str(proc.pid)+ "\n")
                        print('{} {} '.format(gpu, row.pid))

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

                usegpu.remove(Process.query.filter_by(pid=pid).one().gpu)
                Process.query.filter_by(pid=pid).delete()
                db.session.commit()
                print('kill process {}'.format(pid))

        all_proc = Process.query.all()
        if len(all_proc)==0:
            return redirect(url_for("start"))
        else:
            return redirect(url_for("detector"))





@app.route('/placement', methods=['GET', 'POST'])
def placement():
    if request.method == "GET":
        return render_template(
            "placement.html"
        )
    else:
        if 'All' in  request.form:
            return redirect(url_for("placement_all"))
        if 'Add' in  request.form:
            return redirect(url_for("new_placement"))

@app.route('/new_placement', methods=['GET', 'POST'])
def new_placement():
    if request.method == 'POST':
        if not request.form['name']:
            flash('Please enter all the fields', 'error')
        else:
            place = Placement(request.form['name'])

            db.session.add(place)
            db.session.commit()
            flash('Record was successfully added')
            return redirect(url_for('placement_all'))
    return render_template('placement_new.html')

@app.route('/placement_all')
def placement_all():
   return render_template('placement_all.html', list = Placement.query.all() )





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
        if not request.form['name']  or not request.form['path']:
            flash('Please enter all the fields', 'error')
        else:
            form = PlaceSourc()
            cam = Camera(request.form['name'],
                         request.form['path'],
                         form.group_web.data)

            db.session.add(cam)
            db.session.commit()
            flash('Record was successfully added')
            return redirect(url_for('cameras_all'))
    group_web = [(i.id, i.name) for i in Placement.query.all()]
    form = PlaceSourc()
    form.group_web.choices = group_web
    return render_template(
        "cameras_new.html", form=form
    )

@app.route('/cameras_all')
def cameras_all():
   return render_template('cameras_all.html', list = Camera.query.all() )




@app.route('/new_homography', methods=['GET', 'POST'])
def new_homography():
    if request.method == 'POST':
        if not request.form['xloc']  or not request.form['yloc'] or\
            not request.form['xglob']  or not request.form['yglob']:
            flash('Please enter all the fields', 'error')
        else:
            form = PlaceSourc()
            cam = Homography(form.group_web.data,
                             request.form['xloc'],
                             request.form['yloc'],
                             request.form['xglob'],
                             request.form['yglob'],
                             )

            db.session.add(cam)
            db.session.commit()
            flash('Record was successfully added')
            return redirect(url_for('homography_all'))
    group_web = [(i.id, i.name) for i in Camera.query.all()]
    form = CameraSourc()
    form.group_web.choices = group_web
    return render_template(
        "homography_new.html", form=form
    )

@app.route('/homography_all')
def homography_all():
   return render_template('homography_all.html', list = Homography.query.all() )





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
