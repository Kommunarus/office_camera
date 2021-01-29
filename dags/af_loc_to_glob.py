import cv2
import sqlite3
import numpy as np
from clickhouse_driver import Client
import uuid
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

args = {
    'owner': 'airflow',
}



dag = DAG(
    dag_id='VA_local_coordinates_to_global_coordinates',
    default_args=args,
    schedule_interval=None,
    start_date=days_ago(2),
    tags=['coordinates'],
)


def get_homography(id_camera):
    con = sqlite3.connect('/home/alex/PycharmProjects/office_camera/cameras2.sqlite3')
    cursorObj = con.cursor()
    cursorObj.execute("SELECT xloc, yloc, xglob, yglob FROM homography WHERE camera_id = '{}'".format(id_camera))
    table = cursorObj.fetchall()
    loc = []
    glob = []
    for row in table:
       loc.append((row[0], row[1]))
       glob.append((row[2], row[3]))

    pts_src = np.array(glob)
    # Homography processing to make flat image
    pts_dst = np.array(loc)
    h, status = cv2.findHomography(pts_dst, pts_src, cv2.RANSAC, 3)
    con.close()
    return h, status

def glob_coord(x, y, h):
    test_point = (x, y, 1)
    new_coord = h.dot(test_point)
    xx = new_coord[0] / new_coord[2]
    yy = new_coord[1] / new_coord[2]

    return xx, yy

def doGlob():

    client = Client(host='localhost')
    with open('camera.txt', 'r') as f:
        reader = f.readlines()
        for id in reader:
            id_camera = id.strip()
            h, status = get_homography(id_camera)

            tab = client.execute(f"SELECT \
                                    cam_coordinates.x1, \
                                    cam_coordinates.y1, \
                                    cam_coordinates.x2, \
                                    cam_coordinates.y2, \
                                    cam_coordinates.cam_coordinates_id\
                                FROM dbDetector.cam_coordinates  \
                                LEFT JOIN dbDetector.glob_coordinates  \
                                ON cam_coordinates.cam_coordinates_id=glob_coordinates.cam_coordinates_id \
                                    WHERE glob_coordinates.glob_coordinates_id ='00000000-0000-0000-0000-000000000000' \
                                    and cam_coordinates.cam_coordinates.camera = '{id_camera}' \
                                    and cam_coordinates.cam_coordinates.type_bbox = '1.0'\
                                     "
                                 )

            for row in tab:
                x1 = row[0]
                y1 = row[1]
                x1glob, y1glob = glob_coord(x1, y1, h)
                x2 = row[2]
                y2 = row[3]
                x2glob, y2glob = glob_coord(x2, y2, h)
                x1glob = 0 if np.isnan(x1glob) else x1glob
                y1glob = 0 if np.isnan(y1glob) else y1glob
                x2glob = 0 if np.isnan(x2glob) else x2glob
                y2glob = 0 if np.isnan(y2glob) else y2glob

                client.execute(
                    "INSERT INTO dbDetector.glob_coordinates (x1, y1, x2, y2, cam_coordinates_id, glob_coordinates_id) VALUES",
                    [{'x1':x1glob, 'y1':y1glob,  'x2':x2glob, 'y2':y2glob, 'cam_coordinates_id':row[4], 'glob_coordinates_id':uuid.uuid1()}]
                )


def start():
    con = sqlite3.connect('/home/alex/PycharmProjects/office_camera/cameras2.sqlite3')
    cursorObj = con.cursor()

    cursorObj.execute('SELECT camera_id FROM homography GROUP BY camera_id')
    cameras = cursorObj.fetchall()  # list tuple [(1,), (2,)]
    with open('./camera.txt', 'w') as f:
        for row in cameras:
            f.write(str(row[0]) + '\n')
    con.close()

run_this = PythonOperator(
    task_id='start',
    python_callable=start,
    dag=dag,
)

task = PythonOperator(
    task_id='end',
    python_callable=doGlob,
    dag=dag,
)

run_this >> task

if __name__ == '__main__':
    start()
    doGlob()