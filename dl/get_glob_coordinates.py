import numpy as np
import cv2
import pandas as pd
import random

import psycopg2

grid = 1


def get_homography(name):
    df = pd.read_csv('../homography/Result_33.csv')
    df = df.replace(',','.', regex=True).astype(float).to_numpy()

    if name == "кассы":
        df2 = df[df[:,0]==1]
        loc = []
        glob = []
        for i in range(df2.shape[0]):
            loc.append((df2[i,1], df2[i,2]))
            glob.append((df2[i,3], df2[i,4]))

    if name == "зал":
        df2 = df[df[:,0]==4]
        loc = []
        glob = []
        for i in range(df2.shape[0]):
            loc.append((df2[i,1], df2[i,2]))
            glob.append((df2[i,3], df2[i,4]))

    if name == "очередь":
        df2 = df[df[:,0]==2]
        loc = []
        glob = []
        for i in range(df2.shape[0]):
            loc.append((df2[i,1], df2[i,2]))
            glob.append((df2[i,3], df2[i,4]))

    if name == "рабочая зона":
        df2 = df[df[:,0]==3]
        loc = []
        glob = []
        for i in range(df2.shape[0]):
            loc.append((df2[i,1], df2[i,2]))
            glob.append((df2[i,3], df2[i,4]))

    np.random.seed(1000)
    pts_src = np.array(glob)
    # Homography processing to make flat image
    pts_dst = np.array(loc)
    h, status = cv2.findHomography(pts_dst, pts_src, cv2.RANSAC, 3)
    #h, status = cv2.findHomography(pts_dst, pts_src)
    return h, status


def glob_coord(x, y, h):
    test_point = (x, y, 1)
    new_coord = h.dot(test_point)
    xx = new_coord[0] / new_coord[2]
    yy = new_coord[1] / new_coord[2]

    return xx, yy

def doGlob(data):
    con = psycopg2.connect(
        database="coordinates",
        user="kommunar",
        password="123",
        host="127.0.0.1",
        port="5432"
    )
    cur = con.cursor()


    h1, status1 = get_homography('кассы')
    h2, status2 = get_homography('очередь')
    h3, status3 = get_homography('рабочая зона')
    h4, status4 = get_homography('зал')

    if True:
        tab = pd.read_sql(f"select A.key_id, A.camera, A.x1, A.y1, A.x2, A.y2 from coordinates.base.cam_coordinates as A \
                            left join coordinates.base.glod_coordinates as B on A.key_id=B.key_coordinats \
                                where B.x1 IS NULL and A.camera in(2,3,4) and date_trunc('day', A.t) = '{data}' and A.type_bbox = '1.0'",
                          con=con)

        cur.execute(
            f"DELETE FROM coordinates.base.glod_coordinates WHERE key_coordinats in (select A.key_id\
                                from coordinates.base.cam_coordinates as A \
                            left join coordinates.base.glod_coordinates as B on A.key_id=B.key_coordinats \
                                where B.x1 IS NULL and A.camera in(2,3,4) and date_trunc('day', A.t) = '{data}' and A.type_bbox = '1.0') "
        )
        con.commit()

        for i, row in tab.iterrows():
            if row.camera == 1:
                h = h1
            if row.camera == 2:
                h = h2
            elif row.camera == 3:
                h = h3
            elif row.camera == 4:
                h = h4

            x1 = row.x1
            y1 = row.y1
            x1glob, y1glob = glob_coord(x1, y1, h)
            x2 = row.x2
            y2 = row.y2
            x2glob, y2glob = glob_coord(x2, y2, h)
            x1glob = 0 if np.isnan(x1glob) else x1glob
            y1glob = 0 if np.isnan(y1glob) else y1glob
            x2glob = 0 if np.isnan(x2glob) else x2glob
            y2glob = 0 if np.isnan(y2glob) else y2glob

            #if row.camera == 20:
            #    x2glob -= 0.3
            #    y2glob -= 0.2

            cur.execute(
                f"INSERT INTO coordinates.base.glod_coordinates (x1, y1, x2, y2, key_coordinats) VALUES ({x1glob}, {y1glob},\
                            {x2glob}, {y2glob}, {row.key_id})"
            )

            con.commit()

    #print(glob_coord(880, 370, h2))


