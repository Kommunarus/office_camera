"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016 Alex Bewley alex@dynamicdetection.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

from numba import jit
import os.path
import numpy as np
import pandas as pd
from sklearn.utils.linear_assignment_ import linear_assignment
import argparse
from filterpy.kalman import KalmanFilter
import cv2
# import psycopg2
import datetime


@jit
def iou(bb_test, bb_gt):
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return (o)


class SimpleBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 1

    def __init__(self, bbox, key_id):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.time_since_update = 0
        self.id = SimpleBoxTracker.count
        SimpleBoxTracker.count += 1
        self.hit_streak = 0
        self.age = 0

        self.pos = []
        self.pos.append(bbox)

        self.key = []
        self.key.append(key_id)

    def update(self, bbox, key_id):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.hit_streak += 1
        self.pos.append(bbox)
        self.key.append(key_id)

    def save_bd(self, id_coordinats, cur, con):
        cur.execute(
            f"INSERT INTO coordinates.base.tracking (type_tracker, key_coordinats, id) VALUES ('Sort',{id_coordinats}, {self.id})"
        )
        con.commit()

    def start(self, ):
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det[:4], trk[:4])
    matched_indices = linear_assignment(-iou_matrix)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0

    def update(self, dets, cur, con):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t in range(len(self.trackers)):
            trk = self.trackers[t]
            trk.start()
            pos = trk.pos[-1]
            trks[t, :4] = trk.pos[-1]
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, iou_threshold=0.5)

        # update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if (t not in unmatched_trks):
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(dets[d, :4][0], dets[d, 4][0])
                trk.save_bd(dets[d, 4][0], cur, con)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = SimpleBoxTracker(dets[i, :4], dets[i, 4])
            self.trackers.append(trk)

        # for i in unmatched_trks:
        #  trk = self.trackers[i]
        #  self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.pos[-1]
            key = trk.key[-1]

            if ((trk.time_since_update < self.max_age) and (
                    trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
                # ret.append(np.concatenate((d,[trk.id+1], [trk.score_detect])).reshape(1,-1)) # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id], [key])).reshape(1, -1))  # +1 as MOT benchmark requires positive

            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',
                        action='store_true')
    args = parser.parse_args()
    return args


def get_im_for_key(key_id, con):
    tab = pd.read_sql(
        f"select path_to_bbox from coordinates.base.cam_coordinates WHERE key_id = {key_id} ",
        con=con)
    orig_det = tab.to_numpy()
    image_path = orig_det[0, 0]
    return extract_image_patch(image_path, )


def extract_image_patch(image_path, ):
    # print image_path
    image_path = '../' + image_path
    image = cv2.imread(image_path)
    return image


def run_track(num_cam=2, max_age=50, min_hits=10, save_video=False, data = datetime.datetime.now(),
              timeStart = datetime.time(0,0,0), timeEnd = datetime.time(0,0,0)):
    con = psycopg2.connect(
        database="coordinates",
        user="kommunar",
        password="123",
        host="127.0.0.1",
        port="5432"
    )
    print("Database opened successfully")
    cur = con.cursor()
    d_start = datetime.datetime.combine(data, timeStart)
    d_end = datetime.datetime.combine(data, timeEnd)

    cur.execute(
        f"DELETE FROM coordinates.base.tracking \
            WHERE type_tracker = 'Sort' and \
            key_coordinats in (SELECT \
                                    key_id \
                                    FROM \
                                    coordinates.base.cam_coordinates \
                                    WHERE \
                                        camera = {num_cam} and t > '{d_start}' and t <= '{d_end}' )"
    )
    con.commit()

    if num_cam == 2:
        tab = pd.read_sql(
            f"select frame, x1, y1, x2, y2, key_id from coordinates.base.cam_coordinates \
                WHERE camera = 2 and t > '{d_start}' and t <= '{d_end}'\
                    and (  not (x1>880 and y1>370)) \
                ORDER by t ",
            con=con)
    else:
        tab = pd.read_sql(
            f"select frame, x1, y1, x2, y2, key_id from coordinates.base.cam_coordinates \
            WHERE camera = {num_cam}  and t > '{d_start}' and t <= '{d_end}' \
            ORDER by t ",
            con=con)
    orig_det = tab.to_numpy()
    total_row = tab.shape[0]
    index = 0
    size = (1280, 720)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    dtrack = Sort(max_age=max_age, min_hits=min_hits)
    SimpleBoxTracker.count = 1

    if save_video:
        out = cv2.VideoWriter('Sort-{}.avi'.format(num_cam), cv2.VideoWriter_fourcc(*'DIVX'), 25, size)

    while index < total_row:
        curr_person_id = orig_det[index, 0]

        det = orig_det[orig_det[:, 0] == curr_person_id]
        trk = dtrack.update(det[:, 1:], cur, con)

        if save_video:
            im = np.uint8(np.zeros((720, 1280, 3)))
            for t in trk:
                bbb = get_im_for_key(t[5], con)
                im[int(t[1]):int(t[1]) + int(t[3]), int(t[0]):int(t[0]) + int(t[2]), :] = bbb
                im = cv2.putText(im, '{}'.format(int(t[4])),
                                 (int(t[0]), int(t[1])),
                                 font,
                                 fontScale,
                                 fontColor,
                                 lineType)
            out.write(im)

        index = index + len(det)

    con.close()
    if save_video:
        out.release()


if __name__ == '__main__':
    run_track(num_cam=3, max_age=25, min_hits=10, save_video=False)
