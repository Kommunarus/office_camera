from __future__ import print_function
from numba import jit
import numpy as np
import pandas as pd
from sklearn.utils.linear_assignment_ import linear_assignment
import argparse
import cv2
import datetime
from clickhouse_driver import Client
import uuid

client = Client(host='localhost')

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
    return o

def bbox_iou2(bb_test, bb_gt):

    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]))
    return (o)

class SimpleBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 1

    def __init__(self, box):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.time_since_update = 0
        self.id = SimpleBoxTracker.count
        SimpleBoxTracker.count += 1
        self.hit_streak = 0
        self.age = 0

        self.boxHead = []
        self.boxShed = []
        self.boxBody = []
        self.boxHead.append(box[0,:4])
        self.boxShed.append(box[1,:4])
        self.boxBody.append(box[2,:4])


    def update(self, box):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.hit_streak += 1
        if box[0,:4].all() != 0:
            self.boxHead.append(box[0,:4])
        if box[1,:4].all() != 0:
            self.boxShed.append(box[1,:4])
        if box[2,:4].all() != 0:
            self.boxBody.append(box[2,:4])

    def save_bd(self, id_coordinats):
        for uid in id_coordinats:
            if uid != 0:
                client.execute(
                    "INSERT INTO dbDetector.tracking (type_tracker, cam_coordinates_id, id, tracking_id, t) VALUES ",
                    [{'type_tracker':'Sort',
                      'cam_coordinates_id':uid,
                      'id':str(self.id),
                      'tracking_id':uuid.uuid1(),
                      't':datetime.datetime.now()}]
                )

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
            a_0 = []
            for i in [0,1,2]:
                det_i = det[i,:4]
                trk_i = trk[i,:4]
                if det_i[0]!=0 and trk_i[0]!=0:
                    a_0.append(iou(det_i, trk_i))
            iou_matrix[d, t] = 0 if len(a_0)==0 else sum(a_0)
            # iou_matrix[d, t] = 0 if len(a_0) == 0 else (sum(a_0) / len(a_0) + 0.1 * len(a_0))
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

    def update(self, dets):
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 3, 4))
        to_del = []
        ret = []
        for t in range(len(self.trackers)):
            trk = self.trackers[t]
            trk.start()
            boxHead = trk.boxHead[-1]
            boxShed = trk.boxShed[-1]
            boxBody = trk.boxBody[-1]
            trks[t, 0, :4] = boxHead
            trks[t, 1, :4] = boxShed
            trks[t, 2, :4] = boxBody

        udet = np.zeros((len(dets), 3, 4))
        id_det = np.zeros((len(dets), 3), dtype=np.object)
        isEmpty = True
        for t in range(len(dets)):
            cls = dets[t,4]
            if pd.isna(cls):
                break
            if cls == '0.0': # shelders
                w = 1
            if cls == '1.0': # head
                w = 0
            if cls == '2.0':  # head
                w = 2
            udet[t,w,:4] = dets[t,:4]
            id_det[t,w] = dets[t, 5]
            isEmpty = False



        udet = rarUdet(udet, 1, 0)
        udet = rarUdet(udet, 2, 1)
        udet = rarUdet(udet, 2, 0)




        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(udet, trks, iou_threshold=0.15)

        # update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if (t not in unmatched_trks):
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(udet[d, :, :][0])
                trk.save_bd(id_det[d, ][0])

        # create and initialise new trackers for unmatched detections
        if not (isEmpty==True and len(dets)==1):
            for i in unmatched_dets:
                trk = SimpleBoxTracker(udet[i, :, :])
                self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        return np.empty((0, 5))

def rarUdet(udet, q1=1, q2=0):
    iomatrix = np.zeros((len(udet), len(udet)))
    for i in range(len(udet)):
        for j in range(len(udet)):
            if udet[i, q1, 0] != 0 and udet[j, q2, 0] != 0 and udet[i, q2, 0] == 0 and udet[j, q1, 0] == 0:
                iomatrix[i, j] = bbox_iou2(udet[i, q1, :4], udet[j, q2, :4])
    iomatrix[iomatrix[:, :] < 0.6] = 0

    matched_indices = linear_assignment(-iomatrix)

    dop_1a = np.count_nonzero(iomatrix, 1)
    dop_1b = np.count_nonzero(iomatrix, 0)
    dop_2 = np.nonzero(iomatrix)
    list_to_del = []
    if len(dop_2) > 0:
        num = len(dop_2[0])
        for i in range(num):
            ind_i = dop_2[0][i]
            ind_j = dop_2[1][i]
            if dop_1a[ind_i] == 1 and dop_1b[ind_j] == 1:
                udet[ind_j, q1, :] = udet[ind_i, q1, :]
                #udet[ind_i, q2, :] = udet[ind_j, q2, :]
                list_to_del.append(ind_i)
            if dop_1a[ind_i] > 1:
                j = matched_indices[np.where(matched_indices[:, 0] == ind_i)[0], 1]
                udet[j, q1, :] = udet[ind_i, q1, :]
                list_to_del.append(ind_i)

            # if dop_1b[ind_j] > 1:
            #     i = matched_indices[np.where(matched_indices[:, 1] == ind_j)[0], 0]
            #     udet[j, q1, :] = udet[ind_i, q1, :]
            #     list_to_del.append(ind_i)

        ls = list(set(list_to_del))
        ls.sort()
        for i in ls[::-1]:
            udet = np.delete(udet, i, 0)

    return udet


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',
                        action='store_true')
    args = parser.parse_args()
    return args


def run_track(num_cam=2, max_age=50, min_hits=10, data = datetime.datetime.now(),
              timeStart = datetime.time(0,0,0), timeEnd = datetime.time(0,0,0)):
    d_start = datetime.datetime.combine(data, timeStart)
    d_end = datetime.datetime.combine(data, timeEnd)

    tab = client.execute(
            f"select t, x1, y1, x2, y2, type_bbox, cam_coordinates_id from dbDetector.cam_coordinates \
            WHERE camera = {num_cam}  and t > '{d_start}' and t <= '{d_end}'\
            ORDER by t ")
    df_tab = pd.DataFrame(tab, columns=['t', 'x1', 'y1', 'x2', 'y2', 'type_bbox', 'cam_coordinates_id'])

    timeindex = pd.date_range(start=d_start, end=d_end, freq='100L', tz='Europe/Moscow').to_frame()

    result = pd.merge(timeindex, df_tab, how="left", left_on=0, right_on="t")

    dtrack = Sort(max_age=max_age, min_hits=min_hits)
    SimpleBoxTracker.count = 1


    oldframe = -1
    listRow = []
    # while index < total_row:
    for i,row in result.iterrows():
        curr_frame = row[0]

        if (curr_frame != oldframe) and oldframe != -1:
            det = np.array(listRow)
            det = delInsideBbox(det[:, 1:])

            dtrack.update(det)

            listRow = []
        listRow.append(row[1:])
        oldframe = curr_frame


def delInsideBbox(det):
    if len(det) > 1:
        i = 0
        while True:
            type_box = det[i, 4]
            key_box = det[i, 5]

            allbbox = det[(det[:, 4] == type_box)]
            bedel = False

            if len(allbbox) > 1:
                for j in range(len(allbbox)):
                    if allbbox[j, 5] != key_box:
                        iou = bbox_iou2(allbbox[j, :4], det[i, :4])
                        if iou > 0.9:
                            det = np.delete(det, i, 0)
                            bedel = True
                            break
            if not bedel:
                i += 1
            if i >= len(det):
                break
    return det



if __name__ == '__main__':
    for num_cam in [2]:
        data = datetime.datetime(2021, 1, 28)
        timeStart = datetime.time(18, 0, 0)
        timeEnd = datetime.time(19, 0, 0)
        run_track(num_cam=num_cam, max_age=15, min_hits=1, data=data,timeStart=timeStart,timeEnd=timeEnd)
