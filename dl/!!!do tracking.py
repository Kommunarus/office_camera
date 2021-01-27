from tracking.get_glob_coordinates import doGlob
from tracking.updateFrame import doNewFrame
import tracking.sort as tS
import tracking.multiSort as mS
from tracking.end_union_track_II import run_track
import datetime
from multiprocessing import Process, Pool

def runDetect(num_cam):
    # tA.run_track(num_cam = num_cam, max_age=max_age, min_hits=min_hits)
    # tI.run_track(num_cam = num_cam, max_age=max_age, min_hits=min_hits)
    # tM.run_track(num_cam = num_cam, max_age=max_age, min_hits=min_hits)
    # tS.run_track(num_cam = num_cam, max_age=max_age, min_hits=min_hits, data = data,
    #              timeStart = timeStart, timeEnd = timeEnd)
    mS.run_track(num_cam = num_cam, max_age=max_age, min_hits=min_hits, data = data,
                 timeStart = timeStart, timeEnd = timeEnd)
    # ut.run_track(num_cam = num_cam, max_age=max_age)
    #mut.run_track(num_cam = num_cam, max_age=max_age, data = data)

datalist = [datetime.datetime(2020,3,11),
            datetime.datetime(2020,3,12),
            datetime.datetime(2020,3,15),
            datetime.datetime(2020,3,16),
            datetime.datetime(2020,3,17),
            datetime.datetime(2020,3,18)]

timeStart = datetime.time(7, 0, 0)
timeEnd = datetime.time(23, 0, 0)
max_age = 10
min_hits = 1

for data in datalist:
    # data = datetime.datetime(2020,3,14)
    print(data)

    s = datetime.datetime.now()
    print('Start do glob')
    doGlob(data)
    e1 = datetime.datetime.now()
    print(str(e1-s))

    print('Start sort tracking')
    # pool = Pool(12)
    # pool.map(runDetect, [2,3,4])
    for i in [2,3,4]:
         runDetect(i)
    e3 = datetime.datetime.now()
    print(str(e3-s))

    # print('Start multi track 1,2,3,4')
    # run_track(data)
    # e4 = datetime.datetime.now()
    # print(str(e4-s))
