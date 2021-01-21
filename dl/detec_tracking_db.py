from sys import platform
from dl.models import *  # set ONNX_EXPORT in models.py
from dl.utils.datasets import *
from dl.utils.utils import *
# from dl.tracking.sort import *
#from lstmTracking.track_for_img import  Deep_Tracker
# from dl.time_and_date import *
# import psycopg2
import uuid
import argparse
from multiprocessing import Process
import pytesseract
import datetime
import re
os.environ['TESSDATA_PREFIX'] = 'other scripts/'


# cur.execute(
#     "DELETE FROM coordinates.base.cam_coordinates"
# )
# con.commit()


def detect(opt, layer, save_img=False):
    # def detect(opt, con, cur, layer, save_img=False):
    img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    lstm_tracker = None
    print(source)
    print(webcam)

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    # if os.path.exists(out):
    #     shutil.rmtree(out)  # delete output folder
    # os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, img_size)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)


    # Eval mode
    model.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()


    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer, N_frame, frameGlob = None, None, 0, 0
    if webcam:
        # view_img = True
        save_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=img_size)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=img_size)

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    colours = np.random.rand(32, 3)  # used only for display

    # Run inference
    t0 = time.time()
    ncadr = -1
    name_cam = ''
    datastart = datetime.datetime(2020, 1, 1, 0, 0, 0)
    datatime = datetime.datetime(2020, 1, 1, 0, 0, 0)

    oldtime = datetime.time(23,59,59)
    olddate = datetime.date(2020,1,1)
    dtime = datetime.time(0,0,0)

    img = torch.zeros((1, 3, img_size, img_size), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        N_frame += 1
        frameGlob += 1
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        if lstm_tracker is not None  or datatime == datastart: # or
            # if N_frame % 4 == 0 :
            # datatime = gettime(im0s, name_cam, datastart)
            if webcam:  # batch_size >= 1
                im0 = im0s[0]
            else:
                im0 = im0s

            gray = cv2.cvtColor(im0[:80,930:], cv2.COLOR_BGR2GRAY)
            thresh = 255 - gray
            ret, thresh = cv2.threshold(thresh, 40, 255, cv2.THRESH_BINARY)
            thresh = 255 - thresh

            text = pytesseract.image_to_string(thresh,
                                               config=r'--oem 1 --psm 6', lang="digitsall_layer")
            text = text.strip()
            print(text)
            if text!='':
                try:
                    res = re.findall('\w+', text)
                    if len(res)==6:
                        datatim = datetime.datetime(int(res[0]), int(res[1]), int(res[2]), int(res[3]), int(res[4]), int(res[5]))
                        # datatim = datetime.datetime.strptime(text, '%Y-%m-%d %H:%M:%S')
                        date = datatim.date()
                        dtime = datatim.time()
                except:
                    dtime == datetime.time(0, 0, 0)

            if dtime == datetime.time(0, 0, 0):
                date = olddate
                dtime = oldtime

        if lstm_tracker is not None and dtime != oldtime and dtime != datetime.time(0,0,0):
            ncadr = -1
            oldtime = dtime
            olddate = date

        ncadr += 1
        #rint(str(ncadr), str(oldtime))


        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = torch_utils.time_synchronized()

        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms) #method = 'my'
        # print(opt)


        # Process detections
        # print(pred )
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string

            if p.find('кассы') != -1:
                name_cam = 'кассы'
                number_camer = 1
            if p.find('очередь') != -1:
                name_cam = 'очередь'
                number_camer = 2
            if p.find('рабочая зона') != -1:
                name_cam = 'рабочая зона'
                number_camer = 3
            if p.find('зал') != -1:
                name_cam = 'зал'
                number_camer = 4

            # KAV 06.05.2020
            # gf = np.random.randint(0, 2147483646, size=(1))[0]
            if det is not None and len(det) and N_frame > 1:
                detections_class = regresbox(det)
                scale_coords(img.shape[2:], detections_class, im0.shape).round().cpu()
                # Write results
                for *xyxy, conf, cls in detections_class:
                    # out_bbox = os.path.join('storage_box',
                    #                         str(number_camer),
                    #                         str(cls.item()),
                    #                         str(date.year),
                    #                         str(date.month),
                    #                         str(date.day),
                    #                         str(dtime.hour),
                    #                         str(dtime.minute),
                    #                         str(dtime.second))
                    # if not os.path.exists(out_bbox):
                    #     #shutil.rmtree(out_bbox)  # delete output folder
                    #     os.makedirs(out_bbox)  # make new output folder
                    #
                    # path_bb = os.path.join(out_bbox, '{}.jpg'.format(str(uuid.uuid4())))
                    # cv2.imwrite(path_bb, im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2]), :])
                    path_bb = ''
                    gf = dtime.hour * 10000000 + dtime.minute * 100000 + dtime.second * 1000 + ncadr

                    # тут закомментил пока
                    # cur.execute(
                    #     f"INSERT INTO coordinates.base.cam_coordinates (x1, y1, x2, y2, camera, t, type_bbox, path_to_bbox, frame, layer, score) \
                    #     VALUES ({int(xyxy[0])},{int(xyxy[1])}, {int(xyxy[2])}, {int(xyxy[3])}, {int(number_camer)}, \
                    #     '{datetime.datetime(date.year, date.month, date.day, dtime.hour, dtime.minute, dtime.second, int(1000*ncadr)) }', \
                    #     '{cls}', '{path_bb}', {gf}, '{layer}', {conf})"
                    # )
                    # con.commit()

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness = 2)

                        # print(dtime)


                # Write results
                '''for d in trackers:
                    id = d[0]
                    d = d[1:5]
                    #id = int(d[4])
                    #d = d[0:4]
                    d = [int(x) for x in d]
                    cv2.rectangle(im0, (d[0], d[1]), (d[2], d[3]), 255 * colours[id % 32, :], 2)
                    cv2.putText(im0, '{}'.format(int(id)),
                                     (d[0] + 2, d[1] + 25),
                                     cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 255, 255), 2)
                    cv2.putText(im0, '{}'.format(N_frame),
                                (10, 25),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                    cur.execute(
                        f"INSERT INTO coordinates.base.cam_coordinates (x1, y1, x2, y2, id, camera, data, time, frame) VALUES ({d[0]},{d[1]}, {d[2]}, {d[3]}, {int(id)}, {int(number_camer)}, '{date}', '{dtime}', {int(N_frame)})"
                    )
                    con.commit()'''
            if det is None:
                gf = dtime.hour * 10000000 + dtime.minute * 100000 + dtime.second * 1000 + ncadr

                # тут коммент
                # cur.execute(
                #     f"INSERT INTO coordinates.base.cam_coordinates (camera, t, frame, layer) \
                #     VALUES ( {int(number_camer)}, \
                #     '{datetime.datetime(date.year, date.month, date.day, dtime.hour, dtime.minute, dtime.second, int(1000 * ncadr))}', \
                #     {gf}, '{layer}')"
                # )
                # con.commit()

            # Print time (inference + NMS)
            #print('%sDone. (%.3fs)' % (s, t2 - t1))
            cv2.putText(im0, str(dtime), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, [225, 0, 0],
                              thickness=2, lineType=cv2.LINE_AA)

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    pass
                    #cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        print(save_path)
                        # vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))

                        #lstm_tracker = Deep_Tracker(max_iou_distance=0.3, max_age=100, n_init=10)
                        #lstm_tracker = Sort(max_age = 100)
                        lstm_tracker = True
                        N_frame = 0

                        datastart = datetime.datetime(2020, 1, 1, 0, 0, 0)
                        datatime = datetime.datetime(2020, 1, 1, 0, 0, 0)
                        date = None
                        dtime = None
                        first = True
                        name_cam = ''
                        if p.find('кассы') != -1:
                            name_cam = 'кассы'
                            number_camer = 1
                        if p.find('очередь') != -1:
                            name_cam = 'очередь'
                            number_camer = 2
                        if p.find('рабочая зона') != -1:
                            name_cam = 'рабочая зона'
                            number_camer = 3
                        if p.find('зал') != -1:
                            name_cam = 'зал'
                            number_camer = 4

                    # vid_writer.write(im0)
                cv2.imwrite('static/img/detector.jpg', im0)


    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        # vid_writer.release()
        #if platform == 'darwin':  # MacOS
            #os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))

    # тут коммент
    # con.close()

def multiDetect(source, device, layer):
    # parser = argparse.ArgumentParser()
    # opt = parser.parse_args()
    opt = paramyolo(source)

    # con = psycopg2.connect(
    #     database="coordinates",
    #     user="kommunar",
    #     password="123",
    #     host="127.0.0.1",
    #     port="5432"
    # )
    # cur = con.cursor()
    #
    # with torch.no_grad():
    #     detect(opt, con, cur, layer)
    with torch.no_grad():
        # detect(opt, layer)
        p = Process(target=runDetect, args=(opt, layer))
        p.start()

    return p



    # con.close()
def runDetect(opt, layer):
    detect(opt, layer)

class paramyolo():
    def __init__(self, source):
        self.cfg='dl/cfg/yolov3-spp.cfg'
        self.names='dl/data/cofix.names'
        self.weights='dl/weights/best_1.pt'
        self.source=source
        self.output='dl/output'
        self.img_size=640
        self.conf_thres=0.5
        self.iou_thres=0.4
        self.fourcc='mp4v'
        self.half=False
        self.device= '1'
        # self.device= 'cpu'
        self.view_img=False
        self.save_txt=False
        self.classes=None
        self.agnostic_nms=False
        self.augment=False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='./data/cofix.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='./weights/best_1.pt', help='weights path')
    parser.add_argument('--source', type=str, default='../static/uploads/oijinp_очередь_Белорусская_20200318163945_20200318185042_190337762.mp4', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='../static/img', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='1', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(opt, 'layer')
