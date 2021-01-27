from dl.models import *  # set ONNX_EXPORT in models.py
from dl.utils.datasets import *
from dl.utils.utils import *
from clickhouse_driver import Client
import argparse
from multiprocessing import Process
import pytesseract
import psutil
import uuid
import datetime
import re
import os

os.environ['TESSDATA_PREFIX'] = 'other scripts/'
client = Client(host='localhost')


# cur.execute(
#     "DELETE FROM coordinates.base.cam_coordinates"
# )
# con.commit()


def detect(opt, layer, camera_id, pid, write_bd=False, show_in_page_with_box=False, show_in_page_without_box=False,
           save_img=False):
    # def detect(opt, con, cur, layer, save_img=False):
    img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    next_step = None
    # print(source)
    # print(webcam)

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    # if os.path.exists(out):
    #     shutil.rmtree(out)  # delete output folder
    # os.makedirs(out)  # make new output folder

    # Initialize model
    if write_bd == True or show_in_page_with_box == True:

        model = Darknet(opt.cfg, img_size)

        # Load weights
        attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            load_darknet_weights(model, weights)

        # Eval mode
        model.to(device).eval()

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
    ncadr = 0
    name_cam = ''
    datastart = datetime.datetime(2020, 1, 1, 0, 0, 0)
    datatime = datetime.datetime(2020, 1, 1, 0, 0, 0)

    oldtime = datetime.time(23, 59, 59)
    olddate = datetime.date(2020, 1, 1)
    dtime = datetime.time(0, 0, 0)

    img = torch.zeros((1, 3, img_size, img_size), device=device)  # init img

    if write_bd == True or show_in_page_with_box == True:
        _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        N_frame += 1
        frameGlob += 1
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        ncadr += 1
        if show_in_page_without_box and not show_in_page_with_box:
            if webcam:  # batch_size >= 1
                cv2.imwrite('static/img/{}.jpg'.format(pid), im0s[0])
            else:
                cv2.imwrite('static/img/{}.jpg'.format(pid), im0s)

        if write_bd == True or show_in_page_with_box == True:
            # Inference
            pred = model(img, augment=opt.augment)[0]

            # to float
            if half:
                pred = pred.float()

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                       multi_label=False, classes=opt.classes,
                                       agnostic=opt.agnostic_nms)  # method = 'my'
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
                number_camer = 100

                if next_step is not None or datatime == datastart:  # or
                    # if N_frame % 4 == 0 :
                    # datatime = gettime(im0s, name_cam, datastart)

                    gray = cv2.cvtColor(im0[:80, 930:], cv2.COLOR_BGR2GRAY)
                    thresh = 255 - gray
                    ret, thresh = cv2.threshold(thresh, 35, 255, cv2.THRESH_BINARY)
                    thresh = 255 - thresh

                    text = pytesseract.image_to_string(thresh,
                                                       config=r'--oem 1 --psm 6', lang="digitsall_layer")
                    text = text.strip()
                    # print(text)
                    if text != '':
                        try:
                            res = re.findall('\w+', text)
                            if len(res) == 6:
                                datatim = datetime.datetime(int(res[0]), int(res[1]), int(res[2]), int(res[3]),
                                                            int(res[4]),
                                                            int(res[5]))
                                # datatim = datetime.datetime.strptime(text, '%Y-%m-%d %H:%M:%S')
                                date = datatim.date()
                                dtime = datatim.time()
                        except:
                            pass
                            # dtime == datetime.time(0, 0, 0)

                    if dtime == datetime.time(0, 0, 0):
                        date = olddate
                        dtime = oldtime
                # print('{} {}'.format(dtime, oldtime))

                if next_step is not None and dtime != oldtime and dtime != datetime.time(0, 0, 0):
                    ncadr = 0
                    oldtime = dtime
                    olddate = date

                # KAV 06.05.2020
                # gf = np.random.randint(0, 2147483646, size=(1))[0]
                if det is not None and len(det) and N_frame > 1:
                    detections_class = regresbox(det)
                    scale_coords(img.shape[2:], detections_class, im0.shape).round().cpu()
                    # Write results
                    for *xyxy, conf, cls in detections_class:
                        path_bb = ''
                        gf = dtime.hour * 10000000 + dtime.minute * 100000 + dtime.second * 1000 + ncadr

                        if write_bd:
                            # print(cls)
                            client.execute(
                                "INSERT INTO dbDetector.cam_coordinates \
                                (x1, y1, x2, y2, t, type_bbox, layer, score, cam_coordinates_id, camera) \
                                VALUES ",
                                [{'x1': int(xyxy[0]),
                                  'y1': int(xyxy[1]),
                                  'x2': int(xyxy[2]),
                                  'y2': int(xyxy[3]),
                                  't': datetime.datetime(date.year, date.month, date.day, dtime.hour, dtime.minute,
                                                         dtime.second, int(1000 * ncadr)),
                                  'type_bbox':str(cls.cpu().numpy()),
                                  'layer': layer,
                                  'score': conf,
                                  'cam_coordinates_id':uuid.uuid1(),
                                  'camera': camera_id,
                                  }])

                        if save_img or view_img:  # Add bbox to image
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)

                            # print(dtime)

                if det is None:
                    # gf = dtime.hour * 10000000 + dtime.minute * 100000 + dtime.second * 1000 + ncadr
                    # print(datetime.datetime.now())
                    if write_bd:
                        client.execute(
                            "INSERT INTO dbDetector.cam_coordinates (t, layer, cam_coordinates_id, camera) VALUES",
                            [{'t': datetime.datetime(date.year, date.month, date.day, dtime.hour, dtime.minute,
                                                     dtime.second, int(1000 * ncadr)),
                              'cam_coordinates_id':uuid.uuid1(),
                              # 'frame':gf,
                              'layer': layer,
                              'camera': camera_id,
                              }]
                        )
                    # con.commit()

                # Print time (inference + NMS)
                # print('%sDone. (%.3fs)' % (s, t2 - t1))
                cv2.putText(im0, str(dtime), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, [225, 0, 0],
                            thickness=2, lineType=cv2.LINE_AA)

                # Stream results
                if view_img:
                    cv2.imshow(p, im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                # Save results (image with detections)
                if save_img:
                    # print(dataset.mode)
                    if dataset.mode == 'images':
                        next_step = True
                        if show_in_page_with_box:
                            cv2.imwrite('static/img/{}.jpg'.format(pid), im0)
                    else:
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer

                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            # print(save_path)
                            # vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))

                            next_step = True
                            N_frame = 0

                            datastart = datetime.datetime(2020, 1, 1, 0, 0, 0)
                            datatime = datetime.datetime(2020, 1, 1, 0, 0, 0)
                            date = None
                            dtime = None
                            first = True
                            name_cam = ''
                            number_camer = 100

                        # vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        # vid_writer.release()
        # if platform == 'darwin':  # MacOS
        # os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))

    # тут коммент
    # con.close()


def multiDetect(source, device, layer, camera_id, write, showb, showwb):
    opt = paramyolo(source, device)

    with torch.no_grad():
        # detect(opt, layer)
        p = Process(target=runDetect, args=(opt, layer, camera_id, write, showb, showwb))
        p.start()

    return p

    # con.close()


def runDetect(opt, layer, camera_id, write, show_with_box, show_without_box):
    proc = psutil.Process()
    detect(opt, layer, camera_id, proc.pid , write, show_with_box, show_without_box)


class paramyolo():
    def __init__(self, source, device):
        self.cfg = 'dl/cfg/yolov3-spp.cfg'
        self.names = 'dl/data/cofix.names'
        self.weights = 'dl/weights/best_1.pt'
        self.source = source
        self.output = 'dl/output'
        self.img_size = 640
        self.conf_thres = 0.5
        self.iou_thres = 0.4
        self.fourcc = 'mp4v'
        self.half = False
        self.device = device
        # self.device= 'cpu'
        self.view_img = False
        self.save_txt = False
        self.classes = None
        self.agnostic_nms = False
        self.augment = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='./data/cofix.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='./weights/best_1.pt', help='weights path')
    parser.add_argument('--source', type=str,
                        default='../static/uploads/oijinp_очередь_Белорусская_20200318163945_20200318185042_190337762.mp4',
                        help='source')  # input file/folder, 0 for webcam
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
