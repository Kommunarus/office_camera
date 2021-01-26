from clickhouse_driver import Client
import datetime
client = Client(host='localhost')

client.execute(
    "INSERT INTO dbDetector.cam_coordinates (x1, y1, x2, y2, t, type_bbox, frame, layer, score) VALUES ",
    [{'x1': int(0), 'y1': int(1), 'x2': int(2), 'y2': int(3),
      't': datetime.datetime(2022, 1, 10, 7, 10, 10, 1000),
      'type_bbox': 'cls', 'frame': 111, 'layer': 'layer', 'score': 0.5}]
)


