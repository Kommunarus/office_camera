from clickhouse_driver import Client
import datetime
client = Client(host='localhost')

tab = client.execute("SELECT cam_coordinates.cam_coordinates_id, \
                        cam_coordinates.x1, \
                        cam_coordinates.y1, \
                        cam_coordinates.x2, \
                        cam_coordinates.y2 \
                    FROM dbDetector.cam_coordinates  \
                    LEFT JOIN dbDetector.glob_coordinates  \
                    ON cam_coordinates.cam_coordinates_id=glob_coordinates.cam_coordinates_id \
                        WHERE glob_coordinates.glob_coordinates_id ='00000000-0000-0000-0000-000000000000' \
                        and cam_coordinates.cam_coordinates.camera = 2 \
                        and cam_coordinates.cam_coordinates.type_bbox = '1.0'\
                         "
                     )


print(len(tab))
print(tab)