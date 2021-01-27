from clickhouse_driver import Client
client = Client(host='localhost')
res = client.execute(" DROP DATABASE IF EXISTS dbDetector")

res = client.execute(" CREATE DATABASE IF NOT EXISTS dbDetector")

res = client.execute("""CREATE TABLE IF NOT EXISTS dbDetector.cam_coordinates  
    (   cam_coordinates_id UUID,
        x1 Float64, 
        y1 Float64,
        x2 Float64, 
        y2 Float64, 
        camera Int32,
        t DateTime64(3, 'Europe/Moscow'), 
        type_bbox String, 
        path_to_bbox String,
        frame Int32, 
        layer String, 
        score Float64
    ) ENGINE = MergeTree()
    ORDER BY (cam_coordinates_id, t)
    SETTINGS index_granularity = 8192""")

res = client.execute("""CREATE TABLE IF NOT EXISTS dbDetector.glob_coordinates  
    (   glob_coordinates_id UUID,
        cam_coordinates_id UUID,
        x1 Float64, 
        y1 Float64,
        x2 Float64, 
        y2 Float64 
    ) ENGINE = MergeTree()
    ORDER BY glob_coordinates_id
    SETTINGS index_granularity = 8192""")

res = client.execute("""CREATE TABLE IF NOT EXISTS dbDetector.tracking  
    (   tracking_id UUID,
        cam_coordinates_id UUID,
        type_tracker String, 
        id String
    ) ENGINE = MergeTree()
    ORDER BY tracking_id
    SETTINGS index_granularity = 8192""")

