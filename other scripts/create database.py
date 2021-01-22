from clickhouse_driver import Client
client = Client(host='localhost')
res = client.execute(" CREATE DATABASE IF NOT EXISTS dbDetector")

res = client.execute("""CREATE TABLE IF NOT EXISTS dbDetector.cam_coordinates  
    ( 
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
    ORDER BY t
    SETTINGS index_granularity = 8192""")
print(res)
