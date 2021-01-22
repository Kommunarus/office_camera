from clickhouse_driver import Client
client = Client(host='localhost')
res = client.execute(" INSERT INTO insert_select_testtable (*) VALUES (-1, '-a', -1)  ")
res = client.execute(' SELECT * FROM insert_select_testtable')
print(res)