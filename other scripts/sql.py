import sqlite3


con = sqlite3.connect('/home/alex/PycharmProjects/office_camera/cameras2.sqlite3')
cursorObj = con.cursor()

cursorObj.execute('SELECT xloc, yloc, xglob, yglob FROM homography WHERE camera_id = {}'.format(1))
cameras = cursorObj.fetchall()

print(cameras)