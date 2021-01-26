import cv2
import pytesseract
from datetime import datetime
import os
import re
os.environ['TESSDATA_PREFIX'] = ''
img = cv2.imread('../static/img/154635.jpg')[:80,930:]

# img = get_grayscale(img)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
thresh = 255 - gray
ret, thresh = cv2.threshold(thresh, 40 , 250, cv2.THRESH_BINARY)
thresh = 255 - thresh

# showing the image
cv2.imshow("img", thresh)
cv2.waitKey()





# img = cv2.medianBlur(img, 3)
# Adding custom options
custom_config = '--oem 1 --psm 6 '
text = pytesseract.image_to_string(thresh, config=custom_config, lang="digitsall_layer" ) # digitsall_layer  digits_comma
text = text.strip()
print(text)
res=re.findall('\w+', text)
print(len(res))
datatim = datetime(int(res[0]), int(res[1]), int(res[2]), int(res[3]), int(res[4]), int(res[5]))
print('datatim {}'.format(datatim))
# date_time_obj = datetime.strptime(text, '%Y-%m-%d %H.%M.%S')
# print(date_time_obj)
# cv2.imshow('',thresh)
# cv2.waitKey()
from pytesseract import Output
# height = img.shape[0]
# width = img.shape[1]

# d = pytesseract.image_to_boxes(img, output_type=Output.DICT, config=custom_config, lang="digits")
# n_boxes = len(d['char'])
# for i in range(n_boxes):
#     (text,x1,y2,x2,y1) = (d['char'][i],d['left'][i],d['top'][i],d['right'][i],d['bottom'][i])
#     cv2.rectangle(img, (x1,height-y1), (x2,height-y2) , (255,255,0), 2)
#     print(text)
# cv2.imshow('img',img)
# cv2.waitKey(0)