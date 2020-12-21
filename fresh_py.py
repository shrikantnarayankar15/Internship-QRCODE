from pyzbar.pyzbar import decode
import cv2
import numpy as np
from pyzbar.pyzbar import ZBarSymbol
img = cv2.imread('/home/striker/Downloads/IMG_20201217_145025.jpg')
from image_stab import Stabilizer



def detect_zbar(img):

    for qrcode in decode(img,symbols=[ZBarSymbol.QRCODE]):
        mydata = qrcode.data.decode('utf-8')
        print(mydata)
        pts = np.array([qrcode.polygon], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img, [pts], True, (0,0,0), 5)
        # print(pts)
        pts2 = qrcode.rect
        # print(pts2)
        cv2.putText(img, mydata, (pts2.left, pts2.top), cv2.FONT_HERSHEY_SIMPLEX,0.9, (0,0,0))

    return img

# cap.set(cv2.CAP_PROP_FPS,10)
def detect_QRcode(image):
    # image = cv2.imread('1.jpg')
    original = image.copy()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    mask_gray = cv2.inRange(gray, 0,200)
    mask_gray = (255-mask_gray)
    
    img = detect_zbar(mask_gray)

    cv2.imshow('image', img)

    cv2.waitKey(1)

    return image    

cap = cv2.VideoCapture('lol.mp4')
codec = cv2.VideoWriter_fourcc(*"mp4v")
# vid_fps =20
vid_fps =int(cap.get(cv2.CAP_PROP_FPS))
# vid_width,vid_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('ujuju.mp4', codec, vid_fps, (1000,800))
import time
start = time.time()
while cap.isOpened():
    success, img = cap.read()

    if not success and img is None: 
        break
    
    img = detect_QRcode(img)
    img = cv2.resize(img, (1000,800))

    out.write(img)

    

end = time.time()
print("time", (end-start))
cap.release()
out.release()
cv2.destroyAllWindows()