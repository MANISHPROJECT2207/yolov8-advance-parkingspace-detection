import cv2
import numpy as np
import cvzone
import pickle 


cap = cv2.VideoCapture('easy1.mp4')
drawing = False
area_names = []

# for reading previous info from file
try:
     
    with open("manish","rb") as f:
            data = pickle.load(f)
            polylines,area_names = data['polylines'],data['area_names']
except:
    polylines = []

current_names = " "
points = []
def draw(event,x,y,flags,param):
    global points,drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        if drawing == False:
            drawing = True
            print(x,y)
            points=[(x,y)]
        else:
            print(x,y)
            drawing = False
            current_name = input('areaname:-')
            if current_name:
                area_names.append(current_name)
                polylines.append(np.array(points,np.int32))
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            print(x,y)
            points.append((x,y))
    
        



while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    frame=cv2.resize(frame,(1020,500))
    for i,polyline in enumerate(polylines):
        cv2.polylines(frame,[polyline],True,(0,0,255),2)
        # for writing a text in between the area
        cvzone.putTextRect(frame,f'{area_names[i]}',tuple(polyline[0]),1,1)
    cv2.imshow('FRAME', frame)
    cv2.setMouseCallback('FRAME',draw)
    Key = cv2.waitKey(100) & 0xFF

    # for writing info to file
    if Key==ord('s'): 
        with open("manish","wb") as f:
            data = {'polylines':polylines,'area_names':area_names}
            pickle.dump(data,f)
cap.release()
cv2.destroyAllWindows()
