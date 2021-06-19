from Person import detectPeople
import cv2
import numpy as np
import imutils
import argparse
import numpy


argparser = argparse.ArgumentParser()

argparser.add_argument('--input', '-i', type=str, default="C:/Users/HP/.spyder-py3/YOLOV3_416/detections/video3_Trim.avi", help="C:/Users/HP/.spyder-py3/YOLOV3_416/detections/video3_Trim.avi" )

argparser.add_argument('--output', '-o', type=str, default="C:/Users/HP/.spyder-py3/YOLOV3_416/videou26.avi", help="C:/Users/HP/.spyder-py3/YOLOV3_416/video7.avi" )

argparser.add_argument('--display', '-d', type=int, default=0, help="" )

args = vars(argparser.parse_args())

cap=cv2.VideoCapture("C:/Users/HP/.spyder-py3/YOLOV3_416/detections/video3_Trim.avi")


#videonun genişliği ayarlıyoruz
w=cap.get(3)
h=cap.get(4)
frameArea=h*w
areaTH=frameArea/300




#Çizgiler
yukarı_cizgi=int(1.35*(w/2.7))
asagı_cizgi=int(2.5*(w/3.3))


print ("Red line y:",str(yukarı_cizgi))
print ("Blue line y:", str(asagı_cizgi))


yukarı=int(1.05*(w/2.7))
asagı=int(2.75*(w/3.3))

line_down_color=(255,0,0)
line_up_color=(0,0,255)

# sağdan sola renk sıralaması beyaz kırmızı mavi beyaz şeklinde olacak
pt5 =  [asagı-90,0 ];
pt6 =  [asagı-90,w ];
pts_L3 = np.array([pt5,pt6], np.int32)#beyaz
pts_L3 = pts_L3.reshape((-1,1,2))

pt3=[asagı-135,0];
pt4=[asagı-135,w];
pts_L2=np.array([pt3,pt4], np.int32)#kırmızı
pts_L2=pts_L2.reshape((-1,1,2))


pt1=[yukarı+50,0];
pt2=[yukarı+50,w];
pts_L1=np.array([pt1,pt2], np.int32)#mavi
pts_L1=pts_L1.reshape((-1,1,2))

pt7 =  [yukarı+5,0 ]
pt8 =  [yukarı+5,w ]
pts_L4 = np.array([pt7,pt8], np.int32)#beyaz
pts_L4 = pts_L4.reshape((-1,1,2))

pt10=[550,150]
pt11=[270,150]
pts_L10 = np.array([pt10,pt11], np.int32)
pts_L10 = pts_L10.reshape((-1,1,2))


labels_path = 'C:/Users/HP/.spyder-py3/YOLOV3_416/coco.names'

labels = open(labels_path).read().strip().split("\n")

# load YOLO
net = cv2.dnn.readNet("C:/Users/HP/.spyder-py3/YOLOV3_416/yolov3.cfg", "C:/Users/HP/.spyder-py3/YOLOV3_416/yolov3.weights")

# getting the layers form network
ln = net.getLayerNames()
ln = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]

# loading video
cap = cv2.VideoCapture(args["input"] if args["input"] else 0)
# cap = cv2.VideoCapture("pedestrians.py")
out = None
kisiSayisi=0
cikan=0
giren=0
#SART=0
idler=0


a=0
b=0
c=0
z=0
koordinatx=[]
koordinaty=[]
idler=[]
idler1=[]
cikx=[]
ciky=[]
SART=0
KOSUL=0
b=0
while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=800)
    frame=frame[:, 100:]
    results = detectPeople(frame, net, ln,)
   
   
    for (i, (prob, box, centroid)) in enumerate(results):
        (sX, sY, eX, eY) = box
        (cX, cY) = centroid
        color = (0,255,0)
        # draw rectangle
        cv2.rectangle(frame, (sX, sY), (eX, eY), color, 2)
        cv2.circle(frame, (cX, cY), 2, (0,0,255), 2)
        #print("ptsL2 ün 0 ı",pts_L1[0)

        #print("cnin boyutu",len(c))
       

        #numpy.append(d,cY)
        text = "id"+str(i)
        cv2.putText(frame, text, (cX,cY),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)
        
        class insan:
            def __init__(self,cX,cY,i):
                self.cX=cX
                koordinatx.append(self.cX)
                self.cY=cY
                koordinaty.append(self.cY)
                self.i=i
                idler.append(self.i)
               
   
        insan=insan(cX,cY,i)
        print("xuzunluk",len(koordinatx))
        print("yuzunluk",len(koordinaty))
       

       
        #print(insan.cX) sonuc veriyor
        a=0
        for x in range(len(koordinatx)):

            if koordinatx[a]<pt3[0]and koordinatx[a]>pt1[0] and SART ==1 and 270<koordinatx[a] and koordinatx[a]<550 and koordinaty[a]>=150 and idler[a]==i :
                SART = 0  
            if koordinatx[a]>=pt3[0] and koordinatx[a]< pt5[0]  and koordinaty[a] >= pt3[1] and koordinaty[a]<= pt4[1] and SART ==0 and 270<koordinatx[a] and koordinatx[a]<550 and koordinaty[a]>=150 and idler[a]==i  :
                SART = 1
            if koordinatx[a]>pt5[0] and koordinaty[a]>= pt5[1] and koordinaty[a]<= pt6[1] and SART == 1 and 270<koordinatx[a] and koordinatx[a]<550 and koordinaty[a]>=150 and idler[a]==i :
                giren=giren+1
                SART=0
                
            if koordinatx[a]>pt5[0] and 270<koordinatx[a] and koordinatx[a]<550 and koordinaty[a]>=150 and KOSUL==1 and idler[a]==i :
                KOSUL=0
            if koordinatx[a]>=pt3[0] and koordinatx[a]< pt5[0]  and koordinaty[a] >= pt3[1] and koordinaty[a]<= pt4[1] and 270<koordinatx[a] and koordinatx[a]<550 and koordinaty[a]>=150 and KOSUL==0 and idler[a]==i :
                KOSUL=1
            if koordinatx[a]<=pt3[0] and koordinatx[a]>=pt1[0] and 270<koordinatx[a] and koordinatx[a]<550 and koordinaty[a]>=150 and KOSUL==1 and idler[a]==i:
                cikan=cikan+1
                KOSUL=0
            a=a+1
        a=0
        b=0
        koordinaty.clear()
        koordinatx.clear()
        #idler.clear()
        if (giren-cikan)<=4:
            text = "KURALLARA UYGUN"
            cv2.putText(frame, text, (10,90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2)
        
        if (giren-cikan)>4:
            text = "KURALLARA UYGUN DEGIL"
            cv2.putText(frame, text, (10,90),
             cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0,0,255),2)

    if args["display"] > 0:
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    if args["output"] != "" and out is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(args["output"], fourcc, 25,
                                 (frame.shape[1], frame.shape[0]), True)

    frame=cv2.polylines(frame,[pts_L1],False,line_down_color,thickness=2)
    frame=cv2.polylines(frame,[pts_L2],False,line_up_color,thickness=2)
    frame=cv2.polylines(frame,[pts_L3],False,(255,255,255),thickness=1)
    frame=cv2.polylines(frame,[pts_L4],False,(255,255,255),thickness=1)
    frame=cv2.polylines(frame,[pts_L10],False,(10,60,111),thickness=1)
           
       # text = "İcerdeki Kisi Sayisi:"+str(giren-cikan)
        #cv2.putText(frame, text, (10,30),
                  #  cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 3)
        #text = "Kisi Sayisi: {}"
    text = "Giren Kisi Sayisi:"+str(giren)
    cv2.putText(frame, text, (10,30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)
    text = "Cikan Kisi Sayisi:"+str(cikan)
    cv2.putText(frame, text, (10,50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)
   
   
    text = "Icerdeki Toplam Kisi:"+str(giren-cikan)
    cv2.putText(frame, text, (10,70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)
    
        

    if out is not None:
        out.write(frame)

cap.release()
cv2.destroyAllWindows()
