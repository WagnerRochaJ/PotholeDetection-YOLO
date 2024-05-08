#dependencias > 
# ultralytics
# opencv

#codigo baseado neste topico https://www.aranacorp.com/en/object-recognition-with-yolo-and-opencv/

from ultralytics import YOLO
import cv2
import cvzone
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--modelname',help='name of the yolo model name (.pt)',required=True)
parser.add_argument('--video',help='name of the video')
parser.add_argument('--threshold',help='minimun confidence',default=0.5)
args = parser.parse_args()

MODEL_NAME = args.modelname
VIDEO_NAME = args.video
confidence_threshold = args.threshold

cap = cv2.VideoCapture(VIDEO_NAME)
model = YOLO(MODEL_NAME)
classNames = model.names

while True:
    success, img  =cap.read()
    if not success:
        break
    
    results = model(img,stream=True)
    h,w,a = img.shape
    
    for r in results:
        boxes = r.boxes
        img = cv2.resize(img,(w,h))
        for box in boxes:
            data=box.data.tolist()[0]
            confidence = data[4]
            if float(confidence) < confidence_threshold:
                continue #se o confidence for maior, o loop atual ignorado
            
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            cv2.rectangle(img, (xmin, ymin) , (xmax, ymax), (0,0,240), 3)
            # x1, y1, x2, y2 = box.xyxy[0]
            # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # w, h = x2-x1, y2-y1
            # cvzone.cornerRect(img, (x1, y1, w, h),2,2,2,colorC=(0,0,255))
            #conf = math.ceil((box.conf[0]*100))/100

            cls = box.cls[0]
            name = classNames[int(cls)]
            cvzone.putTextRect(img, "{} {:.1f}%".format(name,float(confidence*100)), (max(0,xmin), max(35,ymin)), scale = 0.5,font= cv2.FONT_HERSHEY_PLAIN)
            #cvzone.putTextRect(img, f'{name} 'f'{(box.conf)}', (max(0,xmin), max(35,ymin)), scale = 0.5,font= cv2.FONT_HERSHEY_PLAIN)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break
cap.release()
cv2.destroyAllWindows()

#exemplo para a execucao >
#python3 testvideo.py --modelname=bestV2.pt --video=testVideos/video1.mp4