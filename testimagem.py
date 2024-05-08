from ultralytics import YOLO
import cv2
import cvzone
import argparse
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument('--modelname',help='name of the yolo model name (.pt)',required=True)
parser.add_argument('--image',help='name of the image')
parser.add_argument('--imagedir',help='folder with images')
parser.add_argument('--threshold',help='minimun confidence',default=0.5)
args = parser.parse_args()

MODEL_NAME = args.modelname
IMG_NAME = args.image
IMG_DIR = args.imagedir
confidence_threshold = args.threshold  

image = cv2.imread(IMG_NAME)
model = YOLO(MODEL_NAME)

#para selecionar a imagem ou o diretorio contendo varias imagens
if(IMG_DIR and IMG_NAME):
   print("Error! Please only use the --image argument or the --imagedir argument, not both.")
   sys.exit()
   
#se não selecionar nenhuma da opcoes, ira por padrao na imagem pothole.jpg
if(not IMG_DIR and not IMG_DIR):
   IMG_DIR = 'testImages/pothole.jpg'
   
# pega o diretorio atual
CWD_PATH = os.getcwd()

img = cv2.resize(image,(640,480))
      
classNames = model.names

results = model(img, stream=True)
for r in results:
   boxes = r.boxes
   for box in boxes:
         data=box.data.tolist()[0]
         confidence = data[4]
         if float(confidence) < confidence_threshold:
            continue #se o confidence for maior, o loop atual ignorado
         
         #x1, y1, x2, y2 = box.xyxy[0]
         #x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
         #w, h = x2-x1, y2-y1
         #cvzone.cornerRect(img, (x1, y1, w, h),2,2,2,colorR=(0,0,255))
         #conf = math.ceil((box.conf[0]*100))/100
         #cvzone.putTextRect(img, f'{name} 'f'{conf}', (max(0,x1), max(35,y1)), scale = 0.5,thickness=1)
         xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
         cv2.rectangle(img, (xmin, ymin) , (xmax, ymax), (0,0,240), 3)
         cls = box.cls[0]
         name = classNames[int(cls)]
         cvzone.putTextRect(img, "{} {:.1f}%".format(name,float(confidence*100)), (max(0,xmin), max(35,ymin)), scale = 0.5,font= cv2.FONT_HERSHEY_PLAIN)

   cv2.imshow("Image", img)
   #cv2.imwrite('savef',img)
   cv2.waitKey(0)  # Aguarde até que o usuário feche a janela

#python3 testimagem.py --modelname=bestV2.pt --image=testImages/pothole2.png