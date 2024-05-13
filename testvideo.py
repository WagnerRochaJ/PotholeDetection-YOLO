from ultralytics import YOLO
import cv2
import cvzone
import argparse
import datetime

#Definindo as entradas parse
parser = argparse.ArgumentParser()
parser.add_argument('--modelname',help='name of the yolo model name (.pt)',required=True)
parser.add_argument('--video',help='name of the video')
parser.add_argument('--threshold',help='minimun confidence',default=0.5)
args = parser.parse_args()

MODEL_NAME = args.modelname
VIDEO_NAME = args.video
confidence_threshold = float(args.threshold)

#carregando o arquivo de video
cap = cv2.VideoCapture(VIDEO_NAME)
#carregado o modelo
model = YOLO(MODEL_NAME)
#carrega as classes da label
classNames = model.names

#loop para o video 
while True:
    #começa o start do tempo de execuçao 
    start = datetime.datetime.now()
    
    success, frame  =cap.read()
    
    #se nao houver mais frames, é terminado
    if not success:
        break
    
    results = model(frame,stream=True)
    h,w,a = frame.shape
    
    #loop para iterar sobre os resultados da detecção de objetos
    for r in results:
        boxes = r.boxes
        frame = cv2.resize(frame,(w,h))
        #loop para capturar as boxes
        for box in boxes:
            
            data=box.data.tolist()[0]
            confidence = data[4]
            
            if float(confidence) < confidence_threshold:
                continue #se o confidence for menor, o loop atual é ignorado
            
            #pegando as coordenadas
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            #desenha o retangulo sob o objeto
            cv2.rectangle(frame, (xmin, ymin) , (xmax, ymax), (0,0,240), 3)

            #carrega o nome da classe de acordo com o indice
            cls = box.cls[0]
            name = classNames[int(cls)]
            #desenha a box e a label classe
            cvzone.putTextRect(frame, "{} {:.1f}%".format(name,float(confidence*100)), (max(0,xmin), max(35,ymin)), scale =0.8,thickness=1,colorT=(255,255,255),colorR=(0,0,0),font= cv2.FONT_HERSHEY_PLAIN,offset=5)
    
    #termina o tempo de execucao e mostra o fps
    end = datetime.datetime.now()
    total = (end - start).total_seconds()
    print(f"tempo para 1 frame: {total * 1000:.0f} milisegundos")
    fps = f"FPS: {1/ total:.2f}"
    cv2.putText(frame,fps,(50,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(5,100,255),4)
    
    #pressione "q" para terminar
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break
   
cap.release()
cv2.destroyAllWindows()

#exemplo para a execucao >
#python3 testvideo.py --modelname=bestV2.pt --video=testVideos/video1.mp4