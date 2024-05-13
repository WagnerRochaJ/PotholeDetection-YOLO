from ultralytics import YOLO
import cv2
import cvzone
import argparse
import sys
import os
import glob

#Definindo as entradas parse
parser = argparse.ArgumentParser()
parser.add_argument('--modelname',help='name of the yolo model name (.pt)',required=True)
parser.add_argument('--image',help='name of the image',default=None)
parser.add_argument('--imagedir',help='folder with images',default=None)
parser.add_argument('--save_results',help='save result images in a folder',action='store_true')
parser.add_argument('--threshold',help='minimun confidence',default=0.5)
args = parser.parse_args()

MODEL_NAME = args.modelname
IMG_NAME = args.image
IMG_DIR = args.imagedir
save_results = args.save_results
confidence_threshold = float(args.threshold)  


#para selecionar a imagem ou o diretorio contendo varias imagens
if(IMG_DIR and IMG_NAME):
   print("Error! only use the --image argument or the --imagedir argument, not both.")
   sys.exit()
   
#se não selecionar nenhuma da opcoes, ira por padrao na imagem pothole.jpg
if(not IMG_NAME and not IMG_DIR):
   IMG_DIR = 'testImages/pothole.jpg'
   
# pega o diretorio atual
CWD_PATH = os.getcwd()

# juntar todas as imagens
if IMG_DIR:
   PATH_TO_IMAGES = os.path.join(CWD_PATH,IMG_DIR)
   images = glob.glob(PATH_TO_IMAGES + '/*.jpg') + glob.glob(PATH_TO_IMAGES + '/*.png') + glob.glob(PATH_TO_IMAGES + '/*.bmp')
   if save_results:
      RESULTS_DIR = IMG_DIR + '_results'

elif IMG_NAME:
   PATH_TO_IMAGES = os.path.join(CWD_PATH,IMG_NAME)
   images = glob.glob(PATH_TO_IMAGES)
   if save_results:
      RESULTS_DIR = 'results'

# cria um diretorio com os resultados caso o save_results for True
if save_results:
   RESULTS_PATH = os.path.join(CWD_PATH,RESULTS_DIR)
   if not os.path.exists(RESULTS_PATH):
      os.makedirs(RESULTS_PATH)
        
#carregado o modelo
model = YOLO(MODEL_NAME)

detections = []
#carrega as classes da label
classNames = model.names

#loop para cada imagem carregada
for img_path in images:
   image = cv2.imread(img_path)
   img = cv2.resize(image,(640,480))
        
   results = model(img, stream=True)
   #loop para iterar sobre os resultados da detecção de objetos
   for r in results:
      boxes = r.boxes
      
      #loop para capturar as boxes
      for box in boxes:
         data=box.data.tolist()[0]
         confidence = data[4]
         if float(confidence) < confidence_threshold:
            continue #se o confidence for menor, o loop atual ignorado
            
         #pegando as coordenadas   
         xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
         
         #desenha o retangulo sob o objeto
         cv2.rectangle(img, (xmin, ymin) , (xmax, ymax), (0,0,240), 3)
         #carrega o nome da classe de acordo com o indice
         cls = box.cls[0]
         name = classNames[int(cls)]
         
         #desenha a box e a label classe
         cvzone.putTextRect(img, "{} {:.1f}%".format(name,float(confidence*100)), (max(0,xmin), max(35,ymin)), scale =0.8,thickness=1,colorT=(255,255,255),colorR=(0,0,0),font= cv2.FONT_HERSHEY_PLAIN,offset=5)

         #carrega alguns valores em uma lista
         detections.append([name, (confidence*100), xmin, ymin, xmax, ymax])
         
   cv2.imshow("Imagem", img)
      
      #pressione "q" para terminar o preceso
   if cv2.waitKey(0) == ord('q'):
      break  
      
   #salva a label da imagem para a pasta de results
   if save_results:
      images_fn = os.path.basename(img_path)
      image_savepath = os.path.join(CWD_PATH,RESULTS_DIR,images_fn)
      
      base_fn,ext = os.path.splitext(images_fn)
      txt_results_fn = base_fn +'.txt'
      txt_savepath = os.path.join(CWD_PATH,RESULTS_DIR,txt_results_fn)
      
      # salva a imagem
      cv2.imwrite(image_savepath,img)
      # escreve os resultados em um arquivo txt
      with open(txt_savepath,'w') as f:
               for detection in detections:
                  f.write('%s %.2f %d %d %d %d\n' % (detection[0], detection[1], detection[2], detection[3], detection[4], detection[5]))
                  
cv2.destroyAllWindows()
             
#python3 testimagem.py --modelname=bestV2.pt --image=testImages/pothole2.png