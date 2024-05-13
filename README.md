Detecção de objetos utilizando Opencv, atraves de um modelo YOLO com dataset customizado.

Dependencias necessarias:
ultralytics
opencv-python
cvzone

exemplo de execução>
python detectimagem.py --modelname=bestV2.pt --image=testImages/pothole2.png
python detectvideo.py --modelname=bestV2.pt --video=testVideos/video1.mp4
