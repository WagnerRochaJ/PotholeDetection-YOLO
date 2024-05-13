Detecção de objetos utilizando Opencv, atraves de um modelo YOLO com dataset customizado.

Dependencias necessarias:
ultralytics
opencv-python
cvzone

exemplo de execução>
para imagem:
python detect-imagem.py --modelname=bestV2.pt --image=testImages/pothole2.png
para video(mp4):
python detect-video.py --modelname=bestV2.pt --video=testVideos/video1.mp4
