Detecção de objetos utilizando Opencv, atraves de um modelo YOLO com dataset customizado.

Codigo de predição baseado em:

https://docs.ultralytics.com/modes/predict/#key-features-of-predict-mode

https://www.aranacorp.com/en/object-recognition-with-yolo-and-opencv/

Dependencias necessarias:

*ultralytics

*opencv-python

*cvzone

exemplo de execução>

execução com imagem:
```
python detect-imagem.py --modelname=bestV2.pt --image=testImages/pothole2.png
```
 execução para um diretorio com imagens
```
python detect-imagem.py --modelname=bestV2.pt --imagedir=testImages
```
para video(mp4):
```bash
python detect-video.py --modelname=bestV2.pt --video=testVideos/video1.mp4
```
