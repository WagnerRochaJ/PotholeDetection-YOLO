# Projeto de TCC para a detecção de buracos em vias asfaltadas

Detecção de buracos utilizando OpenCV, através de um modelo YOLO com dataset customizado.

Projeto realizado por [José Wagner](https://github.com/WagnerRochaJ) e [Lucas Pinheiro](https://github.com/gimn0).

#

<img src="resultados/pothole7.jpg" width=400 height=300>
<img src="resultados/video1YOLO.gif" width=500 height=300>
<br>

Codigo de predição baseado em:
<br>
https://docs.ultralytics.com/modes/predict/#key-features-of-predict-mode

https://www.aranacorp.com/en/object-recognition-with-yolo-and-opencv/

#
### Dependencias Necessárias:
- ultralytics
- opencv-python
- cvzone

#
### Exemplos de execução:
Execução com imagem:
```
python detect-imagem.py --modelname=bestV2.pt --image=testImages/pothole2.png
```
Execução para um diretório com imagens:
```
python detect-imagem.py --modelname=bestV2.pt --imagedir=testImages
```
Para video(mp4):
```bash
python detect-video.py --modelname=bestV2.pt --video=testVideos/video1.mp4
```
<img src="resultados/pothole4.jpg" width=400 height=300>
<img src="resultados/maps2.png" width=400 height=300>

Caso queira alterar a quantidade de detecções, adicione "--threshold= 0.5"
<br>

Exemplo para 40%:
```
python detect-imagem.py --modelname=bestV2.pt --imagedir=testImages --threshold= 0.4

```
#
### Comparação de duas imagens. Uma com 50% e outra com 30%.

<br>
<img src="resultados/50percentdetect.jpg" width=400 height=300>
Detecção a partir de 50%.
<br>

#

<br>
<img src="resultados/YOLO30percent.jpg" width=400 height=300>
Detecção a partir de 30%.
<br>
