import cv2
import os

#Ruta de las imagenes
imagesPath="C:/Users/bug_code/Pictures/imagenes_data"

#Listamos las imagenes contenidas en la ruta
imagesPathList=os.listdir(imagesPath)

#Crear una carpeta donde se almacenen los rostros encontrados
if not os.path.exists('Rostros encontrados'):
    print("Carpeta creada: Rostros encontrados")
    os.makedirs("Rostros encontrados")

#Cargando el detector de rosotros
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

#Recocorriendo las imagenes
count = 0

for imageName in imagesPathList:
    print("Imagen name: ",imageName)
    #Leemos y mostramos
    image=cv2.imread(imagesPath+'/'+imageName)
    #cv2.imshow("imagen: ",image)
    imageAux = image.copy()
    #Transformamos a escala de grises 
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #se aplica la detencion de rostros
    faces=faceClassif.detectMultiScale(gray,1.1,2)

    for (x,y,h,w) in faces:
        cv2.rectangle(image,(x,y),(x+h,y+w),(128,0,255),2)
        #Visualizando la imagen con los rostros detectados
        cv2.rectangle(image,(10,5),(450,25),(255,255,255),-1)
        cv2.putText(image,'Presione s, para almacenar los rostros encontrados',(10,20), 2, 0.5,(128,0,255),1,cv2.LINE_AA)


        cv2.imshow("imagen", image)
        k=cv2.waitKey(0)

        if k == ord('s'):
            for (x,y,h,w) in faces:
                rostro = imageAux[y:y+w,x:x+h]
                rostro = cv2.resize(rostro,(150,150), interpolation=cv2.INTER_CUBIC)
			    #cv2.imshow('rostro',rostro)
			    
                cv2.imwrite('Rostros encontrados/rostro_{}.jpg'.format(count),rostro)
                count = count +1    
                cv2.waitKey(0)
        elif k == 27:
            break



    
cv2.destroyAllWindows()

    