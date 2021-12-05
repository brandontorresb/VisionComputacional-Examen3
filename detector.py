import cv2
import numpy as np

#cargar las clases de MSCOCO
with open('archivos/object_detection_classes_coco.txt', 'r') as f:
    class_names = f.read().split('\n')

#definir colores diferentes para cada clase
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

#cargar el DNN
model = cv2.dnn.readNet(model='archivos/frozen_inference_graph.pb',
                        config='archivos/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt', 
                        framework='TensorFlow')

#leer imagen
image = cv2.imread('foto1.jpeg')

image_height, image_width, _ = image.shape

blob = cv2.dnn.blobFromImage(image=image, size=(700, 700), mean=(25, 25, 25), swapRB=True)

model.setInput(blob)

output = model.forward()

#recorrer cada objeto detectado
for detection in output[0, 0, :, :]:
    
    confidence = detection[2]
    
    if confidence > .4:
        #obtener el id de la clase del objeto
        class_id = detection[1]
        
        class_name = class_names[int(class_id)-1]
        color = COLORS[int(class_id)]
        
        box_x = detection[3] * image_width
        box_y = detection[4] * image_height
        
        box_width = detection[5] * image_width
        box_height = detection[6] * image_height
        #dibujar un rectangulo alrededor del objeto
        cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color, thickness=2)
        
        cv2.putText(image, class_name, (int(box_x), int(box_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

cv2.imshow('image', image)
cv2.imwrite('resultado1.png', image)
cv2.waitKey(0)
