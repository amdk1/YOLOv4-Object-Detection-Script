import cv2
import numpy as np
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk


yolo = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
classes = []

def object_detection(name):
    with open("coco.names", "r") as file:
        classes = [line.strip() for line in file.readlines()]
    layer_names = yolo.getLayerNames()
    output_layers = [layer_names[i - 1] for i in yolo.getUnconnectedOutLayers()]

    colorRed = (0,0,255)
    colorGreen = (0,255,0)

    img = cv2.imread(name)
    height, width, channels = img.shape

    # # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    yolo.setInput(blob)
    outputs = yolo.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(img, (x, y), (x + w, y + h), colorGreen, 3)
            cv2.putText(img, label, (x, y + 10), cv2.FONT_HERSHEY_PLAIN, 8, colorRed, 8)

    
    cv2.imshow("Image", img)
    
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    tkimage = ImageTk.PhotoImage(img)
    result.config(image=tkimage)
    result.iamge = tkimage

def search_image():                                                              
    image1 = filedialog.askopenfilename()
    if image1:
        object_detection(image1)


root = Tk()
canvas1 = Canvas(root, width = 500, height = 400)
canvas1.pack()
label1 = Label(root, text=' Object detection Project',font = "Arial 25")
canvas1.create_window(250, 150, window=label1)
button = Button(root, text = "CHOOSE", font = "Arial 15",bg='gray', command = search_image)
canvas1.create_window(250, 200, window=button)
root.mainloop()
