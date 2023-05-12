import tensorflow as tf ;
import numpy as np;
from matplotlib import pyplot as plt;
import cv2

interpreter = tf.lite.Interpreter(model_path='lite-model_movenet_singlepose_lightning_3.tflite') # โหลดโมเดลจาก....
interpreter.allocate_tensors()

def draw_keypoint (frame , keypoint , confidance_t):
    y , x , c  = frame.shape
    shaped = np.squeeze(np.multiply(keypoint , [y,x,1]))
    for kp in shaped:
        ky , kx , kp_conf = kp
        if kp_conf > confidance_t:
            cv2.circle(frame,(int(kx),int(ky)),4,(0,255,0),-1)

# สร้างเส้นเชื่อม -----------------------------------------

def draw_connection(frame , keypoint , edges , confidance_t):
    y , x, c =frame.shape
    shaped = np.squeeze(np.multiply(keypoint , [y,x,1]))

    for edge , color in edges.items():
        p1,p2 = edge 
        y1 , x1 , c1 = shaped[p1]
        y2 , x2 , c2 = shaped[p2]

        if (c1 >confidance_t) & (c2>confidance_t):
            cv2.line(frame,(int(x1),int(y1)),(int(x2),int(y2)) , (0,255,0) , 2)

#----------EDGES เส้น
EDGES = {
   (0,1):'m',
   (0,2):'c',
   (1,3):'m',
   (2,4):'c',
   (0,5):'m',
   (0,6):'c',
    (5,7):'m',
    (7,9):'m',
    (6,8):'c',
    (8,10):'c',
    (5,6):'y',
    (5,11):'m',
    (6,12):'c',
    (11,12):'y',
    (11,13):'m',
    (13,15):'m',
    (12,14):'c',
    (14,16):'c'
}

# หาขนาดรูปภาพ
# 

score_model = []
face_pos = []
# img_parth = '0.avi'
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret , frame  = cap.read() 

    frame = cv2.resize(frame, (384, 384)) 
    frame =cv2.flip(frame , 1)

#ปรับ Shape image -> 192,192 
    img = frame.copy() #คัดลอกเฟรมรูปภาพ
    img = tf.expand_dims(img, axis=0) # expand_dims(img,axis=0)คือเพิ่มมิติอาร์เรย์
    img = tf.image.resize_with_pad(img, 192, 192) #เปลี่ยนขนาดโดยการเพิ่มกรอบส่วนที่ขาด
    input_image = tf.cast(img, dtype=tf.float32) #เปลี่ยนรูปภาพเป็นประเภทใหม่ uint8


#-----------> ตั้งค่าการรับข้อมูลและดารส่งข้อมูล <---------------------
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'],np.array(input_image))
    interpreter.invoke()

    KeyPoint_with_scores = interpreter.get_tensor(output_details[0]['index'])
    score_model.append(KeyPoint_with_scores)
    #print(KeyPoint_with_scores)

#--------------------> วาดเส้น <-----------------------

    
    draw_connection(frame , KeyPoint_with_scores , EDGES  , 0.2)
    draw_keypoint(frame , KeyPoint_with_scores,0.2)
    print(frame.shape)

    
    cv2.imshow('test',frame)
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
        