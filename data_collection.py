import mediapipe as mp
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

name = input("Enter the name of the data : ")

holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

X = []
data_size = 0 

while True:
    lst = []
    
    _, frm = cap.read()
    
    frm = cv2.flip(frm,1)
    
    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    if res.face_landmarks:
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)
            
    X.append(lst)
    data_size = data_size +1
    
    drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
    #drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    #drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)
    
    cv2.imshow("window", frm)
    
    if cv2.waitKey(1) == 27 or data_size>99:
        cv2.destroyAllWindows()
        cap.release()
        break
    
np.save(f"{name}.npy", np.array(X))
print(np.array(X).shape)