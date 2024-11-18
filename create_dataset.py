import cv2
import mediapipe as mp
import numpy as np
import time, os

def Coordinate_Normalization(joint):
    x_coordinates = []
    y_coordinates = []
    for i in range(21):
        x_coordinates.append(joint[i][0] - joint[0][0])
        y_coordinates.append(joint[i][1] - joint[0][1])

    x_left_hand = x_coordinates[:21]
    y_left_hand = y_coordinates[:21]

    if max(x_left_hand) == min(x_left_hand):
        x_left_hand_scale = np.array(x_left_hand)
    else:
        x_left_hand_scale = np.array(x_left_hand)/(max(x_left_hand)-min(x_left_hand))
    
    if max(y_left_hand) == min(y_left_hand):
        y_left_hand_scale = np.array(y_left_hand)
    else:
        y_left_hand_scale = np.array(y_left_hand)/(max(y_left_hand)-min(y_left_hand))
            
    full_scale = np.concatenate([x_left_hand_scale.flatten(),
                                 y_left_hand_scale.flatten(),])
    return full_scale

actions = ['good', 'more', 'when', 'hot', 'cold', 'apron', 'connect']
seq_length = 30
secs_for_action = 45

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

os.makedirs('dataset', exist_ok=True)

while cap.isOpened():
    for idx, action in enumerate(actions):
        data = []

        ret, img = cap.read()

        img = cv2.flip(img, 1)

        cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.imshow('img', img)
        cv2.waitKey(3000)

        start_time = time.time()

        while time.time() - start_time < secs_for_action:
            ret, img = cap.read()

            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 2))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y]

                    # Compute angles between joints
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :2] # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :2] # Child joint
                    v = v2 - v1 # [20, 2]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                    angle = np.degrees(angle) # Convert radian to degree

                    angle_label = np.array([angle], dtype=np.float32)
                    angle_label = np.append(angle_label, idx)

                    joint = Coordinate_Normalization(joint)

                    d = np.concatenate([joint, angle_label])

                    data.append(d)

                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break

        data = np.array(data)
        print(action, data.shape)
        np.save(os.path.join('dataset', f'raw_{action}'), data)

        # Create sequence data
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join('dataset', f'seq_{action}'), full_seq_data)
    break
