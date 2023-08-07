import cv2
import numpy as np
import mediapipe
import pyautogui


def draw_transparent_text(img, text, position, font, scale, color, thickness, transparency):
    overlay = img.copy()
    cv2.putText(overlay, text, position, font, scale, color, thickness)
    cv2.addWeighted(overlay, transparency, img, 1 - transparency, 0, img)

frame_hight = 240
frame_width = 320

#transparent lines
y1 = frame_hight // 3
y2 = (2 * frame_hight) // 3

drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands

cap = cv2.VideoCapture(0)

with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2) as hands:
    while True:
        ret, frame = cap.read()
        frame1 = cv2.resize(frame, (frame_width, frame_hight))
        results = hands.process(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
        
        # draw lines and add text
        cv2.line(frame1, (0, y1), (frame_width, y1), (255, 255, 255), 1)
        cv2.line(frame1, (0, y2), (frame_width, y2), (255, 255, 255), 1)
        draw_transparent_text(frame1, "JUMP", (10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, 0.6)
        draw_transparent_text(frame1, "IDLE", (10, (y1 + y2) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, 0.6)
        draw_transparent_text(frame1, "DUCK", (10, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, 0.6)
        
        if results.multi_hand_landmarks:
            for handLandmarks in results.multi_hand_landmarks:
                drawingModule.draw_landmarks(frame1, handLandmarks, handsModule.HAND_CONNECTIONS)
                
                for point in handsModule.HandLandmark:
                    normalizedLandmark = handLandmarks.landmark[point]
                    pixelCoordinatesLandmark = drawingModule._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, 640, 480)
                    
                    if point == 8:
                        finger_tip_y = int(normalizedLandmark.y*frame_hight)
                        if finger_tip_y < y1:
                            #print("jump")
                            pyautogui.press('up')
                        if y2 > finger_tip_y > y1:
                            #print("idle")
                            pyautogui.keyUp('down')
                            pyautogui.keyUp('up')
                        if finger_tip_y > y2:
                            #print("duck")
                            pyautogui.keyUp('up')
                            pyautogui.keyDown('down')
        cv2.imshow("Frame", frame1)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
