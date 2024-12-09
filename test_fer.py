from fer import FER
import cv2

img = cv2.imread("jl.jpg")
detector = FER(mtcnn=True)
detector.detect_emotions(img)
print(detector.detect_emotions(img))
cv2.rectangle(img,257, 19, 189, 189)
emotion, score = detector.top_emotion(img) # 'happy', 0.99
print(detector.top_emotion(img))