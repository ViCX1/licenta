#https://analyticsindiamag.com/face-emotion-recognizer-in-6-lines-of-code/
from fer import FER
import matplotlib.pyplot as plt 


img = plt.imread("sample.jpg")
detector = FER(mtcnn=True)
#print(detector.detect_emotions(img))
#plt.imshow(img)
emotion, score = detector.top_emotion(img)
print('Face emotion:',emotion,score)

f = open("f_test.txt", "a")


f.write(f'Face emotion: {emotion} {score}')


f.close()