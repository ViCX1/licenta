import tesserocr
from PIL import Image
api = tesserocr.PyTessBaseAPI()
pil_image = Image.open('t3.jpg')
api.SetImage(pil_image)
text = api.GetUTF8Text()
print(text)