import keras
import PIL.Image
import json
import numpy as np


def upscale(image, model):
    data=model.predict(np.array([image]))[0]
    img=PIL.Image.new("L",(4,4))
    img.putdata(data)
    img=img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    img=img.rotate(90)
    return img.getdata()


def imageTo3x3Chunks(image: PIL.Image):
    chunks = []
    for i in range(0, image.width, 3):
        for j in range(0, image.height, 3):
            chunk = []
            for x in range(3):
                for y in range(3):
                    chunk.append(image.getpixel((i+x, j+y)))
            chunks.append(chunk)
    return chunks


model = keras.saving.load_model("model.keras")

image = PIL.Image.open("testing.jpg")
image = image.convert("L")
image = image.resize(((image.width//3)*3, (image.height//3)*3))
chunks = imageTo3x3Chunks(image)
upscaledChunks = [upscale(chunk, model) for chunk in chunks]
out2 = []

# 2d chucks
for i in range((image.height//3)):
    out2.append([upscaledChunks[i*(image.width//3)+k]
                for k in range((image.width//3))])

upscaledChunks = out2[:]
tmp = []
for upscaledChunk in upscaledChunks:
    a, b, c, d = [], [], [], []
    for i in upscaledChunk:
        a += [i[0], i[1], i[2], i[3]]
        b += [i[4], i[5], i[6], i[7]]
        c += [i[8], i[9], i[10], i[11]]
        d += [i[12], i[13], i[14], i[15]]
    tmp += a+b+c+d
out = PIL.Image.new("L", ((image.width//3)*4, (image.height//3)*4))
out.putdata(tmp)
out.show()
out.save("testingOut.jpg")
