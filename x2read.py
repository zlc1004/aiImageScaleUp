import keras
import PIL.Image
import tqdm
import numpy as np


def upscale(image, model):
    data=model.predict(np.array([image]),verbose = 0)[0]
    img=PIL.Image.new("L",(4,4))
    img.putdata(data)
    img=img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    img=img.rotate(90)
    return img.getdata()


def imageTo3x3Chunks(image: PIL.Image):
    _pixels = list(image.getdata())
    pixels=[]
    for i in range((image.height//3)*3):
        pixels.append([_pixels[i*(image.width//3)*3+k]
                for k in range((image.width//3)*3)])
    chunks = []
    for i in range(len(pixels)//3):
        for j in range(len(pixels[0])//3):
            chunks.append(
                [
                    pixels[i*3][j*3],
                    pixels[i*3+1][j*3],
                    pixels[i*3+2][j*3],
                    pixels[i*3][j*3+1],
                    pixels[i*3+1][j*3+1],
                    pixels[i*3+2][j*3+1],
                    pixels[i*3][j*3+2],
                    pixels[i*3+1][j*3+2],
                    pixels[i*3+2][j*3+2]
                ]
            )
    return chunks


model = keras.saving.load_model("model.keras")

image = PIL.Image.open("testing.jpg")
image = image.convert("L")
image = image.resize(((image.width//3)*3, (image.height//3)*3))
chunks = imageTo3x3Chunks(image)
upscaledChunks = [upscale(chunk, model) for chunk in tqdm.tqdm(chunks)]
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
