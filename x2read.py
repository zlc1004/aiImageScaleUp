import keras
import PIL.Image
import tqdm
import numpy as np


def upscale(image, model):
    data=model.predict(np.array([image]),verbose = 0)[0]
    img=PIL.Image.new("L",(5,5))
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
def imageTo4x4Chunks(image: PIL.Image):
    _pixels = list(image.getdata())
    pixels=[]
    for i in range((image.height//4)*4):
        pixels.append([_pixels[i*(image.width//4)*4+k]
                for k in range((image.width//4)*4)])
    chunks = []
    for i in range(len(pixels)//4):
        for j in range(len(pixels[0])//4):
            chunks.append(
                [
                    pixels[i*4][j*4],
                    pixels[i*4+1][j*4],
                    pixels[i*4+2][j*4],
                    pixels[i*4+3][j*4],
                    pixels[i*4][j*4+1],
                    pixels[i*4+1][j*4+1],
                    pixels[i*4+2][j*4+1],
                    pixels[i*4+3][j*4+1],
                    pixels[i*4][j*4+2],
                    pixels[i*4+1][j*4+2],
                    pixels[i*4+2][j*4+2],
                    pixels[i*4+3][j*4+2],
                    pixels[i*4][j*4+3],
                    pixels[i*4+1][j*4+3],
                    pixels[i*4+2][j*4+3],
                    pixels[i*4+3][j*4+3],
                ]
            )
    return chunks


model = keras.saving.load_model("model.keras")

image = PIL.Image.open("testing.jpg")
image = image.convert("L")
image = image.resize(((image.width//4)*4, (image.height//4)*4))
chunks = imageTo4x4Chunks(image)
upscaledChunks = [upscale(chunk, model) for chunk in tqdm.tqdm(chunks)]
out2 = []

# 2d chucks
for i in range((image.height//4)):
    out2.append([upscaledChunks[i*(image.width//4)+k]
                for k in range((image.width//4))])

upscaledChunks = out2[:]
tmp = []
for upscaledChunk in upscaledChunks:
    a, b, c, d,e = [], [], [], [],[]
    for i in upscaledChunk:
        a += [i[0], i[1], i[2], i[3], i[4]]
        b += [i[5], i[6], i[7], i[8], i[9]]
        c += [i[10], i[11], i[12], i[13], i[14]]
        d += [i[15], i[16], i[17], i[18], i[19]]
        e += [i[20], i[21], i[22], i[23], i[24]]
    tmp += a+b+c+d+e
out = PIL.Image.new("L", ((image.width//4)*5, (image.height//4)*5))
out.putdata(tmp)
out.show()
out.save("testingOut.jpg")
