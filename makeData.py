inputSize=3
outputSize=5

import PIL.Image
import json
import PIL.ImageChops


def imageToChunks(image: PIL.Image, chunkSize: int):
    chunks = []
    for i in range(0, image.width, chunkSize):
        for j in range(0, image.height, chunkSize):
            chunk = []
            for x in range(chunkSize):
                for y in range(chunkSize):
                    chunk.append(image.getpixel((i+x, j+y)))
            chunks.append(chunk)
    return chunks


def resizeChunks(chunks, inputSize, outputSize):
    chunkOut = []
    for i in chunks:
        img = PIL.Image.new("L", (inputSize, inputSize))
        img.putdata(i)
        img = img.resize((outputSize, outputSize))
        chunkOut.append(list(img.getdata()))
    return chunkOut


image = PIL.Image.open("image.jpg")
image = image.resize(((image.width//5)*5, (image.height//5)*5))


y = imageToChunks(image, outputSize)
x = resizeChunks(y, outputSize, inputSize)

data = [x, y]

with open("imageTrainData.json", "w") as f:
    json.dump(data, f)
