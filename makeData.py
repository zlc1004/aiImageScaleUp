import os
import json
import PIL.ImageChops
import PIL.Image
import tqdm
inputSize = 3
outputSize = 5


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


files = os.listdir("./trainImg")
x, y = [], []
print("Training data converting")
for f in tqdm.tqdm(files):
    image = PIL.Image.open("./trainImg/"+f)
    image = image.convert("L")
    image = image.resize(
        ((image.width//outputSize)*outputSize, (image.height//outputSize)*outputSize))
    y += imageToChunks(image, outputSize)
    x += resizeChunks(y, outputSize, inputSize)


with open("imageTrainData.json", "w") as f:
    print("Dumping training data")
    json.dump([x, y], f)
    print("Dumped training data")
