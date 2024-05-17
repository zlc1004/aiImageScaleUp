import PIL.Image
import json

def imageTo3x3Chunks(image:PIL.Image):
    chunks=[]
    for i in range(0,image.width,3):
        for j in range(0,image.height,3):
            chunk=[]
            for x in range(3):
                for y in range(3):
                    chunk.append(image.getpixel((i+x,j+y)))
            chunks.append(chunk)
    return chunks
def imageTo4x4Chunks(image:PIL.Image):
    chunks=[]
    for i in range(0,image.width,4):
        for j in range(0,image.height,4):
            chunk=[]
            for x in range(4):
                for y in range(4):
                    chunk.append(image.getpixel((i+x,j+y)))
            chunks.append(chunk)
    return chunks

def imageTo5x5Chunks(image:PIL.Image):
    chunks=[]
    for i in range(0,image.width,5):
        for j in range(0,image.height,5):
            chunk=[]
            for x in range(5):
                for y in range(5):
                    chunk.append(image.getpixel((i+x,j+y)))
            chunks.append(chunk)
    return chunks

def take4x4(chunks):
    chunkOut=[]
    for i in chunks:
        chunkOut.append([i[0],i[1],i[2],i[3],
                         i[5],i[6],i[7],i[8],
                         i[10],i[11],i[12],i[13],
                         i[15],i[16],i[17],i[18]])
    return chunkOut

image=PIL.Image.open("image.jpg")
image=image.resize(((image.width//5)*5,(image.height//5)*5))


y=imageTo5x5Chunks(image)
x=take4x4(y)

data=[x,y]

with open("imageTrainData.json","w") as f:
    json.dump(data,f)