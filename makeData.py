import PIL.Image
import json

def imageTo3x3Chunks(image:PIL.Image)->[((int,int,int),(int,int,int),(int,int,int))]:
    chunks=[]
    for i in range(0,image.width,3):
        for j in range(0,image.height,3):
            chunk=[]
            for x in range(3):
                for y in range(3):
                    chunk.append(image.getpixel((i+x,j+y)))
            chunks.append(chunk)
    return chunks
def imageTo4x4Chunks(image:PIL.Image)->[((int,int,int,int),(int,int,int,int),(int,int,int,int),(int,int,int,int))]:
    chunks=[]
    for i in range(0,image.width,4):
        for j in range(0,image.height,4):
            chunk=[]
            for x in range(4):
                for y in range(4):
                    chunk.append(image.getpixel((i+x,j+y)))
            chunks.append(chunk)
    return chunks

image=PIL.Image.open("image.jpg")
image=image.resize((image.width//3)*3,(image.height//3)*3)
lowerRes=image.resize((image.width//4)*4,(image.height//4)*4)

x=imageTo3x3Chunks(lowerRes)
y=imageTo4x4Chunks(image)

data=[[],[]]

data[0]=x
data[1]=y
with open("imageTrainData.json","w") as f:
    json.dump(data,f)