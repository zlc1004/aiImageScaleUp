import PIL.Image
#open testing.jpg and make it half the size
image = PIL.Image.open("testing.jpg")
image = image.convert("L")
image = image.resize((image.width//2, image.height//2))
#save the image
image.save("testing.jpg")