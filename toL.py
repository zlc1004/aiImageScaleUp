import PIL.Image
img = PIL.Image.open("testing.jpg")
img = img.convert("L")
img.save("testing.jpg")
