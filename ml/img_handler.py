from base64 import b64decode
import os

def imageDecoder(img_64):
    img = b64decode((img_64))
    img_file = open('../data/a/image.jpeg', 'wb')
    img_file.write(img)
    img_file.close()

def deleteImage():
    os.remove('../data/a/image.jpeg')