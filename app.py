import moviepy
from flask import Flask, request, render_template
import json
from urllib.request import urlopen
from PIL import Image
from moviepy.editor import *
import numpy as np
from linearcolorinc import adjust_rgb, increase_rgb
#from colormap import 

app = Flask(__name__)
@app.route("/")
def mainfunc(): 
    return render_template("Pixelator.html")



if __name__ == '__main__':
    app.run(debug=True, port=9103)







#image = Image.open('image.jpg')
# def image_pixelate(image, pixelnum): #pixelnum will dictate the number of pixels on the larger side of the image)
#     originalHoriz = image.size[0]
#     originalVert = image.size[1]
#     if originalHoriz > originalVert:
#         return image.resize((pixelnum, int(pixelnum * (originalVert / originalHoriz))))
#     else: 
#         return image.resize((int(pixelnum * (originalHoriz/originalVert)), pixelnum))
#     #return image.resize((horiz, int(horiz * (originalVert / originalHoriz))))    # resize it to a relatively tiny size
# new_image = image_pixelate(image, 100)
# # pixeliztion is resizing a smaller image into a larger one with some resampling
# pixelated = new_image.resize(image.size,Image.NEAREST)   # resizing the smaller image to the original size
# # Image.NEARESEST is the resampling function predefined in the Image class
# pixelated.show()

