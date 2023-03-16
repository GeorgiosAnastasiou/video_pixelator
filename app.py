import moviepy

from PIL import Image
image = Image.open('image.jpg')

def image_tiny(image, horiz):
    originalHoriz = image.size[0]
    originalVert = image.size[1]
    return image.resize((horiz, int(horiz * (originalVert / originalHoriz))))    # resize it to a relatively tiny size

new_image = image_tiny(image, 100)

# pixeliztion is resizing a smaller image into a larger one with some resampling
pixelated = new_image.resize(image.size,Image.NEAREST)   # resizing the smaller image to the original size
# Image.NEARESEST is the resampling function predefined in the Image class
pixelated.show()
