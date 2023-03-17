from moviepy.editor import *
import numpy as np

# This is the second tool in our three part process.
# !!!!!!!!!!!!!!! There is a bug in the process: !!!!!!!!!!!!!!!!!!!!
# If RGB = (200, 200, 200) and you add (100, 50, 50) the result circles through 255
# You get (44, 250, 250)
# Must modify to get (255, 250, 250)

# Possible code to solve this(not tested):
# def increase_rgb(frame, r_offset=0, g_offset=0, b_offset=0):
#     # Add the specified offsets to each RGB value
#     new_frame = frame.copy()
#     new_frame[:, :, 0] = np.clip(new_frame[:, :, 0] + r_offset, 0, 255)
#     new_frame[:, :, 1] = np.clip(new_frame[:, :, 1] + g_offset, 0, 255)
#     new_frame[:, :, 2] = np.clip(new_frame[:, :, 2] + b_offset, 0, 255)
#     # Return the new frame
#     return new_frame

# Define the function to map the RGB values
def increase_rgb(frame, r_offset=0, g_offset=0, b_offset=0):
    # Add the specified offsets to each RGB value
    new_frame = frame.copy()
    new_frame[:, :, 0] += r_offset
    new_frame[:, :, 1] += g_offset
    new_frame[:, :, 2] += b_offset
    # Clip the values to be between 0 and 255
    new_frame = np.clip(new_frame, 0, 255)
    # Return the new frame
    return new_frame

# This function takes 3 arguments: the path/to/a/video, R value offset, G value offset, B value offset
# Then it adds x,y,z to the RGB values respectively of every pixel in every frame.
def adjust_rgb(videoname, x, y, z):
    # Load the video
    video = VideoFileClip(videoname)

    # Apply the function to each frame
    new_video = video.fl_image(lambda f: increase_rgb(f, r_offset=np.uint8(x), g_offset=np.uint8(y), b_offset=np.uint8(z)))

    # Save the new video
    new_video.write_videofile('new_' + videoname)

adjust_rgb("example.mp4", 255, -255, -255)