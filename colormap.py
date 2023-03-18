from moviepy.editor import *
import numpy as np

# Very slow implementation. Only for ultralowres.
# So, first use pixelation.
# Also, when the result is played on a video player
# the video is getting smooth edges. Probably an editor
# feature.


palette = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255)
]

def map_color_to_palette(r, g, b, palette):
    # Determine the index of the closest color in the palette
    closest_index = None
    min_distance = float('inf')
    for i, color in enumerate(palette):
        distance = ((r - color[0])**2 + (g - color[1])**2 + (b - color[2])**2)**0.5
        if distance < min_distance:
            min_distance = distance
            closest_index = i
    # Return the closest color from the palette
    return palette[closest_index]


# def colormapFrame(frame, palette):
#     new_frame = frame.copy()
#     new_frame[:, :, 0], new_frame[:, :, 1], new_frame[:, :, 2] = map_color_to_palette(new_frame[:, :, 0], new_frame[:, :, 1], new_frame[:, :, 2])
#     return new_frame

def map_color_to_palette_frame(frame, palette):
    # Create an empty array to hold the mapped colors
    mapped_frame = np.zeros_like(frame)
    print("frame")
    # Map each pixel in the frame to the closest color in the palette
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            r, g, b = frame[i, j]
            closest_color = map_color_to_palette(r, g, b, palette)
            mapped_frame[i, j] = closest_color

    # Return the mapped frame
    return mapped_frame

def colormap(videoname, palette):
    video = VideoFileClip(videoname)
    video = video.resize((100, 70))
    new_video = video.fl_image(lambda f: map_color_to_palette_frame(f, palette))
    new_video.write_videofile('pal_' + videoname)

colormap("Untitled.mp4", palette)