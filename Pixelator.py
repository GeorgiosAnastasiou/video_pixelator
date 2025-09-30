# Cell 1: Install dependencies
# core libs + ffmpeg CLI for remuxing
!apt-get update -qq && apt-get install -qq -y ffmpeg
!pip install --quiet opencv-python ipywidgets numpy matplotlib scikit-learn ffmpeg-python tqdm

# Cell 2: Imports
import os
import glob
import json
import ffmpeg
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KDTree
from ipywidgets import (
    IntSlider, Dropdown, Button, VBox, HBox, Output, Text,
    Layout, ColorPicker, Label, HTML, FloatProgress
)
from IPython.display import display, FileLink, HTML as HTMLDisplay

# Cell 3: Helper: find input videos

def list_input_videos(folder='/content'):
    return [f for f in os.listdir(folder)
            if f.lower().endswith('.mp4') and 'output' not in f.lower()]

# Cell 4: Frame extraction & FPS sampling with progress

def extract_frames(video_path: str, target_fps: int, progress_bar: FloatProgress=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    available = [f for f in (30,25,24) if f <= orig_fps]
    if target_fps not in available:
        raise ValueError(f"FPS {target_fps} unavailable; choose from {available}")
    interval = int(round(orig_fps/target_fps))
    frames = []
    idx = 0
    if progress_bar:
        progress_bar.max = total_frames
        progress_bar.value = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
        if progress_bar:
            progress_bar.value = idx
    cap.release()
    return frames, orig_fps

# Cell 5: Pixelation pipeline

def adjust_rgb(image, offsets):
    img = image.astype(np.int16)
    for i in range(3):
        img[:,:,i] = np.clip(img[:,:,i] + offsets[i], 0, 255)
    return img.astype(np.uint8)

def pixelate_image(image, bw, bh):
    h, w = image.shape[:2]
    small = cv2.resize(image, (bw, bh), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def map_to_palette(image, palette):
    flat = image.reshape(-1, 3)
    tree = KDTree(np.array(palette))
    _, idx = tree.query(flat)
    mapped = np.array(palette)[idx.flatten()]
    return mapped.reshape(image.shape).astype(np.uint8)

# Cell 6: Video processing with progress bars

def process_video(in_path, out_path, fps, offsets, bw, bh, palette, smooth, proc_bar: FloatProgress, fix_bar: FloatProgress):
    # extraction + pixelation
    frames, _ = extract_frames(in_path, fps, progress_bar=proc_bar)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    prev = None
    alpha = smooth / 100.0
    proc_bar.max = len(frames)
    proc_bar.value = 0
    for i, f in enumerate(frames):
        adj = adjust_rgb(f, offsets)
        pix = pixelate_image(adj, bw, bh)
        if prev is None or smooth == 0:
            blended = pix.astype(np.float32)
        else:
            blended = (1 - alpha) * pix + alpha * prev
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        mapped = map_to_palette(blended, palette)
        writer.write(cv2.cvtColor(mapped, cv2.COLOR_RGB2BGR))
        prev = blended
        proc_bar.value = i+1
    writer.release()
    # remux/re-encode
    fixed = os.path.splitext(out_path)[0] + '_fixed.mp4'
    # estimate total for fix
    fix_bar.max = proc_bar.max
    fix_bar.value = 0
    (
        ffmpeg
        .input(out_path)
        .output(fixed, vcodec='libx264', pix_fmt='yuv420p', acodec='aac')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    # crude: just set fix_bar to max when done
    fix_bar.value = fix_bar.max
    return fixed


# Cell 7: Palette Manager
PALETTE_STORE = '/content/palettes.json'

def load_palettes():
    if os.path.exists(PALETTE_STORE):
        with open(PALETTE_STORE, 'r') as f:
            return json.load(f)
    return {'Default': ['#FF0000', '#00FF00', '#0000FF']}

def save_palettes(p):
    with open(PALETTE_STORE, 'w') as f:
        json.dump(p, f)

palettes = load_palettes()
remove_mode = False

# Widgets
palette_dd     = Dropdown(options=list(palettes.keys()), description='Palette:')
create_btn     = Button(description='Create new empty palette')
name_txt       = Text(description='Name:')
save_name_btn  = Button(description='Save name', button_style='success')
delete_label   = HTML("<div style='font-size:12px;text-align:center;'>delete current palette</div>")
delete_btn     = Button(description='X', button_style='danger', layout=Layout(width='24px', height='24px'))
picker         = ColorPicker(description='Pick Color:')
add_btn        = Button(description='Add Color')
remove_btn     = Button(description='Remove Color')
preview_out    = Output()

# Preview function

def show_preview(name):
    global remove_mode
    preview_out.clear_output()
    cols = palettes.get(name, [])
    n = len(cols) if cols else 1
    with preview_out:
        if remove_mode and cols:
            btns = []
            for i in range(n):
                btn = Button(description='X', button_style='danger',
                              layout=Layout(width=f'{600/n}px', height='20px', padding='0', margin='0'))
                btn.on_click(lambda b, i=i: remove_color(i, name))
                btns.append(btn)
            display(HBox(btns))
        # Single-row swatch bar
        html = (
            "<div style='border:6px solid #000; width:600px; height:60px; "
            "display:flex; flex-wrap:nowrap; overflow-x:auto;'>"
        )
        for c in cols:
            html += f"<div style='flex:1; background:{c};'></div>"
        html += "</div>"
        display(HTML(html))

def remove_color(idx, nm):
    global remove_mode
    palettes[nm].pop(idx)
    save_palettes(palettes)
    remove_mode = False
    show_preview(nm)

# Event handlers

def on_palette_change(change):
    global remove_mode
    remove_mode = False
    show_preview(change['new'])

def on_create(b):
    create_btn.layout.display = 'none'
    name_txt.layout.display = 'inline-block'
    save_name_btn.layout.display = 'inline-block'

def on_save_name(b):
    nm = name_txt.value.strip()
    if nm and nm not in palettes:
        palettes[nm] = []
        save_palettes(palettes)
        # update and select
        palette_dd.options = list(palettes.keys())
        palette_dd.value = nm
    name_txt.layout.display = 'none'
    save_name_btn.layout.display = 'none'
    create_btn.layout.display = 'inline-block'
    show_preview(palette_dd.value)

def on_delete(b):
    nm = palette_dd.value
    if nm != 'Default':
        palettes.pop(nm)
        save_palettes(palettes)
        opts = list(palettes.keys())
        palette_dd.options = opts
        palette_dd.value = opts[0]


def on_add(b):
    nm = palette_dd.value
    col = picker.value
    if nm and col:
        palettes[nm].append(col)
        save_palettes(palettes)
        show_preview(nm)

def on_remove(b):
    global remove_mode
    remove_mode = True
    show_preview(palette_dd.value)

# Wire up
palette_dd.observe(on_palette_change, names='value')
create_btn.on_click(on_create)
save_name_btn.on_click(on_save_name)
delete_btn.on_click(on_delete)
add_btn.on_click(on_add)
remove_btn.on_click(on_remove)

# Hide creation UI initially
name_txt.layout.display = 'none'
save_name_btn.layout.display = 'none'

# Display
ui = VBox([
    HBox([palette_dd, create_btn, name_txt, save_name_btn, delete_label, delete_btn]),
    HBox([picker, add_btn, remove_btn]),
    preview_out
])
display(ui)
show_preview(palette_dd.value)

# Cell 8: UI: Video Settings & Run

# File dropdown
file_dd      = Dropdown(options=list_input_videos(), description='Input File:')
# real-time palette label
palette_label= Label()
# parameter widgets
fps_slider   = IntSlider(value=24, min=1, max=30, description='FPS:')
r_slider     = IntSlider(value=0, min=-255, max=255, description='R Off:')
g_slider     = IntSlider(value=0, min=-255, max=255, description='G Off:')
b_slider     = IntSlider(value=0, min=-255, max=255, description='B Off:')
block_w      = IntSlider(value=192, min=10, max=960, description='Blocks W:')
smooth_slider= IntSlider(value=0, min=0, max=75, description='Smooth %:')
run_btn      = Button(description='Run Process')
video_log    = Output()
proc_bar     = FloatProgress(value=0, description='Processing:')
fix_bar      = FloatProgress(value=0, description='Fixing:')

# update palette label
def update_palette_label(change):
    palette_label.value = 'Palette: ' + palette_dd.value
palette_dd.observe(update_palette_label, names='value')
update_palette_label(None)

# action
def on_run_click(b):
    with video_log:
        video_log.clear_output()
        inp = file_dd.value
        outname = os.path.splitext(inp)[0] + f'output{len(glob.glob(f"/content/{os.path.splitext(inp)[0]}output*.mp4"))+1}.mp4'
        display(proc_bar)
        display(fix_bar)
        offsets = (r_slider.value, g_slider.value, b_slider.value)
        bw = block_w.value
        # compute bh
        frames, _ = extract_frames(inp, fps_slider.value)
        h,w = frames[0].shape[:2]
        bh = int(round(bw*h/w))
        pal = [tuple(int(hx.lstrip('#')[i:i+2],16) for i in (0,2,4)) for hx in palettes[palette_dd.value]]
        fixed = process_video(inp, outname, fps_slider.value, offsets, bw, bh, pal, smooth_slider.value, proc_bar, fix_bar)
        print('✔ Ready:', fixed)
        display(FileLink(f'/content/{fixed}', result_html_prefix='Download: '))
        display(HTMLDisplay(f"<video width=600 controls><source src='/content/{fixed}' type='video/mp4'></video>"))

run_btn.on_click(on_run_click)

manager_ui = VBox([
    HBox([palette_dd, create_btn, name_txt, save_name_btn, delete_label, delete_btn]),
    HBox([picker, add_btn, remove_btn]),
    preview_out
])

# Layout
ui = VBox([
    file_dd,
    palette_label,
    fps_slider,
    HBox([r_slider, g_slider, b_slider]),
    block_w,
    smooth_slider,
    #Palette dropdown and manager UI inserted here,
    palette_dd,
    manager_ui,
    run_btn,
    proc_bar,
    fix_bar,
    video_log
])
display(ui)
