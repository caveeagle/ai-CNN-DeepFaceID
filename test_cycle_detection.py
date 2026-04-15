import os
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
from ultralytics import YOLO

#################################

MODEL = 'yolo.v8.nano-face.pt'
model = YOLO(MODEL)

#################################

image_files = sorted([
    f for f in os.listdir('images')
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])

if not image_files:
    raise RuntimeError('No images')

#################################

root = tk.Tk()
root.title('Face detection')
label = tk.Label(root)
label.pack()

index = [0]  # list to allow mutation inside nested functions

def show(i):
    filename = image_files[i]
    root.title(f'Face detection - {filename} ({i + 1}/{len(image_files)})')

    img = Image.open(os.path.join('images', filename)).convert('RGB')
    draw = ImageDraw.Draw(img)

    results = model(img)
    faces_detected = False

    for r in results:
        for box in r.boxes:
            faces_detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=2)

    if not faces_detected:
        draw.text((30, 50), 'No detected', fill=(255, 0, 0))

    draw.text((30, img.height - 40), filename, fill=(255, 255, 255))

    root.geometry(f'{img.width}x{img.height}')
    label.tk_img = ImageTk.PhotoImage(img)
    label.config(image=label.tk_img)

def on_next(event=None):
    index[0] += 1
    if index[0] >= len(image_files):
        root.destroy()
        return
    show(index[0])

def on_quit(event=None):
    root.destroy()

root.bind('<space>', on_next)
root.bind('<Return>', on_next)
root.bind('<Escape>', on_quit)

show(0)
root.mainloop()

#################################

print('Job finished')
