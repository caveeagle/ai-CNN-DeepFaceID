import cv2
from ultralytics import YOLO

model = YOLO('yolo.v8.nano-face.pt')

test_file = './images/Cave-002.png'

img = cv2.imread(test_file)
if img is None:
    raise FileNotFoundError('File image.png not found')

# 3. Детекция
results = model(img)

if(1):

    from PIL import Image, ImageDraw  # Pillow library
    
    # 4. Отрисовка рамок
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
    
            # рамка лица
            draw.rectangle(
                [(x1, y1), (x2, y2)],
                outline=(0, 255, 0),
                width=2
            )
    
    # 5. Показ результата
    img_pil.show()  

print('Task finished')    
