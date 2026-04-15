
if(1):
    import cv2
    from ultralytics import YOLO
    
    model = YOLO('yolo.v8.nano-face.pt')
    
    print('yolo.v8.nano-face - ok')

#######################################

if(1):
    from facenet_pytorch import InceptionResnetV1
    import torch
    
    model = InceptionResnetV1(pretrained='vggface2').eval()
    print('vggface2 - OK')

#######################################

print('\nScript finished')

