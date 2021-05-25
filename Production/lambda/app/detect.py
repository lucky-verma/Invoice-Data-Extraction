import argparse
import time
from pathlib import Path
import pytesseract
import subprocess
from pdf2image import convert_from_path
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from PIL import Image

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
# import easyocr

# reader = easyocr.Reader(['en'])  # need to run only once to load model into memory

best_model = "model.pt"


def detect(image, weights):
    # Final Declaration
    final_det = None
    final_class = []
    final_text = []

    source, weights, save_txt, imgsz, ocr = image, weights, True, 1280, True
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    # save_dir = Path(increment_path(Path('runs/detect') / 'exp', exist_ok=False))  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device('')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.6, 0.45, classes=None, agnostic=False)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # img.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                print()
                print(img.shape[2:], "  ----------------------   ", im0.shape)
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                final_det = det[:, :4].tolist()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # python detect.py --source test\ --weights weights\model.pt --save-txt --save-conf

                final_class_reverse = []

                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (conf, cls, *xywh) if True else (cls, *xywh)  # label format
                        # print(names[int(cls)], str(line[0].squeeze().tolist()), 'box(xywh) = ' + str(line[2:]))
                        final_class_reverse.append(
                            names[int(cls)] + '(' + str(round((line[0].squeeze().tolist()), 2)) + ')')

                for i in final_class_reverse:
                    final_class.append(i)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            if ocr:

                # Opens a image in RGB mode
                im = Image.open(path)
                newlist = []

                for I in final_det:
                    # [27.0, 14.0, 177.0, 91.0]
                    left, top, right, bottom = I[0], I[1], I[2], I[3]
                    im1 = im.crop((left, top, right, bottom))
                    # imimimim = np.asarray(im1)
                    # text = reader.readtext(imimimim, detail=0)
                    # print(text)
                    text = pytesseract.image_to_string(im1)
                    # print(text)

                    # print(text.split('\n'))
                    newlist = text.split('\n')
                    newlist.pop(-1)

                    final_text.append(' '.join(map(str, newlist)))
                    # im1.show()

                print('LENGHTS', len(final_class), len(final_text))
                final_class.reverse()
                res = dict(zip(final_class, final_text))
                print("Resultant dictionary is : " + str(res))

    return res


def transferFiles(image):
    # pages = convert_from_path(pdf, 300)
    # pdf_file = pdf[:-4]
    #
    # for page in pages:
    #     page.save("%s-page%d.jpg" % (pdf_file, pages.index(page)), "JPEG")
    #
    # # image = convert_from_path(pdf)
    JSON = detect(image, weights=best_model)

    return JSON
