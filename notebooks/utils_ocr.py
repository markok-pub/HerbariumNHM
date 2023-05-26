import numpy as np
import cv2 as cv
import pytesseract
from pytesseract import Output
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from numpy import median


def get_ocr_in_json(img, tess_conf=50):
    #cfg_1 = "--oem 3 --psm 6"
    #cfg_2 = "--oem 1 --psm 12 -c textord_min_xheight=12"
    #data = pytesseract.image_to_data(img, output_type=Output.DICT, lang="deu", config=cfg_1)
    #n_boxes = len(data['level'])
    
    #json_structure = {
    #    "originalImage": {
    #        "width": img.shape[1],
    #        "height": img.shape[0]
    #    },
    #    "ocrTerms": []
    #}
    #for idx in range(0, int(n_boxes)):

        #if int(float(data['conf'][idx])) > tess_conf:
        #    json_structure["ocrTerms"].append({"term": data['text'][idx], "x": data['left'][idx], "y": data['top'][idx],
        #                                       "w": data['width'][idx], "h": data['height'][idx]})

    #return json_structure
    return ""


def show_ocr(img, json_structure):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    n_boxes = len(json_structure['ocrTerms'])
    xheights = []
    for idx in range(n_boxes):
        xheights.append(json_structure['ocrTerms'][idx]['h'])
    xheight = median(xheights)
    #xheight=40
    if xheight > 50 or xheight < 10:
        xheight = 40
    for idx in range(n_boxes):
        (x, y, w, h) = (json_structure['ocrTerms'][idx]['x'], json_structure['ocrTerms'][idx]['y'],
                        json_structure['ocrTerms'][idx]['w'], json_structure['ocrTerms'][idx]['h'])
        shape = [(x, y), (x + w, y + h)]
        #font = ImageFont.truetype("DejaVuSans-ExtraLight.ttf", size=h)
        font = ImageFont.truetype("DejaVuSans-ExtraLight.ttf", size=int(xheight))
        draw.rectangle(shape, outline="yellow", width=4)
        draw.text((x, y-xheight), json_structure['ocrTerms'][idx]['term'], font=font,
                  fill=(0, 0, 255, 0))
    return np.array(img_pil)


def print_ocr(json_structure):
    df = pd.DataFrame(json_structure['ocrTerms'])
    print(df.to_markdown())


