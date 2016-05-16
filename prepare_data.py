import os
from os import path

import cv2
import numpy as np


FILES_PATH = path.join('/', 'home', 'caique', 'Downloads', 'lines')
PROCESSED = path.join(FILES_PATH, 'processed')
if not path.isdir(PROCESSED):
    os.mkdir(PROCESSED)


def parse_line(line):
    vals = line.split()
    return {
        'image': vals[0],
        'ok': vals[1] == 'ok',
        'gray': int(vals[2]),
        'text': ' '.join(vals[8].split('|'))
    }


def pre_process_img(img_info):
    n = img_info['image'].split('-')
    imgpath = '{}/{}-{}/{}.png'.format(n[0], n[0], n[1], img_info['image'])
    img = cv2.imread(path.join(FILES_PATH, imgpath))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, img = cv2.threshold(img, img_info['gray'], 255, cv2.THRESH_BINARY)
    img = cv2.resize(img, (img.shape[1], 100), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path.join(PROCESSED, img_info['image'] + '.png'), img)
    # img_info['image'] = img


def main():
    with open(path.join(FILES_PATH, 'lines.txt')) as file:
        file.seek(1025) # skip header
        data = [parse_line(line) for line in file]
        for item in data:
            pre_process_img(item)


if __name__ == '__main__':
    main()
