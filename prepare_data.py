import os
from os import path
import cPickle

import cv2
import numpy as np
import theano


FILES_PATH = path.join('/', 'home', 'caique', 'Downloads', 'lines')
PROCESSED = path.join(FILES_PATH, 'processed')
if not path.isdir(PROCESSED):
    os.mkdir(PROCESSED)

CLASSES = "0123456789abcdefghijklmnopqrstuvwxyz.,?!*/-+'\"#():;& "


def parse_line(line):
    vals = line.split()
    vals[8] = ' '.join(vals[8:])
    return {
        'image': vals[0],
        'gray': int(vals[2]),
        'text': ' '.join(vals[8].split('|')).lower()
    }


def stringify(indexes):
    return ''.join(map(lambda i: CLASSES[i], [i for i in indexes if i < len(CLASSES)]))


def indexify(text):
    return np.asarray(map(lambda c: CLASSES.index(c), text),
                      dtype=np.int32)


def load_img(img):
    img = cv2.imread(path.join(PROCESSED, img + '.png'))
    img = img[:,:,0]
    img = img / 255.0
    img = 1 - img
    img = img.astype(theano.config.floatX)
    if img.shape[1] % 20 != 0:
        pad = np.zeros((100, img.shape[1] % 20), dtype=theano.config.floatX)
        img = np.hstack((img, pad))
    return img


def get_data(index=0):
    SIZE = 4451
    # SIZE = 100
    with open(path.join(FILES_PATH, 'lines.txt')) as file:
        file.seek(1025) # skip header
        data = [parse_line(line) for line in file]

    data_y = [indexify(item['text']) for item in data[index*SIZE:(index+1)*SIZE]]
    data_x = [load_img(item['image']) for item in data[index*SIZE:(index+1)*SIZE]]

    return data_x, data_y


def pre_process_img(img_info):
    n = img_info['image'].split('-')
    imgpath = '{}/{}-{}/{}.png'.format(n[0], n[0], n[1], img_info['image'])
    img = cv2.imread(path.join(FILES_PATH, imgpath))
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # _, img = cv2.threshold(img, img_info['gray'], 255, cv2.THRESH_BINARY)
    # img = cv2.resize(img, (100 * img.shape[1] / img.shape[0], 100),
    #                  interpolation=cv2.INTER_AREA)
    # cv2.imwrite(path.join(PROCESSED, img_info['image'] + '.png'), img)
    # img_info['image'] = img
    return img.shape[1]


def main():
    with open(path.join(FILES_PATH, 'lines.txt')) as file:
        file.seek(1025) # skip header
        data = [parse_line(line) for line in file]
        # for item in data:
        #     pre_process_img(item)
        print max([pre_process_img(item) for item in data])


if __name__ == '__main__':
    main()
