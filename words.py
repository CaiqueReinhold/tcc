import os
from os import path
import cPickle
import difflib

import numpy as np
import cv2
import theano

FILES_PATH = path.join('/', 'home', 'caique', 'words')
# PROCESSED = path.join(FILES_PATH, 'proc_test')
PROCESSED = path.join(FILES_PATH, 'processed')
if not path.isdir(PROCESSED):
    os.mkdir(PROCESSED)

CLASSES = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,?-'/\"*+ "


def parse_line(line):
    vals = line.split()
    vals[8] = ' '.join(vals[8:])
    return {
        'image': vals[0],
        'ok': vals[1] == 'ok',
        'gray': int(vals[2]),
        'text': vals[8]
    }


def CER(a, b):
	dif = difflib.ndiff(a, b)
	return sum([1 for d in dif if d[0] != ' ']) / float(len(a))


def stringify(indexes):
    return ''.join(map(lambda i: CLASSES[i], [i for i in indexes if i < len(CLASSES)]))


def indexify(text):
    return np.asarray(map(lambda c: CLASSES.index(c), text),
                      dtype=np.int32)


def load_img(img):
    img = cv2.imread(path.join(PROCESSED, img + '.png'))
    img = img[:,:,0]
    img = img / 255.0
    img = 1.0 - img
    img = img.astype(theano.config.floatX)
    return img


def get_data(index=0):
    SIZE = 27484
    # SIZE = 10
    with open(path.join(FILES_PATH, 'words.txt')) as file:
        file.seek(802) # skip header
        data = [parse_line(line) for line in file]
        data = [item for item in data if len(item['text']) > 1 and item['ok']]
        data = data[index*SIZE:(index+1)*SIZE]

    data_y = [indexify(item['text']) for item in data]
    data_x = [load_img(item['image']) for item in data]

    for i, img in enumerate(data_x):
        if img.shape[1] <= 20:
            print data[i]['text'], data[i]['image']
            cv2.imshow('img', img)
            cv2.waitKey(0)

    return data_x, data_y


def pre_process_img(img_info):
    n = img_info['image'].split('-')
    imgpath = '{}/{}-{}/{}.png'.format(n[0], n[0], n[1], img_info['image'])
    img = cv2.imread(path.join(FILES_PATH, imgpath))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, img = cv2.threshold(img, img_info['gray'], 255, cv2.THRESH_BINARY)

    # pad height
    MAX_HEIGHT = 160
    if img.shape[0] < MAX_HEIGHT:
        pad = np.zeros(((MAX_HEIGHT - img.shape[0]) / 2, img.shape[1]))
        pad.fill(255)
        img = np.vstack((pad, img))
        pad = np.zeros((MAX_HEIGHT - img.shape[0], img.shape[1]))
        pad.fill(255)
        img = np.vstack((img, pad))
    elif img.shape[0] > MAX_HEIGHT:
        img = cv2.resize(img, (MAX_HEIGHT * img.shape[1] / img.shape[0], MAX_HEIGHT),
                         interpolation=cv2.INTER_AREA)

    # pad width
    WINDOW_SIZE = 20
    if img.shape[1] % WINDOW_SIZE != 0:
        pad = np.zeros((MAX_HEIGHT, img.shape[1] % WINDOW_SIZE))
        pad.fill(255)
        img = np.hstack((img, pad))

    cv2.imwrite(path.join(PROCESSED, img_info['image'] + '.png'), img)
    # return img.shape[0]


def main():
    with open(path.join(FILES_PATH, 'words.txt')) as file:
        file.seek(802) # skip header
        data = [parse_line(line) for line in file]
        data = [item for item in data if len(item['text']) > 1 and not item['ok']]
        
        test_data = []
        da = os.listdir(PROCESSED)
        da = [file.split('.')[0] for file in da]

        for item in data:
        	if item['image'] in da:
        		test_data.append(item)

        f = open('test.pkl', 'wb')
        cPickle.dump(test_data, f)
        f.close()


if __name__ == '__main__':
    main()
