from os import path

import numpy as np
import matplotlib.pyplot as plot
from matplotlib import image as mpimg

FILES_PATH = 'C:\\Users\\caiqu\\Downloads\\lines'


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
	imgpath = '{}\\{}-{}\\{}.png'.format(n[0], n[0], n[1], img_info['image'])
	img = mpimg.imread(path.join(FILES_PATH, imgpath))
	img_info['image'] = img


def main():
	with open(path.join(FILES_PATH, 'lines.txt')) as file:
		file.seek(1025) # skip header
		data = [parse_line(line) for line in file]
		for item in data[-10:]:
			pre_process_img(item)
			plot.imshow(item['image'])
			plot.show()


if __name__ == '__main__':
	main()
