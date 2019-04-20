#!/usr/bin/env python
"""PyTimeLapse.py: processes several photos to a timelapse video. The deshake functionality is heavily based on https://github.com/pmoret/deshake"""

__author__ = "M. Hadi Timachi"
__license__ = "GPLv2"

import os
import concurrent.futures
import warnings

from PIL import Image, ImageDraw, ImageFont
from skimage.feature import register_translation
from skimage import transform, io
from skimage.util import img_as_ubyte, img_as_uint
from skimage.color import rgb2gray
from skimage.filters import sobel
import argparse
import tqdm



class Frame:
    def __init__(self, location, output='output',
                 resize=False, width=None, height=None, keep_ratio=False,
                 watermark=False, text='', font_size=40, watermark_location='', position=None, color='black', stroke=True, stroke_color='white',
                 deshake=False, **kwargs):
        self._location = location
        self._save_location = os.path.join(os.path.dirname(self._location), output, os.path.basename(self._location))
    
    def load(self):
        self._img = Image.open(self._location)
        self._size = self._img.size

    def resize(self, width, height, keep_ratio=True):
        if keep_ratio:
            self._img.thumbnail((width, height))
        else:
            self._img = self._img.resize((width, height))
        self._size = self._img.size
    
    def find_shift(self, other):
        """ Find a translation between two images. Other is the reference image. """
        self.shift, self.error, self.diffphase = register_translation(sobel(rgb2gray(io.imread(other._location))), sobel(rgb2gray(io.imread(self._location))), 100)

    def watermark_with_image(self, _watermark_image, position=(0, 0)):
        self._img.paste(_watermark_image, position, mask=_watermark_image)

    def watermark_with_text(self, text, font_size, position=(0, 0), fill='black', stroke=True, stroke_color='white'):
        drawing = ImageDraw.Draw(self._img)
        font = ImageFont.truetype("DroidSerif-Bold.ttf", font_size)
        if not position:
            w, h = drawing.textsize(text, font=font)   
            position = (self._size[0]/2-w/2, self._size[1]-2*h)
        if stroke:    
            drawing.text((position[0]+.5,position[1]-.5), text, fill=stroke_color, font=font)
            drawing.text((position[0]-.5,position[1]-.5), text, fill=stroke_color, font=font)
            drawing.text((position[0]-.5,position[1]+.5), text, fill=stroke_color, font=font)
            drawing.text((position[0]+.5,position[1]+.5), text, fill=stroke_color, font=font)
        drawing.text(position, text, fill=fill, font=font)

    def show_preview(self):
        self._img.show()

    def save(self, output):
        if os.path.exists(self._save_location): os.remove(self._save_location)
        self._img.save(self._save_location)
    
    def ndarray_to_pil(self, res):
        """ Converts from scikit-image ndarray to pil"""
        arr = img_as_ubyte(res)
        mode = {3: 'RGB', 4: 'RGBA'}[arr.shape[2]]
        try:
            array_buffer = arr.tobytes()
        except AttributeError:
            array_buffer = arr.tostring() # Numpy < 1.9
        image_shape = (arr.shape[1], arr.shape[0])
        try:
            self._img = Image.frombytes(mode, image_shape, array_buffer)
        except AttributeError:
            self._img = Image.fromstring(mode, image_shape, array_buffer) # PIL 1.1.7
        self._size = self._img.size

    def correct(self, crop=None):
        """ Apply the correction and crops based on the shift. """
        (y0, y1, x0, x1) = crop
        (shift_y, shift_x) = self.shift
        tf_shift = transform.SimilarityTransform(translation=[-shift_x, -shift_y])
        img = transform.warp(self._img, tf_shift)
        # crop
        res = img[-y0:-y1, -x0:-x1, :]
        self.ndarray_to_pil(res)
        


def return_files(input_dir, pattern='.jpg'):
    """ Returns photos based on an extension pattern. """
    if os.path.exists(input_dir):
        try:
            files = glob.glob(os.path.dirname(input_dir)+'*'+pattern)
        except Exception as e:
            print('There was an error in getting photos', e)
        files.sort()
        return files


if __name__ == "__main__":
    # Parse arguments sent by command
    parser = argparse.ArgumentParser(description='Make a timelapse video out of several photos.')
    parser.add_argument('-i', '--input', type=str, metavar='', required=True, help='Source directory including photos')
    parser.add_argument('-o', '--output', type=str, metavar='', help='Destination directory to save processed photos and the video (relative to input).')
    parser.add_argument('-r', '--resize', action='store_true', help='To resize the photos to w and h with and without keeping ratio')
    parser.add_argument('-k', '--keep_ratio', action='store_true', help='keep ratio')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
    parser.add_argument('--width', type=int, metavar='', help='Width to be resized to')
    parser.add_argument('--height', type=int, metavar='', help='Height to be resized to')  
    parser.add_argument('-w', '--watermark', action='store_true', help='To watermark photos with text or image')
    parser.add_argument('-wl', '--watermark_location', action='store_true', help='The location of the image for image watermark')
    parser.add_argument('-t', '--text', type=str, metavar='', help='Text for warermark')
    parser.add_argument('-p', '--position', type=str, metavar='', help='Position for text or image warermark, default: (0, 0)')
    parser.add_argument('-f', '--font_size', type=int, metavar='', default=40, help='Font size for warermark, default: 40')
    parser.add_argument('-c', '--color', type=str, metavar='', default='gray', help='Fill color for text warermark, default: gray')
    parser.add_argument('-s', '--stroke', action='store_true', help='Add a stroke around the watermark text.')
    parser.add_argument('-sc', '--stroke_color', type=str, metavar='', default='white', help='Stroke fill color for text warermark, default: white')
    parser.add_argument('-d', '--deshake', action='store_true', help='Deshake photos')
    args = parser.parse_args()

    kw = vars(args)
    
    
    def init_worker(file):
        frame = Frame(file, **kw)
        if kw['deshake']:
            frame.find_shift(base_frame)
        return frame

    def process_worker(frame):
        frame.load()
        if kw['deshake']: frame.correct(crop = crop)
        if kw['resize']: frame.resize(kw['width'], kw['height'], keep_ratio=kw['keep_ratio'])
        if kw['watermark']:
            if kw['watermark_location'] and kw['text']:
                print(
                    'you passed both an image and a text for watermark. I\'ll take the former')
                kw['text'] = ''
        if kw['watermark_location']:
            watermark_image = Image.open(kw['watermark_location'])
            frame.watermark_with_image(watermark_image, position=kw['position'])
        elif kw['text']:
            frame.watermark_with_text(
                    text=kw['text'], font_size=kw['font_size'], position=kw['position'], fill=kw['color'], stroke=kw['stroke'], stroke_color=kw['stroke_color'])
        if not os.path.isdir(os.path.join(os.path.dirname(frame._location), kw['output'])):
            os.mkdir(os.path.join(os.path.dirname(frame._location), kw['output']))
        frame.save(kw['output'])
        del frame
        return True

    if kw['verbose']:
        print('verbose on')
    else:
        warnings.filterwarnings("ignore")
    
    files = return_files(kw['input'])
    base_frame = Frame(files[0], **kw)
    print('Initializing files...')
    # result = (init_worker(file) for file in files)
    with concurrent.futures.ProcessPoolExecutor() as pool:
        result = list(tqdm.tqdm(pool.map(init_worker, files), total = len(files)))
    print('Done initializing files')
    try:
        y_shift = list(map(lambda x: x[0], [res.shift for res in result]))
        x_shift = list(map(lambda x: x[1], [res.shift for res in result]))
        crop = [int(min(y_shift)) - 1, int(max(y_shift)) + 1,
                int(min(x_shift)) - 1, int(max(x_shift)) + 1]
    except: pass
    print('Processing files...')
    # result = [process_worker(file) for file in result]
    with concurrent.futures.ProcessPoolExecutor() as pool:
        results = list(tqdm.tqdm(pool.map(process_worker, result), total = len(files)))
    print('All done!')