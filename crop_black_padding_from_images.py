import numpy as np
from tqdm import tqdm
from traceback_with_variables import activate_by_import
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import cv2
import string
import shutil
import hashlib
from random import randint
from PIL import Image,ImageOps,ImageFile
from PIL.ExifTags import TAGS
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import pandas as pd
from matplotlib import pyplot as plt

fp = str("")

IMG_TYPES = list([
            'blp', 'bmp', 'dib', 'bufr', 'cur'
            , 'pcx', 'dcx', 'dds', 'ps', 'eps'
            , 'fit', 'fits', 'fli', 'flc', 'ftc'
            , 'ftu', 'gbr', 'gif', 'grib', 'h5'
            , 'hdf', 'png', 'apng', 'jp2', 'j2k'
            , 'jpc', 'jpf', 'jpx', 'j2c', 'icns'
            , 'ico', 'im', 'iim', 'tif', 'tiff'
            , 'jfif', 'jpe', 'jpg', 'jpeg', 'mpg'
            , 'mpeg', 'mpo', 'msp', 'palm', 'pcd'
            , 'pxr', 'pbm', 'pgm', 'ppm', 'pnm'
            , 'psd', 'bw', 'rgb', 'rgba', 'sgi'
            , 'ras', 'tga', 'icb', 'vda', 'vst'
            , 'webp', 'wmf', 'emf', 'xbm', 'xpm'
            ,'nef'])

def tqdm_statbar(tot:int,desc:str)->tqdm:
    l_bar='{desc}: {percentage:3.0f}%|'
    r_bar='| {n_fmt}/{total_fmt} [elapsed: {elapsed} / Remaining: {remaining}] '
    bar = '{rate_fmt}{postfix}]'
    status_bar = tqdm(total=tot, desc=desc,bar_format=f'{l_bar}{bar}{r_bar}')
    return status_bar

def f_split(f:str)->list: #  0: basedir / 1: filename no ext / 2: ext
    return list(
        [f[:len(f)-(f[::-1].find('/')):].lower(),
         f[len(f)-(f[::-1].find('/')):(len(f))-1-len(f[-(f[::-1].find('.')):])],
         f[-(f[::-1].find('.')):].lower()])

def proc_fp()->str:
    print("File Path:",end="")
    fp = input()
    fp = fp.replace(chr(92),chr(47)).replace(chr(34),'')
    if fp[-1:] != "/":
        fp = fp+"/"
    print("\n",fp,"\n")
    return fp

def proc_img_list(fp:str)->list:
    return [fp+f for f in os.listdir(fp[:-1:]) if os.path.isfile(fp+f) and f[-(f[::-1].find('.')):].lower() in IMG_TYPES]

def md5_rename(file:str)->None:
    i_f = f_split(file)
    with open(file,"rb") as fi:
        file_bytes = fi.read()
    md5_calc = str(hashlib.md5(file_bytes).hexdigest())
    os.rename(file,f"{i_f[0]+'faces'}/{md5_calc}.{i_f[2]}")
    
def crop_image_only_outside(img:np.array,tol:int=0)->list:
    mask = img>tol
    if img.ndim==3:
        mask = mask.all(3)
    m,n = mask.shape
    mask0,mask1 = mask.any(0),mask.any(1)
    col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
    row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()
    return [row_start,row_end,col_start,col_end]

def proc_img(img_file:str)->None:
    image = cv2.imread(img_file)
    gray = np.array(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
    ci = crop_image_only_outside(gray,tol=40)
    c_image=image[ci[0]:ci[1],ci[2]:ci[3]]
    cv2.imwrite(img_file,c_image)


if __name__ == "__main__":
    fp = proc_fp()
    img_list = proc_img_list(fp)
    statbar = tqdm_statbar(len(img_list),"Image Crop")
    with ThreadPoolExecutor(8) as executor:
        futures = [executor.submit(proc_img,file) for file in img_list]
        for _ in as_completed(futures):
            statbar.update(n=1)
        statbar.close()    
