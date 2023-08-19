#   resizes image files, and retains IPTC/EXIF data. Capitalizes keywords, ignores stopwords. Can choose to disregard alphabetizing the IPTC keywords.
import os
from PIL import Image,ImageOps,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from iptcinfo3 import IPTCInfo
from unidecode import unidecode
import codecs
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
sw = set(stopwords.words("english"))
import numpy as np


IMG_TYPES = [
            'blp', 'bmp', 'dib', 'bufr', 'cur', 'pcx', 'dcx', 'dds', 'ps', 'eps'
            , 'fit', 'fits', 'fli', 'flc', 'ftc', 'ftu', 'gbr', 'gif', 'grib', 'h5'
            , 'hdf', 'png', 'apng', 'jp2', 'j2k', 'jpc', 'jpf', 'jpx', 'j2c', 'icns'
            , 'ico', 'im', 'iim', 'tif', 'tiff', 'jfif', 'jpe', 'jpg', 'jpeg', 'mpg'
            , 'mpeg', 'mpo', 'msp', 'palm', 'pcd', 'pxr', 'pbm', 'pgm', 'ppm', 'pnm'
            , 'psd', 'bw', 'rgb', 'rgba', 'sgi', 'ras', 'tga', 'icb', 'vda', 'vst'
            , 'webp', 'wmf', 'emf', 'xbm', 'xpm','nef'
            ]

FILE_WALK_WITH_ME = False
CONTAIN = False
CONTAIN_W_H = (2048,2048)
IMG_SAVE_TYPE = "JPEG"
ALPHABETIZE_KEYWORDS = False

def img_list()->list:
    print("File Path: ",end="")
    fp = input()
    fp = fp.replace(chr(92),chr(47)).replace(chr(34),'')
    if fp[-1:] != chr(47):
        fp = fp+chr(47)
    print("\n",fp,"\n")
    if FILE_WALK_WITH_ME:
        file_list=[]
        for r, d, f in os.walk(fp[:-1:]):
            for file in f:
                if (file[-(file[::-1].find('.')):] in IMG_TYPES):
                    file_list.append(os.path.join(r, file))
        return file_list
    elif not FILE_WALK_WITH_ME:
        return [
                fp+f for f 
                in os.listdir(fp[:-1:]) 
                if os.path.isfile(fp+f) 
                and f[-(f[::-1].find('.')):].lower() in IMG_TYPES]
 

def img_contain(file:str)->None:
    tag_list=[]
    f_name = file[-(file[::-1].find('/')):].lower()
    iptc_info = IPTCInfo(file, force=True)
    tags = [codecs.decode(x,encoding='utf-8').strip().lower() for x in iptc_info['keywords']]
    l_w = lambda x: str(x).capitalize() if not set(x).issubset(sw) else str(x).capitalize()
    if ALPHABETIZE_KEYWORDS:
        txt_write =str(
                        ', '.join(
                            t for t 
                            in [' '.join(l_w(s) for s 
                            in str(w).split(chr(32))) 
                            for w in np.unique(np.array(tags)).tolist()])
                        )
    elif not ALPHABETIZE_KEYWORDS:
        txt_write =str(
                        ', '.join(
                            t for t 
                            in [' '.join(l_w(s) for s 
                            in str(w).split(chr(32))) for w 
                            in np.array([y[np.argsort(z)].astype('str').flatten().tolist() for y,z 
                            in [np.unique(np.array(tags),return_index=True)]])[0]]
                            )
                        )
    img = Image.open(file)
    img = img.convert("RGB")
    exif = img.getexif()
    if CONTAIN:
        img = ImageOps.contain(
                                img, 
                                CONTAIN_W_H,
                                method=Image.Resampling.BICUBIC
                                )
    img.save(
            file,
            IMG_SAVE_TYPE,
            quality=100,
            exif=exif)
    iptc_info = IPTCInfo(file, force=True)
    iptc_info['keywords'] = tag_list
    iptc_info.save()  


if __name__ == "__main__":
    file_list = img_list()
    l_bar='{desc}: {percentage:3.0f}%|'
    r_bar='| {n_fmt}/{total_fmt} [elapsed: {elapsed} / Remaining: {remaining}] '
    bar = '{rate_fmt}{postfix}]'
    status_bar = tqdm(total=len(file_list), desc=r'Image Contain',bar_format=f'{l_bar}{bar}{r_bar}')
    with ThreadPoolExecutor(8) as executor:
        futures = [
            executor.submit(img_contain,file) for file in file_list]
        for _ in as_completed(futures):
            status_bar.update(n=1)
        status_bar.close()
