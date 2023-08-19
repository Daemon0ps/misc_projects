#random attempt #12 to use TF-IDF to process pixels as hex "words" to sort by color palette similarity
import numpy as np
from tqdm import tqdm
import pandas as pd
from traceback_with_variables import activate_by_import
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import cv2
import numpy as np
import string
import shutil
import hashlib
from random import randint
from PIL import Image,ImageOps,ImageFile
from PIL.ExifTags import TAGS
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.spatial import cKDTree

file_path = ""
save_to_path = ""
print("File Path:",end="")
file_path = input()
print("\n",file_path,"\n")
print("Save Path:",end="")
save_to_path = input()
print("\n",save_to_path,"\n")
file_path = file_path.replace(chr(92),"/")
if file_path[-1:] != "/":
    file_path = file_path+"/"
save_to_path = save_to_path.replace(chr(92),"/")
if save_to_path[-1:] != "/":
    save_to_path = save_to_path+"/"

RGB_PALETTE = {
        "3c":np.array([[255,255,255],[128,128,128],[0,0,0]]),
        "html":np.array([[255,160,122],[250,128,114],[233,150,122],
                         [240,128,128],[205,92,92],[220,20,60],
                         [178,34,34],[255,0,0],[139,0,0],
                         [255,127,80],[255,99,71],[255,69,0],
                         [255,215,0],[255,165,0],[255,140,0],
                         [255,255,224],[255,250,205],[250,250,210],
                         [255,239,213],[255,228,181],[255,218,185],
                         [238,232,170],[240,230,140],[189,183,107],
                         [255,255,0],[124,252,0],[127,255,0],
                         [50,205,50],[0,255,0],[34,139,34],
                         [0,128,0],[0,100,0],[173,255,47],
                         [154,205,50],[0,255,127],[0,250,154],
                         [144,238,144],[152,251,152],[143,188,143],
                         [60,179,113],[46,139,87],[128,128,0],
                         [85,107,47],[107,142,35],[224,255,255],
                         [0,255,255],[0,255,255],[127,255,212],
                         [102,205,170],[175,238,238],[64,224,208],
                         [72,209,204],[0,206,209],[32,178,170],
                         [95,158,160],[0,139,139],[0,128,128],
                         [176,224,230],[173,216,230],[135,206,250],
                         [135,206,235],[0,191,255],[176,196,222],
                         [30,144,255],[100,149,237],[70,130,180],
                         [65,105,225],[0,0,255],[0,0,205],
                         [0,0,139],[0,0,128],[25,25,112],
                         [123,104,238],[106,90,205],[72,61,139],
                         [230,230,250],[216,191,216],[221,160,221],
                         [238,130,238],[218,112,214],[255,0,255],
                         [255,0,255],[186,85,211],[147,112,219],
                         [138,43,226],[148,0,211],[153,50,204],
                         [139,0,139],[128,0,128],[75,0,130],
                         [255,192,203],[255,182,193],[255,105,180],
                         [255,20,147],[219,112,147],[199,21,133],
                         [255,255,255],[255,250,250],[240,255,240],
                         [245,255,250],[240,255,255],[240,248,255],
                         [248,248,255],[245,245,245],[255,245,238],
                         [245,245,220],[253,245,230],[255,250,240],
                         [255,255,240],[250,235,215],[250,240,230],
                         [255,240,245],[255,228,225],[220,220,220],
                         [211,211,211],[192,192,192],[169,169,169],
                         [128,128,128],[105,105,105],[119,136,153],
                         [112,128,144],[47,79,79],[0,0,0],
                         [255,248,220],[255,235,205],[255,228,196],
                         [255,222,173],[245,222,179],[222,184,135],
                         [210,180,140],[188,143,143],[244,164,96],
                         [218,165,32],[205,133,63],[210,105,30],
                         [139,69,19],[160,82,45],[165,42,42],
                         [128,0,0]]),
        "basic":np.array([[0,0,0],[255,255,255],[255,0,0],
                          [0,255,0],[0,0,255],[255,255,0],
                          [0,255,255],[255,0,255],[192,192,192],
                          [128,128,128],[128,0,0],[128,128,0],
                          [0,128,0],[128,0,128],[0,128,128],[0,0,128]]),
        "red":np.array([[255,160,122],[250,128,114],[233,150,122],
                        [240,128,128],[205,92,92],[220,20,60],
                        [178,34,34],[255,0,0],[139,0,0]]),
        "orange":np.array([[255,127,80],[255,99,71],[255,69,0],
                            [255,215,0],[255,165,0],[255,140,0]]),
        "yellow":np.array([[255,255,224],[255,250,205],[250,250,210],
                            [255,239,213],[255,228,181],[255,218,185],
                            [238,232,170],[240,230,140],[189,183,107],
                            [255,255,0]]),
        "green":np.array([[124,252,0],[127,255,0],[50,205,50],
                           [0,255,0],[34,139,34],[0,128,0],
                           [0,100,0],[173,255,47],[154,205,50],
                           [0,255,127],[0,250,154],[144,238,144],
                           [152,251,152],[143,188,143],[60,179,113],
                           [46,139,87],[128,128,0],[85,107,47],
                           [107,142,35]]),
        "teal":np.array([[224,255,255],[0,255,255],[0,255,255],
                          [127,255,212],[102,205,170],[175,238,238],
                          [64,224,208],[72,209,204],[0,206,209],
                          [32,178,170],[95,158,160],[0,139,139],
                          [0,128,128]]),
        "blue":np.array([[176,224,230],[173,216,230],[135,206,250],
                          [135,206,235],[0,191,255],[176,196,222],
                          [30,144,255],[100,149,237],[70,130,180],
                          [65,105,225],[0,0,255],[0,0,205],[0,0,139],
                          [0,0,128],[25,25,112],[123,104,238],
                          [106,90,205],[72,61,139]]),
        "purple":np.array([[230,230,250],[216,191,216],[221,160,221],
                            [238,130,238],[218,112,214],[255,0,255],
                            [255,0,255],[186,85,211],[147,112,219],
                            [138,43,226],[148,0,211],[153,50,204],
                            [139,0,139],[128,0,128],[75,0,130]]),
        "pink":np.array([[255,192,203],[255,182,193],[255,105,180],
                          [255,20,147],[219,112,147],[199,21,133]]),
        "white":np.array([[255,255,255],[255,250,250],[240,255,240],
                           [245,255,250],[240,255,255],[240,248,255],
                           [248,248,255],[245,245,245],[255,245,238],
                           [245,245,220],[253,245,230],[255,250,240],
                           [255,255,240],[250,235,215],[250,240,230],
                           [255,240,245],[255,228,225]]),
        "gray":np.array([[220,220,220],[211,211,211],[192,192,192],
                          [169,169,169],[128,128,128],[105,105,105],
                          [119,136,153],[112,128,144],[47,79,79],[0,0,0]]),
        "brown":np.array([[255,248,220],[255,235,205],[255,228,196],
                           [255,222,173],[245,222,179],[222,184,135],
                           [210,180,140],[188,143,143],[244,164,96],
                           [218,165,32],[205,133,63],[210,105,30],
                           [139,69,19],[160,82,45],[165,42,42],
                           [128,0,0]]),
        "GOLD":np.array([[250,250,210],[238,232,170],[240,230,140],
                          [218,165,32],[255,215,0],[255,165,0],
                          [255,140,0],[205,133,63],[210,105,30],
                          [139,69,19],[160,82,45],[255,223,0],
                          [212,175,55],[207,181,59],[197,179,88],
                          [230,190,138],[153,101,21]])}

IMG_TYPES = [
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
            ,'nef'
            ]
def statbar(tot:int,desc:str):
    l_bar='{desc}: {percentage:3.0f}%|'
    r_bar='| {n_fmt}/{total_fmt} [elapsed: {elapsed} / Remaining: {remaining}] '
    bar = '{rate_fmt}{postfix}]'
    status_bar = tqdm(total=tot, desc=desc,bar_format=f'{l_bar}{bar}{r_bar}')
    return status_bar

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  res = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return res

def main(img_files, i_f):
    (w,h) = (128,128)
    img = cv2.imread(img_files[i_f])
    img = ImageOps.contain(Image.fromarray(img[:, :, ::-1],mode="RGB"),(128,128),Image.Resampling.LANCZOS)
    c_arr_list = [
        str(('{:02X}' * 3).format(r,g,b)) 
        for r,g,b in 
        np.array(RGB_PALETTE['3c'][cKDTree(RGB_PALETTE['3c']).query(img,k=1)[1]]).astype('uint8').reshape(-1,3).tolist()]
    c3 = ' '.join(c for c in [
        x[0] for x in sorted([[s,c_arr_list.count(s)] 
        for s in [ np.array(y[np.argsort(z)]).astype('str').tolist()
        for y,z in [np.unique(np.array(c_arr_list),return_index=True)]]],
        key=lambda x: x[1], reverse=False)][0][-3:])    
    c_arr_list = [
        str(('{:02X}' * 3).format(r,g,b)) 
        for r,g,b in 
        np.array(RGB_PALETTE['basic'][cKDTree(RGB_PALETTE['basic']).query(img,k=1)[1]]).astype('uint8').reshape(-1,3).tolist()]
    cc = ' '.join(c for c in [#FEED ME A STRAY CAT
        x[0] for x in sorted([[s,c_arr_list.count(s)] 
        for s in [ np.array(y[np.argsort(z)]).astype('str').tolist()
        for y,z in [np.unique(np.array(c_arr_list),return_index=True)]]],
        key=lambda x: x[1], reverse=False)][0][-3:])
    img = np.array(img).astype('uint8')[:, :, ::-1]
    img_p = ImageOps.contain(Image.fromarray(img,mode="RGB"),(w,h),Image.Resampling.LANCZOS)
    img = np.array(img_p).astype('uint8')[:, :, ::-1]
    img2 = cv2.addWeighted(img, 0.6 , cv2.flip(img,flipCode=1), 0.6, 0)
    img3 = cv2.addWeighted(img2, 0.6 , cv2.flip(img2,flipCode=-1), 0.6, 0)    
    img4 = cv2.addWeighted(rotate_image(img3, 90), 0.6, cv2.flip(img3,flipCode=1), 0.6, 0)
    img = cv2.addWeighted(img4, 0.6, img4, 0.6, 0)
    img = np.array(cv2.blur(img,(3,3)))
    norm_img = np.zeros(img.shape)
    norm_img = cv2.normalize(img,norm_img,32,224,cv2.NORM_MINMAX)
    img = norm_img
    c_arr_list = [
        str(('{:02X}' * 3).format(
        int(abs(float(r/65536))),
        int(abs(float(r+b/65536))),
        int(abs(float(b/65536))),
        int(abs(float(b+g/65536))),
        int(abs(float(g/65536)))))
        for r,g,b in np.array(RGB_PALETTE['html'][cKDTree(RGB_PALETTE['html']).query(img[:, :, ::-1],k=1)[1]]).astype('uint8').reshape(-1,3).tolist() if r+g+b!=0]
    pix_list = []
    pix_list = [" ".join(x for x in word_tokenize(str(' '.join( str(w) for w in [y for y in np.array_split(c_arr_list,w)]))))]
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(8,8))
    vectorizer.fit_transform(pix_list)
    tfidf_features = (vectorizer.get_feature_names_out())
    df = pd.DataFrame(tfidf_features)
    res,ind = np.unique(' '.join(x[0] for x in df.iloc[:32].astype('str').to_numpy().tolist()).split(chr(32)),return_index=True)
    tf_s = res[np.argsort(ind)]
    f_head = str(cc).replace('0','').replace(chr(32),'')+"_"+str(c3).replace('0','').replace(chr(32),'')+"_"+str(''.join(str(x).replace('0','').replace(chr(32),'') for x in tf_s)).translate(str.maketrans('', '', string.punctuation))[-128:]
    f_nn = str(save_to_path+f_head+str(i_f)+".jpg")
    os.rename(img_files[i_f],f_nn)

