import cv2
import numpy as np
from PIL import Image,ImageFile,ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from traceback_with_variables import activate_by_import
import os,sys
import shutil
import hashlib
import re
from tqdm import tqdm
from dataclasses import dataclass

@dataclass
class ROI:
    file_path = "./"
    save_to_path = "./"
    avg_blur = 1
    Invert_Binary = 1
    Thresh_OTSU = 0
    Color_Binary = 0
    col_thresh_LR = 0
    col_thresh_LG = 0
    col_thresh_LB = 115
    col_thresh_UR = 255
    col_thresh_UG = 178
    col_thresh_UB = 255
    Cont_Method = 1
    dilate_iterations = 9
    MorphRect_X = 3
    MorphRect_Y = 3
    img_erode = 1
    erode_iter = 1
    thresh_floor = 36
    thresh_ceil = 101
    Cnt_R = 0
    Cnt_G = 0
    Cnt_B = 255
    rect_R = 128
    rect_G = 255
    rect_B = 128
    hMin = 0
    sMin = 0
    vMin = 0
    hMax = 255
    sMax = 255
    vMax = 255
    lower:tuple = (hMin,sMin,vMin)
    upper:tuple = (hMax,sMax,vMax)
    Norm_Floor = 0
    Norm_Ceil = 255
    img_types = [
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
    def __post_init__(self):
        self.Color_Binary = ROI.Color_Binary
        self.col_thresh_LR = ROI.col_thresh_LR
        self.col_thresh_LG = ROI.col_thresh_LG
        self.col_thresh_LB = ROI.col_thresh_LG
        self.col_thresh_UR = ROI.col_thresh_UR
        self.col_thresh_UG = ROI.col_thresh_UG
        self.col_thresh_UB = ROI.col_thresh_UB
        self.Cont_Method = ROI.Cont_Method
        self.Invert_Binary = ROI.Invert_Binary
        self.Thresh_OTSU = ROI.Thresh_OTSU
        self.dilate_iterations = ROI.dilate_iterations
        self.MorphRect_X = ROI.MorphRect_X
        self.MorphRect_Y = ROI.MorphRect_Y
        self.img_erode = ROI.img_erode
        self.erode_iter = ROI.erode_iter
        self.Cnt_R = ROI.Cnt_R
        self.Cnt_G = ROI.Cnt_G
        self.Cnt_B = ROI.Cnt_B
        self.rect_R = ROI.rect_R
        self.rect_G = ROI.rect_G
        self.rect_B = ROI.rect_B
        self.hMin = ROI.hMin
        self.sMin = ROI.sMin
        self.vMin = ROI.vMin
        self.hMax = ROI.hMax
        self.sMax = ROI.sMax
        self.vMax = ROI.vMax
        self.lower = ROI.lower
        self.upper = ROI.upper
        self.file_path = ROI.file_path
        self.save_to_path = ROI.save_to_path
        self.img_files = ROI.img_files
        self.avg_blur = ROI.avg_blur
        self.thresh_floor = ROI.thresh_floor
        self.thresh_ceil = ROI.thresh_ceil
        self.ex_f = ROI.ex_f
        self.img_types = ROI.img_types
        self.Norm_Floor = ROI.Norm_Floor
        self.Norm_Ceil = ROI.Norm_Ceil
        super().__setattr__('attr_name', self)


@staticmethod
def nothing(x):
    pass

def statbar(tot:int,desc:str):
        l_bar='{desc}: {percentage:3.0f}%|'
        r_bar='| {n_fmt}/{total_fmt} [elapsed: {elapsed} / Remaining: {remaining}] '
        bar = '{rate_fmt}{postfix}]'
        status_bar = tqdm(total=tot, desc=desc,bar_format=f'{l_bar}{bar}{r_bar}')
        return status_bar

img_files = [f for f in os.listdir(ROI.file_path[:-1:]) if os.path.isfile(ROI.file_path+f) and f[-(f[::-1].find('.')):] in ROI.img_types]

def sprite_extract_roi(ex_f,cnts,cv_img,):
    with open(ROI.file_path+ex_f,"rb") as fi:
        file_bytes = fi.read()
    md5_calc = str(hashlib.md5(file_bytes).hexdigest())
    f_name = ex_f.replace(ex_f[-((ex_f[::-1].find('.'))+1):],"")
    f_name = re.sub(r'[^A-Za-z0-9_-]+/g','', str(f_name).replace(chr(32),chr(95)))
    f_ext = ex_f[-(ex_f[::-1].find('.')):]
    cnt_dir = ROI.save_to_path+md5_calc+"/"
    cnt_dir_16 = ROI.save_to_path+md5_calc+"/16x16/"
    cnt_dir_32 = ROI.save_to_path+md5_calc+"/32x32/"
    cnt_name = cnt_dir+f_name
    if not os.path.isdir(cnt_dir[:1:]):
        os.makedirs(cnt_dir,exist_ok=True)
        os.makedirs(cnt_dir_16,exist_ok=True)
        os.makedirs(cnt_dir_32,exist_ok=True)
    imgBlank = np.zeros_like(cv_img)
    cnts_len = len(cnts)
    status_bar = statbar(cnts_len,"Extracting Contours")
    with open(cnt_dir+f_name+".txt",'wt') as fi:
        fi.write(f"Cont_Method = {ROI.Cont_Method}\n")
        fi.write(f"Invert_Binary = {ROI.Invert_Binary}\n")
        fi.write(f"dilate_iterations = {ROI.dilate_iterations}\n")
        fi.write(f"ROI.MorphRect_X = {ROI.MorphRect_X}\n")
        fi.write(f"ROI.MorphRect_Y = {ROI.MorphRect_Y}\n")
        fi.write(f"img_erode = {ROI.img_erode}\n")
        fi.write(f"erode_iter = {ROI.erode_iter}\n")
        fi.write(f"Cnt_R = {ROI.Cnt_R}\n")
        fi.write(f"Cnt_G = {ROI.Cnt_G}\n")
        fi.write(f"Cnt_B = {ROI.Cnt_B}\n")
        fi.write(f"rect_R = {ROI.rect_R}\n")
        fi.write(f"rect_G = {ROI.rect_G}\n")
        fi.write(f"rect_B = {ROI.rect_B}\n")
        fi.write(f"hMin = {ROI.hMin}\n")
        fi.write(f"sMin = {ROI.sMin}\n")
        fi.write(f"vMin = {ROI.vMin}\n")
        fi.write(f"hMax = {ROI.hMax}\n")
        fi.write(f"sMax = {ROI.sMax}\n")
        fi.write(f"vMax = {ROI.vMax}\n")
        fi.write(f"lower = {ROI.lower}\n")
        fi.write(f"upper = {ROI.upper}\n")
        fi.write(f"avg_blur = {ROI.avg_blur}\n")
        fi.write(f"thresh_floor = {ROI.thresh_floor}\n")
        fi.write(f"thresh_ceil = {ROI.thresh_ceil}\n")
        fi.write(f"img_types = {ROI.img_types}\n")
        fi.write(f"Norm_Floor = {ROI.Norm_Floor}\n")
        fi.write(f"Norm_Ceil = {ROI.Norm_Ceil}\n")
    for i, cnt in enumerate(cnts):
        cnt_name = md5_calc+str(i).zfill(4)+".png"
        cnt_name_16 = md5_calc+str(i).zfill(4)+"_16.png"
        cnt_name_16_512 = md5_calc+str(i).zfill(4)+"_16_512.png"
        cnt_name_32 = md5_calc+str(i).zfill(4)+"_32.png"
        cnt_name_32_512 = md5_calc+str(i).zfill(4)+"_32_512.png"
        mask = cv2.cvtColor(imgBlank.copy(), cv2.COLOR_BGRA2GRAY)
        cv2.drawContours(mask, cnts, i, 255, -1)
        cnt_img = imgBlank.copy()
        cnt_img[mask == 255] = cv_img[mask == 255]
        bRect = cv2.boundingRect(cnt)
        x, y, w, h = bRect
        cnt_crop = cnt_img[y:y+h, x:x+w]
        interp = cv2.INTER_LANCZOS4
        cnt_crop_16 = cv2.resize(cnt_crop, (16,16), interpolation=interp)
        cnt_crop_32 = cv2.resize(cnt_crop, (32,32), interpolation=interp)
        cnt_crop_512 = cv2.resize(cnt_crop, (512,512), interpolation=interp)
        cv2.imwrite(cnt_dir+cnt_name, cnt_crop)
        cv2.imwrite(cnt_dir_16+cnt_name_16, cnt_crop_16)
        cv2.imwrite(cnt_dir_16+cnt_name_16_512, cnt_crop_512)
        cv2.imwrite(cnt_dir_32+cnt_name_32, cnt_crop_32)
        cv2.imwrite(cnt_dir_32+cnt_name_32_512, cnt_crop_512)
        # shutil.move(ROI.file_path+ex_f,cnt_dir+ex_f)
        status_bar.update(n=1)
    status_bar.close()    

def main():
    for ex_f in img_files:
        image_o = cv2.imread(ROI.file_path+ex_f)
        hMin = sMin = vMin = hMax = sMax = vMax = 0
        phMin = psMin = pvMin = phMax = psMax = pvMax = 0
        cv2.namedWindow('image',cv2.WINDOW_FREERATIO)
        cv2.namedWindow('original',cv2.WINDOW_FREERATIO)
        cv2.namedWindow('img_boxes', cv2.WINDOW_FREERATIO)
        cv2.resizeWindow('image', 1536,768)
        cv2.createTrackbar('avg_blur', 'image', 1, 20, nothing)
        cv2.createTrackbar('Invert_Binary', 'image', 0, 1, nothing)
        cv2.createTrackbar('Thresh_OTSU', 'image', 0, 1, nothing)
        cv2.createTrackbar('Color_Binary', 'image', 0, 1, nothing)
        cv2.createTrackbar('col_thresh_LR', 'image', 1, 255, nothing)
        cv2.createTrackbar('col_thresh_UR', 'image', 1, 255, nothing)
        cv2.createTrackbar('col_thresh_LG', 'image', 1, 255, nothing)
        cv2.createTrackbar('col_thresh_UG', 'image', 1, 255, nothing)
        cv2.createTrackbar('col_thresh_LB', 'image', 1, 255, nothing)
        cv2.createTrackbar('col_thresh_UB', 'image', 1, 255, nothing)
        cv2.createTrackbar('Cont_Method', 'image', 0, 1, nothing)
        cv2.createTrackbar('dilate_iterations', 'image', 1, 20, nothing)
        cv2.createTrackbar('MorphRect_X', 'image', 3, 10, nothing)
        cv2.createTrackbar('MorphRect_Y', 'image', 3, 10, nothing)
        cv2.createTrackbar('img_erode', 'image', 0, 1, nothing)
        cv2.createTrackbar('erode_iter', 'image', 1, 20, nothing)
        cv2.createTrackbar('Norm_Floor', 'image', 1, 255, nothing)
        cv2.createTrackbar('Norm_Ceil', 'image', 1, 255, nothing)
        cv2.createTrackbar('Thresh_Floor', 'image', 1, 255, nothing)
        cv2.createTrackbar('Thresh_Ceil', 'image', 1, 255, nothing)
        cv2.createTrackbar('Cnt_R', 'image', 0, 255, nothing)
        cv2.createTrackbar('rect_R', 'image', 0, 255, nothing)
        cv2.createTrackbar('Cnt_G', 'image', 0, 255, nothing)
        cv2.createTrackbar('rect_G', 'image', 0, 255, nothing)
        cv2.createTrackbar('Cnt_B', 'image', 0, 255, nothing)
        cv2.createTrackbar('rect_B', 'image', 0, 255, nothing)
        cv2.createTrackbar('HMin', 'image', 0, 179, nothing)
        cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
        cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
        cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
        cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
        cv2.createTrackbar('VMax', 'image', 0, 255, nothing)
        cv2.setTrackbarPos('avg_blur', 'image', 1)
        cv2.setTrackbarPos('Invert_Binary', 'image',0)
        cv2.setTrackbarPos('Thresh_OTSU', 'image',1)
        cv2.setTrackbarPos('Color_Binary', 'image', 0,)
        cv2.setTrackbarPos('col_thresh_LR', 'image', 0)
        cv2.setTrackbarPos('col_thresh_LG', 'image', 0)
        cv2.setTrackbarPos('col_thresh_LB', 'image', 115)
        cv2.setTrackbarPos('col_thresh_UR', 'image', 255)
        cv2.setTrackbarPos('col_thresh_UG', 'image', 178)
        cv2.setTrackbarPos('col_thresh_UB', 'image', 255)
        cv2.setTrackbarPos('Cont_Method', 'image', 0)
        cv2.setTrackbarPos('dilate_iterations', 'image', 1)
        cv2.setTrackbarPos('MorphRect_X', 'image', 3)
        cv2.setTrackbarPos('MorphRect_Y', 'image', 3)
        cv2.setTrackbarPos('img_erode', 'image', 0)
        cv2.setTrackbarPos('erode_iter', 'image', 0)
        cv2.setTrackbarPos('Norm_Floor','image', 0)
        cv2.setTrackbarPos('Norm_Ceil','image', 255)
        cv2.setTrackbarPos('Thresh_Floor', 'image', 0)
        cv2.setTrackbarPos('Thresh_Ceil', 'image', 255)
        cv2.setTrackbarPos('Cnt_R', 'image', 255)
        cv2.setTrackbarPos('rect_R', 'image', 0)
        cv2.setTrackbarPos('Cnt_G', 'image', 255)
        cv2.setTrackbarPos('rect_G', 'image', 0)
        cv2.setTrackbarPos('Cnt_B', 'image', 255)
        cv2.setTrackbarPos('rect_B', 'image', 0)
        cv2.setTrackbarPos('HMin', 'image', 0)
        cv2.setTrackbarPos('SMin', 'image', 0)
        cv2.setTrackbarPos('VMin', 'image', 0)
        cv2.setTrackbarPos('HMax', 'image', 255)
        cv2.setTrackbarPos('SMax', 'image', 255)
        cv2.setTrackbarPos('VMax', 'image', 255)
        cvo_img = cv2.copyMakeBorder(image_o, 10, 10, 10, 10,cv2.BORDER_CONSTANT,value = [0, 0, 0, 0])
        del image_o
        while 1:
            ROI.avg_blur = cv2.getTrackbarPos('avg_blur', 'image')
            avg_tup = (ROI.avg_blur,ROI.avg_blur)
            ROI.Invert_Binary = cv2.getTrackbarPos('Invert_Binary', 'image')
            ROI.Thresh_OTSU = cv2.getTrackbarPos('Invert_Binary', 'image')
            ROI.Color_Binary = cv2.getTrackbarPos('Color_Binary', 'image')
            ROI.col_thresh_LR = cv2.getTrackbarPos('col_thresh_LR', 'image')
            ROI.col_thresh_LG = cv2.getTrackbarPos('col_thresh_LG', 'image')
            ROI.col_thresh_LB = cv2.getTrackbarPos('col_thresh_LB', 'image')
            ROI.col_thresh_UR = cv2.getTrackbarPos('col_thresh_UR', 'image')
            ROI.col_thresh_UG = cv2.getTrackbarPos('col_thresh_UG', 'image')
            ROI.col_thresh_UB = cv2.getTrackbarPos('col_thresh_UB', 'image')
            ROI.Cont_Method = cv2.getTrackbarPos('Cont_Method', 'image')
            ROI.dilate_iterations = cv2.getTrackbarPos('dilate_iterations', 'image')
            ROI.MorphRect_X = cv2.getTrackbarPos('MorphRect_X', 'image')
            ROI.MorphRect_Y = cv2.getTrackbarPos('MorphRect_Y', 'image')
            ROI.img_erode = cv2.getTrackbarPos('img_erode', 'image')
            ROI.erode_iter = cv2.getTrackbarPos('erode_iter', 'image')
            ROI.thresh_floor = cv2.getTrackbarPos('Thresh_Floor', 'image')
            ROI.thresh_ceil = cv2.getTrackbarPos('Thresh_Ceil', 'image')
            ROI.Cnt_R = cv2.getTrackbarPos('Cnt_R', 'image')
            ROI.Cnt_G = cv2.getTrackbarPos('Cnt_G', 'image')
            ROI.Cnt_B = cv2.getTrackbarPos('Cnt_B', 'image')
            ROI.rect_R = cv2.getTrackbarPos('rect_R', 'image')
            ROI.rect_G = cv2.getTrackbarPos('rect_G', 'image')
            ROI.rect_B = cv2.getTrackbarPos('rect_B', 'image')
            ROI.Norm_Floor = cv2.getTrackbarPos('Norm_Floor', 'image')
            ROI.Norm_Ceil = cv2.getTrackbarPos('Norm_Ceil', 'image')
            ROI.hMin = cv2.getTrackbarPos('HMin', 'image')
            ROI.sMin = cv2.getTrackbarPos('SMin', 'image')
            ROI.vMin = cv2.getTrackbarPos('VMin', 'image')
            ROI.hMax = cv2.getTrackbarPos('HMax', 'image')
            ROI.sMax = cv2.getTrackbarPos('SMax', 'image')
            ROI.vMax = cv2.getTrackbarPos('VMax', 'image')
            ROI.lower = np.array([ROI.hMin, ROI.sMin, ROI.vMin])
            ROI.upper = np.array([ROI.hMax, ROI.sMax, ROI.vMax])
            cv_img = cvo_img.copy()
            norm_img = np.zeros(cv_img.shape)
            norm_img = cv2.normalize(cv_img,norm_img,ROI.Norm_Floor,ROI.Norm_Ceil,cv2.NORM_MINMAX)
            cv_img = norm_img
            hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, ROI.lower, ROI.upper)
            cv_mask = cv2.bitwise_and(cv_img, cv_img, mask=mask)
            cv_mask = cv2.cvtColor(cv_mask,cv2.COLOR_HSV2BGR) 
            if ROI.img_erode == 1:
                ero_kern = np.ones((5,5),np.uint8)
                cv_mask = cv2.erode(cv_mask, ero_kern, ROI.erode_iter)
            if ROI.avg_blur > 0:
                cv_mask = cv2.blur(cv_mask,avg_tup)
            if ROI.Color_Binary == 1:
                col_L = np.array((ROI.col_thresh_LR, ROI.col_thresh_LG, ROI.col_thresh_LB))
                col_U = np.array((ROI.col_thresh_UR, ROI.col_thresh_UG, ROI.col_thresh_UB))
                img_bin = cv2.inRange(cv_mask, col_L, col_U)
            img_grey = cv2.cvtColor(cv_mask,cv2.COLOR_BGR2GRAY)
            if ROI.Invert_Binary == 1:
                _, img_bin = cv2.threshold(img_grey, ROI.thresh_floor, ROI.thresh_ceil, cv2.THRESH_BINARY_INV)
            elif ROI.Color_Binary == 1:
                col_L = np.array((ROI.col_thresh_LR, ROI.col_thresh_LG, ROI.col_thresh_LB))
                col_U = np.array((ROI.col_thresh_UR, ROI.col_thresh_UG, ROI.col_thresh_UB))
                img_bin = cv2.inRange(cv_mask, col_L, col_U)
            elif ROI.Thresh_OTSU ==1:
                blur = cv2.GaussianBlur(cv_mask,(3,3),0)
                _, img_bin = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)                
            else:
                _, img_bin = cv2.threshold(img_grey, ROI.thresh_floor, ROI.thresh_ceil, cv2.THRESH_BINARY)
            if ROI.Cont_Method == 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
                dilated = cv2.dilate(img_bin, kernel, ROI.dilate_iterations)
                cnts, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            else:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ROI.MorphRect_X,ROI.MorphRect_Y))
                dilated = cv2.dilate(img_bin, kernel, ROI.dilate_iterations)
                cnts, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            imgCnts =  cvo_img.copy()
            cv2.drawContours(imgCnts, cnts, -1, (ROI.Cnt_B, ROI.Cnt_G, ROI.Cnt_R), -1)
            for c in cnts:
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(imgCnts, (x, y), (x + w, y + h), (ROI.rect_B,ROI.rect_G,ROI.rect_R), 1)
            size = max(imgCnts.shape[0:2])
            pad_x = size - imgCnts.shape[1]
            pad_y = size - imgCnts.shape[0]
            pad_l = pad_x // 2
            pad_t = pad_y // 2
            imgCnts = np.pad(imgCnts, ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)), mode='constant', constant_values=255)
            imgCnts_S = cv2.resize(imgCnts, (768,768), interpolation=cv2.INTER_LANCZOS4)
            cv2.imshow('image', imgCnts_S)
            cv2.imshow('img_boxes', imgCnts_S)
            cv2.imshow('original', cvo_img)
            k = cv2.waitKey(100) & 0xFF
            if k == 27:
                break
            if k == ord('s'):
                cv2.destroyAllWindows()
                sprite_extract_roi(ex_f,cnts,cvo_img)
                break
            if k == ord('r'):
                cv2.destroyAllWindows()
            if k == ord('n'):
                cv2.destroyAllWindows()
                break
            if k == ord('q'):
                cv2.destroyAllWindows()
                sys.exit()
            print(f"avg_blur: {str(ROI.avg_blur)} bin_inv:{str(ROI.Invert_Binary)} cont_meth:{str(ROI.Cont_Method)} dilate_iter:{str(ROI.dilate_iterations)} thresh_floor: {str(ROI.thresh_floor)} thresh_ceil: {str(ROI.thresh_ceil)} norm_floor: {str(ROI.Norm_Floor)} norm_ceil: {str(ROI.Norm_Ceil)} hMin:{hMin} sMin:{sMin} vMin:{vMin} hMax:{hMax} sMax:{sMax} vMax:{vMax}")
main()

cv2.destroyAllWindows()
