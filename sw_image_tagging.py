from __future__ import annotations
import os
import cv2
import argparse
from PIL import Image,ImageFile,ImageOps
from huggingface_hub import hf_hub_download
from traceback_with_variables import activate_by_import
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import huggingface_hub
import onnxruntime as rt
import numpy as np
import pandas as pd
import keyring
from tqdm import tqdm
import codecs
import unicodedata
from iptcinfo3 import IPTCInfo
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed
from termcolor import cprint

print("")
cuda_check = str(rt.get_device())
print(str(cuda_check))
print("")
assert cuda_check == str("GPU")

CONTAIN_W_H = (576,576)
IMAGE_SIZE = 448
HF_TOKEN = keyring.get_password("hf","hf_key")
MOAT_MODEL_REPO = "SmilingWolf/wd-v1-4-moat-tagger-v2"
SWINV2_MODEL_REPO = "SmilingWolf/wd-v1-4-swinv2-tagger-v2"
CONV_MODEL_REPO = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
CONV2_MODEL_REPO = "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"
VIT_MODEL_REPO = "SmilingWolf/wd-v1-4-vit-tagger-v2"
MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"
EP_LIST = ['CUDAExecutionProvider']
GEN_THRESH:float = 0.35
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

replace_list = {
                r"2018":"",
                r"2020":"",
                r"2020s":"",
                r"2021":"",
                r"2023":"",
                r"!":"",
                r"+_+":"",
                r"...":"",
                r"!":"",
                r"///":"",
                r"\\m/":"",
                r"\\||/":"",
                r"^^^":"",
                r":)":"",
                r":/":"",
                r":3":"",
                r":d":"",
                r":o":"",
                r":p":"",
                r":q":"",
                r"?":"",
                r"\\m/":"",
                r":3":"",
                r":<":"",
                r":i":"",
                r":o":"",
                r":p":"",
                r":q":"",
                r":t":"",
                r";)":"",
                r";d":"",
                r";o":"",
                r"?":"",
                r"\m/":"",
                r"\\||/":"",
                r"^^^":"",
                r"^_^":"",
                r"0_0":"",
                r"(o)_(o)":"",
                r"+_+":"",
                r"+_-":"",
                r"._.":"",
                r"<o>_<o>":"",
                r"<|>_<|>":"",
                r"=_=":"",
                r">_<":"",
                r"3_3":"",
                r"6_9":"",
                r">_o":"",
                r"@_@":"",
                r"^_^":"",
                r"o_o":"",
                r"u_u":"",
                r"x_x":"",
                r"|_|":"",
                r"||_||":"",
                r":>=":""
                }

@staticmethod
def cp_g(x:str)->str: 
    cs =  cprint(x, "green",end="")
    return cs
@staticmethod    
def cp_y(x:str)->str: 
    cs =  cprint(x, "yellow",end="")
    return cs
@staticmethod
def cp_c(x:str)->str: 
    cs =  cprint(x, "cyan",end="")
    return cs

def load_model(model_name: str) -> rt.InferenceSession:
    if model_name == "moat":
        repo = MOAT_MODEL_REPO
        tag_names, general_indexes = load_labels(repo)
    elif model_name == "swinv2":
        repo = SWINV2_MODEL_REPO
        tag_names, general_indexes = load_labels(repo)
    elif model_name == "convnext":
        repo = CONV_MODEL_REPO
        tag_names, general_indexes = load_labels(repo)
    elif model_name == "convnextv2":
        repo = CONV2_MODEL_REPO
        tag_names, general_indexes = load_labels(repo)
    elif model_name == "vit":
        repo = VIT_MODEL_REPO
        tag_names, general_indexes = load_labels(repo)
    path = huggingface_hub.hf_hub_download(repo, MODEL_FILENAME, use_auth_token=HF_TOKEN)
    model = rt.InferenceSession(path,providers=EP_LIST)
    tag_names, general_indexes = load_labels(repo)
    return model, tag_names, general_indexes

def load_labels(repo) -> list[str]:
    tag_path = hf_hub_download(repo, LABEL_FILENAME, use_auth_token=HF_TOKEN)
    df = pd.read_csv(tag_path)
    tag_names = df["name"].tolist()
    general_indexes = list(np.where(df["category"] == 0)[0])
    return tag_names,general_indexes

def txt_write(t_name:str,tag_list:list):
    tags = str(", ".join(x for x in list(str(x).replace("_",chr(32)) for x in map(lambda x: str(x).strip().lower().replace(x, replace_list[x]) if x in replace_list.keys() else x,tag_list) if len(x)>0)))
    if not os.path.isfile(t_name):
        with open(t_name,'wt') as fi:
            fi.write(tags)
            fi.close()
    elif os.path.isfile(t_name):
        with open(t_name,'a') as fi:
            fi.write(str(", ")+tags)
            fi.close()
    return

def img_contain(file:str,save_path:str):
    t_name = file[:(len(file))-1-len(file[-(file[::-1].find('.')):])]+".txt" 
    if not os.path.isfile(t_name):
        with open(t_name,'wt') as fi:
            fi.write(str(""))
            fi.close()
    f_base = os.path.basename(file)
    iptc_info = IPTCInfo(file, force=True)
    tag_list = [codecs.decode(x,encoding='utf-8') for x in iptc_info['keywords']]
    unique_tags = np.unique(np.array(list(tag_list))).tolist()   
    img = Image.open(file)
    img = img.convert("RGB")
    exif = img.getexif()
    img = ImageOps.contain(img, CONTAIN_W_H, method=Image.Resampling.BICUBIC)
    img.save(
            save_path+f_base[:(len(f_base))-1-len(f_base[-(f_base[::-1].find('.')):])]+".jpg"
            ,"JPEG"
            ,quality=100
            ,exif=exif
            )
    iptc_info = IPTCInfo(save_path+f_base[:(len(f_base))-1-len(f_base[-(f_base[::-1].find('.')):])]+".jpg", force=True)
    tag_u = [codecs.encode(x,encoding='utf-8') for x in unique_tags]
    iptc_info['keywords'] = tag_u
    iptc_info.save()  

def img_tag_proc(file:str,file_path:str):
    t_name = file[:(len(file))-1-len(file[-(file[::-1].find('.')):])]+".txt" 
    if os.path.isfile(t_name):
        with open(t_name,'rt') as fi:
            txt_data = fi.read()
    txt_data = bytes(txt_data,'utf-8')
    txt_data = codecs.decode(unicodedata.normalize('NFKD', codecs.decode(txt_data)).encode('ascii', 'ignore'))  
    txt_data = str(txt_data)
    txt_list = txt_data.split(",")
    iptc_info = IPTCInfo(file, force=True)
    tag_list = [codecs.decode(x,encoding='utf-8') for x in iptc_info['keywords']]
    all_Tags=[]
    for x in txt_list:
            all_Tags.append(str(x).strip().lower())
    for x in tag_list:
            all_Tags.append(str(x).strip().lower())
    unique_tags = []
    unique_tags = np.unique(np.array(all_Tags)).tolist()
    tags = [x for x in list(str(x).replace("_",chr(32)) for x in map(lambda x: str(x).strip().lower().replace(x, replace_list[x]) if len(x)>0 and str(x).strip().lower() in replace_list.keys() and len(x)!=32 else x,unique_tags))]
    tag_write = str(",".join(str(x).strip().lower().replace("(",",").replace(")","").replace("_",chr(32)) for x in tags if len(x)>0 and str(x).strip().lower().find("skin")==-1))
    for k in replace_list.keys():
        tag_write = tag_write.replace(k,replace_list[k])
    unique_tags = []
    unique_tags = np.unique(np.array([codecs.encode(x,encoding='utf-8') for x in tag_write.split(",")])).tolist()
    del iptc_info
    iptc_info = IPTCInfo(file,force=True)
    iptc_info['keywords'] = unique_tags
    iptc_info.save()        
    with open(t_name,"wt") as fi:
        fi.write(tag_write)
    # os.remove(t_name)

def model_pred(image,t_name,general_threshold,tag_names,general_indexes,model):
    input_name = model.get_inputs()[0].name
    label_name = model.get_outputs()[0].name
    probs = model.run([label_name], {input_name: image})[0]
    labels = list(zip(tag_names, probs[0].astype(float)))
    general_names = [labels[i] for i in general_indexes]
    general_res = [x for x in general_names if x[1] > general_threshold]
    general_res = dict(general_res)
    b = dict(sorted(general_res.items(), key=lambda item: item[1], reverse=True))
    tag_list = list(b.keys())
    txt_write(t_name,tag_list)
    del image,b,labels,input_name,label_name,probs,general_names,general_res,tag_list
    return

def img_proc(i):
    img = cv2.imread(i)
    size = max(img.shape[0:2])
    pad_x = size - img.shape[1]
    pad_y = size - img.shape[0]
    pad_l = pad_x // 2
    pad_t = pad_y // 2
    img = np.pad(img, ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)), mode='constant', constant_values=255)
    interp = cv2.INTER_CUBIC
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=interp)
    image = np.expand_dims(img, 0)
    image = image.astype(np.float32)
    return image

def predict(i,general_threshold,tag_names,general_indexes,model):
    t_name=i[:(len(i))-1-len(i[-(i[::-1].find('.')):])]+".txt" 
    image = img_proc(i)
    model_pred(image,t_name,general_threshold,tag_names,general_indexes,model)
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default=None, help="Save Path")
    parser.add_argument("--file_path", type=str, default=None, help="File Path")
    parser.add_argument("--gen_thresh", type=float, default=0.35, help="General Tags Threshold - default: 0.35")
    parser.add_argument("--model_name", type=str, default="ViT", help="MOAT,SwinV2,ConvNext,ConvNextV2,ViT - default: ViT")
    parser.add_argument("--process_threads", type=int, default="8", help="Number of Process Threads for ThreadPoolExecutor - default: 8")
    parser.add_argument("--list_gen", type=str, default="list", help="[WALK] - all files in subdirs, [LIST] all files in ListDir directory - default: LIST")
    args = parser.parse_args()
    file_list=[]
    file_path = str(args.file_path).replace('"','').replace("'","")
    file_path = file_path.replace(chr(92),"/")
    if file_path[-1:] != "/":
        file_path = file_path+"/"
    save_path = str(args.save_path)
    save_path = save_path.replace(chr(92),"/")
    if save_path[-1:] != "/":
        save_path = save_path+"/"
    model_name = str(args.model_name).lower()
    list_gen = str(args.list_gen).lower()
    thread_num = int(args.process_threads)
    gen_thresh = str(args.gen_thresh)
    ms = str(f"{cp_g('File Path: ')}{cp_y(file_path)}{cp_c(chr(10))}")
    ms = str(f"{cp_g('Save Path: ')}{cp_y(save_path)}{cp_c(chr(10))}")
    ms = str(f"{cp_g('Model Name: ')}{cp_y(str(model_name))}{cp_c(chr(10))}")
    ms = str(f"{cp_g('General Tags Threshold: ')}{cp_y(gen_thresh)}{cp_c(chr(10))}")
    ms = str(f"{cp_g('File List Gen: ')}{cp_y(str(list_gen).lower())}{cp_c(chr(10))}")
    ms = str(f"{cp_g('Number of Threads: ')}{cp_y(str(thread_num).lower())}")
    ms = ms + str(f"{cp_c(chr(10))}")
    model, tag_names, general_indexes = load_model(model_name)
    l_bar='{desc}: {percentage:3.0f}%|'
    r_bar='| {n_fmt}/{total_fmt} [elapsed: {elapsed} / Remaining: {remaining}] '
    bar = '{rate_fmt}{postfix}]'
    
    if list_gen == "walk":
        for r, d, f in os.walk(file_path[:-1:]):
            for file in f:
                if ((file[-(file[::-1].find('.')):]).lower()) in IMG_TYPES:
                    file_list.append(os.path.join(r, file))
    elif list_gen == "list":
        file_list = [file_path+f for f in os.listdir(file_path[:-1:]) if os.path.isfile(file_path+f) and f[-(f[::-1].find('.')):] in IMG_TYPES]

    # with ThreadPoolExecutor(args.process_threads) as executor:       
    #     status_bar = tqdm(total=len(file_list), desc=r'Image Contain',bar_format=f'{l_bar}{bar}{r_bar}')
    #     futures = [
    #         executor.submit(img_contain,i,save_path) for i in file_list]
    #     for _ in as_completed(futures):
    #         status_bar.update(n=1)
    #     status_bar.close()

    save_list=[save_path+os.path.basename(x) for x in file_list]

    with ThreadPoolExecutor(args.process_threads) as executor:       
        status_bar = tqdm(total=len(save_list), desc=r'Image Tagging',bar_format=f'{l_bar}{bar}{r_bar}')
        futures = [
            executor.submit(predict,i,args.gen_thresh,tag_names,general_indexes,model) for i in save_list]
        for _ in as_completed(futures):
            status_bar.update(n=1)
        status_bar.close()

    with ThreadPoolExecutor(args.process_threads) as executor:       
        status_bar = tqdm(total=len(save_list), desc=r'Image Keywords to EXIF/IPTC ',bar_format=f'{l_bar}{bar}{r_bar}')
        futures = [
            executor.submit(img_tag_proc,i,file_path) for i in save_list]
        for _ in as_completed(futures):
            status_bar.update(n=1)
        status_bar.close()



if __name__ == "__main__":
    main()

#  --gen_thresh 0.25 --model_name moat --list_gen list --process_threads 4 --file_path ./kskjdhfklasdhfklasdhjf --save_path ./kjshdfkjlashdflkashjdf
