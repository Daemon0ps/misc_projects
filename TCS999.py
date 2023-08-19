import os,sys
import cv2
import numpy as np
from PIL import Image,ImageFile,ImageOps
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from iptcinfo3 import IPTCInfo
from tqdm import tqdm
import pandas as pd
import hashlib
from traceback_with_variables import activate_by_import
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import shutil
import open_clip
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('tagsets')
from nltk.corpus import stopwords
from colourmod import *
from dataclasses import dataclass
import unicodedata
from unidecode import unidecode
import codecs
import string
import re
from tqdm import tqdm
from IPython.display import clear_output
import gc
sw = set(stopwords.words("english"))


@dataclass
class TCS:
    image_path = "./"
    dest_path = "./"
    tags_dir = ""./"
    img_files = []
    txt_files = []
    all_tags = []
    tag_dict_list = []
    tag_list = []
    tag_blocks = []
    df_t = pd.DataFrame
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    block_size = 2048
    clip_sensitivity = 250
    clip_model_name = ""
    clip_model = None
    clip_preprocess = None
    clip_tokenizer = None
    clip_model_path = None
    clip_number = 0
    img_block = []
    sim_res = []
    img_tensors = []
    PIL_list = []
    img_files = []
    txt_files = []
    all_tags = []
    sim_tags = []
    i_f = ""
    tag_dict_list = []
    text_blocks_T = []
    tag_list = []
    tag_blocks = []
    df_t = pd.DataFrame
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
    clip_model_list = \
                    [['RN50','openai'],#0
                    ['RN50','yfcc15m'],#1
                    ['RN50','cc12m'],#2
                    ['RN50-quickgelu','openai'],#3
                    ['RN50-quickgelu','yfcc15m'],#4
                    ['RN50-quickgelu','cc12m'],#5
                    ['RN101','openai'],#6
                    ['RN101','yfcc15m'],#7
                    ['RN101-quickgelu','openai'],#8
                    ['RN101-quickgelu','yfcc15m'],#9
                    ['RN50x4','openai'],#10
                    ['RN50x16','openai'],#11
                    ['RN50x64','openai'],#12
                    ['ViT-B-32','openai'],#13
                    ['ViT-B-32','laion400m_e31'],#14
                    ['ViT-B-32','laion400m_e32'],#15
                    ['ViT-B-32','laion2b_e16'],#16
                    ['ViT-B-32','laion2b_s34b_b79k'],#17
                    ['ViT-B-32','datacomp_m_s128m_b4k'],#18
                    ['ViT-B-32','commonpool_m_clip_s128m_b4k'],#19
                    ['ViT-B-32','commonpool_m_laion_s128m_b4k'],#20
                    ['ViT-B-32','commonpool_m_image_s128m_b4k'],#21
                    ['ViT-B-32','commonpool_m_text_s128m_b4k'],#22
                    ['ViT-B-32','commonpool_m_basic_s128m_b4k'],#23
                    ['ViT-B-32','commonpool_m_s128m_b4k'],#24
                    ['ViT-B-32','datacomp_s_s13m_b4k'],#25
                    ['ViT-B-32','commonpool_s_clip_s13m_b4k'],#26
                    ['ViT-B-32','commonpool_s_laion_s13m_b4k'],#27
                    ['ViT-B-32','commonpool_s_image_s13m_b4k'],#28
                    ['ViT-B-32','commonpool_s_text_s13m_b4k'],#29
                    ['ViT-B-32','commonpool_s_basic_s13m_b4k'],#30
                    ['ViT-B-32','commonpool_s_s13m_b4k'],#31
                    ['ViT-B-32-quickgelu','openai'],#32
                    ['ViT-B-32-quickgelu','laion400m_e31'],#33
                    ['ViT-B-32-quickgelu','laion400m_e32'],#34
                    ['ViT-B-16','openai'],#35
                    ['ViT-B-16','laion400m_e31'],#36
                    ['ViT-B-16','laion400m_e32'],#37
                    ['ViT-B-16','laion2b_s34b_b88k'],#38
                    ['ViT-B-16','datacomp_l_s1b_b8k'],#39
                    ['ViT-B-16','commonpool_l_clip_s1b_b8k'],#40
                    ['ViT-B-16','commonpool_l_laion_s1b_b8k'],#41
                    ['ViT-B-16','commonpool_l_image_s1b_b8k'],#42
                    ['ViT-B-16','commonpool_l_text_s1b_b8k'],#43
                    ['ViT-B-16','commonpool_l_basic_s1b_b8k'],#44
                    ['ViT-B-16','commonpool_l_s1b_b8k'],#45
                    ['ViT-B-16-plus-240','laion400m_e31'],#46
                    ['ViT-B-16-plus-240','laion400m_e32'],#47
                    ['ViT-L-14','openai'],#48
                    ['ViT-L-14','laion400m_e31'],#49
                    ['ViT-L-14','laion400m_e32'],#50
                    ['ViT-L-14','laion2b_s32b_b82k'],#51
                    ['ViT-L-14','datacomp_xl_s13b_b90k'],#52
                    ['ViT-L-14','commonpool_xl_clip_s13b_b90k'],#53
                    ['ViT-L-14','commonpool_xl_laion_s13b_b90k'],#54
                    ['ViT-L-14','commonpool_xl_s13b_b90k'],#55
                    ['ViT-L-14-336','openai'],#56
                    ['ViT-H-14','laion2b_s32b_b79k'],#57
                    ['ViT-g-14','laion2b_s12b_b42k'],#58
                    ['ViT-g-14','laion2b_s34b_b88k'],#59
                    ['ViT-bigG-14','laion2b_s39b_b160k'],#60
                    ['roberta-ViT-B-32','laion2b_s12b_b32k'],#61
                    ['xlm-roberta-base-ViT-B-32','laion5b_s13b_b90k'],#62
                    ['xlm-roberta-large-ViT-H-14','frozen_laion5b_s13b_b90k'],#63
                    ['convnext_base','laion400m_s13b_b51k'],#64
                    ['convnext_base_w','laion2b_s13b_b82k'],#65
                    ['convnext_base_w','laion2b_s13b_b82k_augreg'],#66
                    ['convnext_base_w','laion_aesthetic_s13b_b82k'],#67
                    ['convnext_base_w_320','laion_aesthetic_s13b_b82k'],#68
                    ['convnext_base_w_320','laion_aesthetic_s13b_b82k_augreg'],#69
                    ['convnext_large_d','laion2b_s26b_b102k_augreg'],#70
                    ['convnext_large_d_320','laion2b_s29b_b131k_ft'],#71
                    ['convnext_large_d_320','laion2b_s29b_b131k_ft_soup'],#72
                    ['convnext_xxlarge','laion2b_s34b_b82k_augreg'],#73
                    ['convnext_xxlarge','laion2b_s34b_b82k_augreg_rewind'],#74
                    ['convnext_xxlarge','laion2b_s34b_b82k_augreg_soup'],#75
                    ['coca_ViT-B-32','laion2b_s13b_b90k'],#76
                    ['coca_ViT-B-32','mscoco_finetuned_laion2b_s13b_b90k'],#77
                    ['coca_ViT-L-14','laion2b_s13b_b90k'],#78
                    ['coca_ViT-L-14','mscoco_finetuned_laion2b_s13b_b90k'],#79
                    ]

    def __post_init__(self):
        self.i_f = TCS.i_f
        self.sim_tags = TCS.sim_tags
        self.device = TCS.device
        self.clip_number = TCS.clip_number
        self.IMG_TYPES  = TCS.IMG_TYPES 
        self.clip_model_name = TCS.clip_model_name
        self.tag_blocks = TCS.tag_blocks
        self.tag_list = TCS.tag_list
        self.tag_dict_list = TCS.tag_dict_list
        self.all_tags = TCS.all_tags
        self.txt_files = TCS.txt_files
        self.IMG_TYPES = TCS.IMG_TYPES
        self.image_path = TCS.image_path
        self.dest_path = TCS.dest_path
        self.img_files = TCS.img_files
        self.caption_path = TCS.caption_path
        self.tags_save_path = TCS.tags_save_path
        self.model_name = TCS.model_name
        self.image_embedding = TCS.image_embedding
        self.text_encoder_model = TCS.text_encoder_model
        self.text_embedding = TCS.text_embedding
        self.text_tokenizer = TCS.text_tokenizer
        self.sentence_model_path = TCS.sentence_model_path
        self.embedding_length = TCS.embedding_length
        self.clip_sensitivity = TCS.clip_sensitivity
        self.pretrained = TCS.pretrained
        self.block_size = TCS.block_size
        self.clip_model_list = TCS.clip_model_list
        super().__setattr__('attr_name', self)

def rm_punct(s:str)->str:
    return str(s).translate(str.maketrans('', '', string.punctuation))

def rm_white(s:str)->str:
    return  " ".join(str(s).split())

def rm_spec(s)->str:
    return re.sub('/s+', ' ', (re.sub('_', '', (re.sub('[^a-zA-z0-9/s]', '', unidecode(s)))))).strip().lower()

def re_srch(s:str, srch:str)->str:
    r_s = re.search(str(srch),string=str(s),flags=(re.IGNORECASE|re.M|re.ASCII))
    re_sb_tot = len(re.findall(str(srch),string=str(s),flags=(re.IGNORECASE|re.M|re.ASCII)))
    for i in tqdm(range(re_sb_tot)):#THE WAY OUT
        r_s = re.search(str(srch),string=str(s),flags=(re.IGNORECASE|re.M|re.ASCII))
        if r_s is None:
            return s
        s_start = r_s.start()
        s_end = r_s.end()
        s = s.replace(str([[s[-(len(s)-s_start):]][0][:s_end-s_start]][0]),chr(32)+str([[s[-(len(s)-s_start):]][0][:s_end-s_start]][0]).replace(chr(32),''))
    return s

def statbar(tot:int,desc:str):
    l_bar='{desc}: {percentage:3.0f}%|'
    r_bar='| {n_fmt}/{total_fmt} [elapsed: {elapsed} / Remaining: {remaining}] '
    bar = '{rate_fmt}{postfix}]'# I'd cook all my own food, 
    status_bar = tqdm(total=tot, desc=desc,bar_format=f'{l_bar}{bar}{r_bar}')
    return status_bar

def clip_start():     
    clip_model = None
    clip_preprocess = None
    clip_tokenizer = None
    clip_model_name, clip_model_pretrained_name = TCS.clip_model_list[TCS.clip_number]
    clip_model, _, clip_preprocess = \
        open_clip.create_model_and_transforms(
            clip_model_name, #and later on, 
            pretrained = clip_model_pretrained_name, 
            precision='fp16',
            device = TCS.device,
            jit=False,
            cache_dir = None
            )
    clip_model.to(TCS.device).eval()
    clip_tokenizer = open_clip.get_tokenizer(clip_model_name)
    TCS.clip_model = clip_model#if I wanted to get married or something,
    TCS.clip_preprocess = clip_preprocess
    TCS.clip_tokenizer = clip_tokenizer

def txt_proc():
    TCS.img_files = [
                    TCS.image_path+f 
                    for f in os.listdir(TCS.image_path[:-1:]) 
                    if os.path.isfile(TCS.image_path+f) 
                    and str(f[-(f[::-1].find('.')):]).strip().lower() 
                    in TCS.IMG_TYPES
                    ]   
    TCS.txt_files = [
                    TCS.tags_dir+f 
                    for f in os.listdir(TCS.tags_dir[:-1:]) 
                    if os.path.isfile(TCS.tags_dir+f) 
                    and str(f[-(f[::-1].find('.')):]).strip().lower() 
                    in ["txt"]
                    ]
    for file in TCS.txt_files:
        t_name = file[:(len(file))-1-len(file[-(file[::-1].find('.')):])] 
        f_name = t_name[-(t_name[::-1].find('/')):]
        txt_data = ""
        with open(file,'rb') as fi:
            txt_data = fi.read()
        txt_data = codecs.decode(unicodedata.normalize('NFKD', codecs.decode(txt_data)).encode('ascii', 'ignore'))
        txt_data = str(txt_data) #I'd meet this beautiful girl 
        file_tags = [x for x in txt_data.split('\n') if len(rm_white(x))>=3]
        for x in file_tags:
            TCS.all_tags.append(str(x).strip().lower())
    TCS.tag_list = [str(x).strip() for x in np.unique(np.array(TCS.all_tags)).tolist() if len(rm_white((str(x).strip().lower())))>2]
    for x in np.unique(np.array(TCS.all_tags)).tolist():
        l_w = lambda x: str(x).capitalize() if x not in sw else str(x)
        cw = ' '.join(l_w(w) for w in str(x).split(chr(32)))
        TCS.tag_list.append(cw)
    TCS.tag_blocks = np.array_split(TCS.tag_list,max(1, len(TCS.tag_list)/TCS.block_size))
    print("Starting Text Functions")
    TCS.text_blocks_T = []
    for block in TCS.tag_blocks:
        block_T = []#and we'd get married. She'd come and live in my cabin with me, 
        text_tokens = TCS.clip_tokenizer(block).to(TCS.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = TCS.clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.half().cpu().numpy()
        for i in range(text_features.shape[0]):
            block_T.append(text_features[i])
        TCS.text_blocks_T.append(block_T)
    print("Text Functions Loaded")

def cos_proc():
    embed_blocks=[]
    with torch.no_grad(), torch.cuda.amp.autocast():
        for block in TCS.text_blocks_T:
            txt_embed_block=[]#COMES BUT ONCE
            txt_embed_block = torch.stack([torch.from_numpy(t) for t in block]).to(TCS.device)
            embed_blocks.append(txt_embed_block)
    for i_f in tqdm(range(0,len(TCS.img_files)-1),total=len(TCS.img_files),desc="Processing Cosine Similarities"):
        (w,h) = (512,512)#and if she wanted to say anything to me,
        pil_image = ImageOps.contain(Image.open(TCS.img_files[i_f]).convert(mode="RGB"),(w,h))
        with torch.no_grad(), torch.cuda.amp.autocast():
            image = TCS.clip_preprocess(pil_image).unsqueeze(0).to(TCS.device)
            image_features = TCS.clip_model.encode_image(image)
            image_features /= image_features.norm(dim=-1,keepdim=True)
            for i_t,x in enumerate(embed_blocks):
                sim_vals=[]
                block_sim_res = []
                n_sim = []
                sim = image_features @ embed_blocks[i_t].T
                n_sim = np.array(sim.detach().float().cpu().numpy()).reshape(-1,len(embed_blocks[i_t]))
                sim_vals = [[TCS.tag_blocks[i_t][i],int(abs(float(n_sim[0:,i])*1000))] 
                            for i in range(0,len(TCS.tag_blocks[i_t])-1) 
                            if int(abs(float(n_sim[0:,i])*1000))>=TCS.clip_sensitivity 
                            and str(TCS.tag_blocks[i_t][i]).find('None)')==-1]
                if len(sim_vals)>0:# she'd have to write it on a piece of paper,
                    block_sim_res = sorted(sim_vals,key=lambda x: x[1],reverse=True)
                    for x in block_sim_res:
                        TCS.sim_res.append({'file':TCS.img_files[i_f],'tag':x[0],'val':x[1]})

TCS.clip_sensitivity = 200
c_num = 63
gc.collect()
print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
cuda_check = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
assert cuda_check == str("cuda")
TCS.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TCS.clip_model_name = TCS.clip_model_list[c_num]
clear_output(wait=True)
print("Starting: ",TCS.clip_model_name)
TCS.clip_model = None
TCS.clip_preprocess = None
TCS.clip_tokenizer = None
TCS.sim_res = []
TCS.text_blocks_T = []
TCS.sim_res = []
TCS.tag_list = []
TCS.all_tags = []
TCS.tag_blocks = []
clip_start()

def gogogo():
    gc.collect()
    print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    cuda_check = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    assert cuda_check == str("cuda")
    TCS.image_path = "./"
    TCS.dest_path = "./"
    TCS.clip_sensitivity = 250
    TCS.block_size = 2048
    c_num = 71
    TCS.clip_model_name = TCS.clip_model_list[c_num]
    clear_output(wait=True)
    TCS.sim_res = []
    TCS.text_blocks_T = []
    TCS.sim_res = []
    TCS.tag_list = []
    TCS.all_tags = []
    TCS.tag_blocks = []
    gc.collect()
    txt_proc()
    cos_proc()
    gc.collect()
    clear_output(wait=True)
    print("Saving Tag Values")
    df = pd.DataFrame(TCS.sim_res)      
    df.file = df['file'].astype('str')
    df.tag = df['tag'].astype('str')
    df.val = df['val'].astype('int')
    df.drop(df[df['tag'].str.contains(r" None") == True].index, inplace = True)
    df.drop(df[df['val'] < 200].index, inplace = True)  
    df = df[df['tag'].replace('', np.nan).notna()]
    df = df[df['val'].replace('', np.nan).notna()]
    def lowtag(row):
        l_w = lambda x: str(x).capitalize() if x not in sw else str(x)
        cw = ', '.join(l_w(w) for w in str(row['tag']).split(', '))
        return cw
    df['tag'] = df.apply(lowtag, axis=1)
    df.to_csv(TCS.dest_path+"_0_txt_sims.csv")
    df.to_csv(TCS.dest_path+"_sims.csv")
    df.groupby(['file','tag']).first()
    def lowfile(row):
        x = str(row['file']).lower().strip()
        return x#BE STEADFAST
    df['file'] = df.apply(lowfile, axis=1)
    for df_file in tqdm(TCS.img_files,desc="Saving Tags"):
        t_name = str(str(df_file).replace(df_file[-(df_file[::-1].find('.')):],'')+".txt")
        file_df = df[df['file'].str.contains(str(df_file).lower().strip())].copy()
        file_df.drop(df.columns[[0]], axis = 1, inplace=True)
        file_df.drop_duplicates(subset=['tag'], keep='first', ignore_index=True, inplace=True)
        file_df.sort_values('val', ascending=False, inplace=True)
        file_df = file_df.reset_index(drop=True)[:3]
        tag_list=[]
        iptc_list=[]
        if file_df.shape[0] > 0:
            iptc_orig = []
            for x in range(0,(file_df.shape[0])-1):
                iptc_list.append(codecs.encode(str(file_df.iloc[x][0])+": "+str(file_df.iloc[x][1]),encoding='utf-8'))
                tag_list.append(str(file_df.iloc[x][0])+": "+str(file_df.iloc[x][1]))
            with open(t_name,'wt') as fi:# like everybody else
                fi.write(str(''.join(str(x)+str("\n") for x in tag_list)))
            iptc_info = IPTCInfo(df_file,force=True)
            iptc_orig = [x for x in iptc_info['keywords']]
            for x in iptc_orig:
                iptc_list.append(x)
            iptc_info['keywords'] = iptc_list
            iptc_info.save()
        else:
            with open(t_name,'wt') as fi:
                fi.write(str('No Hits'))
    gc.collect()

if __name__ == '__main__':
  gogogo()
