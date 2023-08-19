# Lora training utility
# parses image files for IPTC keywords, txt files for keywords.
# builds 3 files:
#    kw_list.txt   -  all keywords, including counts
#    kw_repl_list.txt - all keywords, in a ready-for-dict format that can be copied and pasted back into the script to strip undesirable keywords in the dataset
#    train_prompt.txt - inserts the OBJECT WORD of the Lora at the front of 5 training prompts.
        # - training prompts are are the first 20 most common keywords + 20-25, 25-30, &c.
        # - designed to verify training from sampling

import os
import numpy as np
from tqdm import tqdm
import random
from traceback_with_variables import activate_by_import
import IPython.display
from IPython.display import clear_output
import unicodedata
import glob
from unidecode import unidecode
import codecs
import string
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('tagsets')
from nltk.corpus import stopwords
sw = set(stopwords.words("english"))

def statbar(tot:int,desc:str):
    l_bar='{desc}: {percentage:3.0f}%|'
    r_bar='| {n_fmt}/{total_fmt} [elapsed: {elapsed} / Remaining: {remaining}] '
    bar = '{rate_fmt}{postfix}]'
    status_bar = tqdm(total=tot, desc=desc,bar_format=f'{l_bar}{bar}{r_bar}')
    return status_bar

@staticmethod
def rm_punct(s:str)->str:
    return str(s).translate(str.maketrans('', '', string.punctuation))

@staticmethod
def rm_white(s:str)->str:
    return  " ".join(str(s).split())

def proc_fp()->str:
    print("File Path:",end="")
    fp = input()
    fp = fp.replace(chr(92),chr(47)).replace(chr(34),'')
    if fp[-1:] != "/":
        fp = fp+"/"
    print("\n",fp,"\n")
    if not os.path.isdir(fp+"faces"):
        os.makedirs(fp+"faces")
    return fp

def rm_spec(s)->str:
    return re.sub('/s+', ' ', (re.sub('_', '', (re.sub('[^a-zA-z0-9/s]', '', unidecode(s)))))).strip().lower()

def re_srch(s:str, srch:str)->str:
    r_s = re.search(str(srch),string=str(s),flags=(re.IGNORECASE|re.M|re.ASCII))
    re_sb_tot = len(re.findall(str(srch),string=str(s),flags=(re.IGNORECASE|re.M|re.ASCII)))
    for i in tqdm(range(re_sb_tot)):
        r_s = re.search(str(srch),string=str(s),flags=(re.IGNORECASE|re.M|re.ASCII))
        if r_s is None:
            return s
        s_start = r_s.start()
        s_end = r_s.end()
        s = s.replace(str([[s[-(len(s)-s_start):]][0][:s_end-s_start]][0]),chr(32)+str([[s[-(len(s)-s_start):]][0][:s_end-s_start]][0]).replace(chr(32),''))
    return s

all_tags = []
FP = proc_fp()
file_path = FP  #  <---   whhyyyyyy ? lol


##############################################################################
##############################################################################
#   Main Subject Tag, to be inserted
repl_tag = ""
#
##############################################################################
##############################################################################


replace_list = {
  
  ##############################################################################
  #keyword replacements go here
  ##############################################################################
  
  "":"",
  
  ##############################################################################
  ##############################################################################
                }

with open(file_path+"train_prompt.txt",'wt') as fi:
    fi.write("")  
with open(file_path+"kw_list.txt",'wt') as fi:
    fi.write("")
with open(file_path+"kw_repl_list.txt",'wt') as fi:
    fi.write("")

txt_files = [file_path+f for f in os.listdir(file_path[:-1:]) 
             if os.path.isfile(file_path+f) 
             and str(f[-(f[::-1].find('.')):]).strip().lower() in ["txt"]
             and f!="kw_list.txt" and f!="kw_repl_list.txt" and f!="train_prompt.txt"
             ]


def text_replace():
    for i_t in tqdm(range(0,len(txt_files)-1)):
        _=[]
        txt_uniq = []
        txt_data = ""
        txt_list=[]
        txt_write=""
        file = txt_files[i_t]
        with open(file,'rt') as fi:
            txt_data = fi.read()
        txt_data = "," + txt_data + ","
        txt_data = txt_data.replace(" ,",",").replace(", ",",").replace(",,",",")
        while txt_data.find(",,")!=-1:
            txt_data.replace(",,",",")
        for k in replace_list.keys():
            txt_data = txt_data.lower().replace(k,replace_list[k])
        while txt_data.find(',,')>0:
                txt_data = txt_data.replace(',,',',').replace(f" {repl_tag},","")
        txt_list = [x.strip().lower() for x in txt_data.split(',') if len(x.strip().lower())>2]
        txt_uniq = np.unique(np.array(txt_list)).tolist()
        _ = [all_tags.append(x) for x in txt_uniq]
        len(_)#TONY THE TIGER
        l_w = lambda x: str(x).capitalize() if x not in sw else str(x)
        txt_write = txt_write + ', '.join(w for w in [' '.join(l_w(s) for s in str(x).split(chr(32))) for x in txt_uniq])
        with open(file,'wt') as fi:
            fi.write(f"{repl_tag}" + txt_write)


def kw_list():
    txt_list = []
    tag_list = []
    txt_uniq = []
    txt_uniq = np.unique(np.array(all_tags))
    txt_list = [t for t in sorted([[all_tags.count(x), x] for x in txt_uniq.tolist()], key=lambda x: x[0], reverse=True)]
    txt_write = "\n".join(str(x).strip() for x in txt_list)
    with open(file_path+"kw_list.txt",'wt') as fi:
        fi.write(txt_write)
    kw_repl = '\n'.join(str(f'" {str(t).strip()},":"",') for t in np.unique(np.array(all_tags)))
    with open(file_path+"kw_repl_list.txt",'wt') as fi:
        fi.write(kw_repl)
    train_prompt = repl_tag + ', '.join(x[1] for x in txt_list[0:20:])
    train_prompt = str(train_prompt + str("\n") + repl_tag + ', '.join(str(x[1]).strip() for x in txt_list[0:20:]) + ', ' + ', '.join(x[1] for x in txt_list[20:25:1]))
    train_prompt = str(train_prompt + str("\n") + repl_tag + ', '.join(str(x[1]).strip() for x in txt_list[0:20:]) + ', ' + ', '.join(x[1] for x in txt_list[25:30:1]))
    train_prompt = str(train_prompt + str("\n") + repl_tag + ', '.join(str(x[1]).strip() for x in txt_list[0:20:]) + ', ' + ', '.join(x[1] for x in txt_list[35:40:1]))
    train_prompt = str(train_prompt + str("\n") + repl_tag + ', '.join(str(x[1]).strip() for x in txt_list[0:20:]) + ', ' + ', '.join(x[1] for x in txt_list[45:50:1]))
    train_prompt = str(train_prompt + str("\n") + repl_tag + ', '.join(str(x[1]).strip() for x in txt_list[0:20:]) + ', ' + ', '.join(x[1] for x in txt_list[55:60:1]))
    with open(file_path+"train_prompt.txt",'wt') as fi:
        fi.write(train_prompt)  


if __name__ == "__main__":
  text_replace()
  print(len(all_tags))
  kw_list()
