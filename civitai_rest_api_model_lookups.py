import os
import ssl
import urllib.request
import urllib
from urllib.error import HTTPError, URLError
from tqdm import tqdm
import keyring
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed


API_TOKEN = keyring.get_password("civitai","api_token")


LORA_DIR = ""
LYCO_DIR = ""
SAVE_PATH = ""


LORA_FILES = []
LYCO_FILES = []


CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


URL0 = "https://civitai.com"

EP = [
    'creators',
    'imgs',
    'models',
    'model_id',
    'model_ver_id',
    'model_hash',
    'tags'
]

API_EP = {
    "creators":"/api/v1/creators",
    "imgs":"/api/v1/images",
    "models":"/api/v1/models",
    "model_id":"/api/v1/models/",
    "model_ver_id":"/api/v1/model-versions/:modelVersionId",
    "model_hash":"/api/v1/model-versions/by-hash/",
    "tags":"/api/v1/tags"
}

HEADERS  = {'User-Agent': "Mozilla/5.0 (Windows; U; Win98; en-US; rv:1.6) Gecko/20040206 Firefox/0.8",
                            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                            'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
                            'Accept-Encoding': 'none',
                            'Accept-Language': 'en-US,en;q=0.8',
                            'Connection': 'keep-alive',
                            "Content-Type": "application/json"}


def statbar(tot: int, desc: str)->tqdm:
    l_bar = "{desc}: {percentage:3.0f}%|"
    r_bar = "| {n_fmt}/{total_fmt} [elapsed: {elapsed} / Remaining: {remaining}] "
    bar = "{rate_fmt}{postfix}]"
    status_bar = tqdm(total=tot, desc=desc, bar_format=f"{l_bar}{bar}{r_bar}")
    return status_bar


def f_split(f: str) -> list:  #  0: "X:/basedir/" , 1: "filename" , 2: "ext"
    return list([f[:len(f)-(f[::-1].find(chr(47))):],
                f[len(f)-(f[::-1].find(chr(47))):(len(f))-1-len(f[-(f[::-1].find(".")):])],
                f[-(f[::-1].find(".")):],]
                for f in[f.replace(chr(92),chr(47))])[0]


def get_hash(file:str,t:bool)->None:
    with open(file,"rb") as fi:
        file_bytes = fi.read()
    h = str(hashlib.sha256(file_bytes).hexdigest())
    if t:
        LORA_FILES.append([file,h])
    else:
        LYCO_FILES.append([file,h])
    return 


def get_model_info_by_hash(hash:str):
    try:
        request = urllib.request.Request(URL0 + API_EP['model_hash'] + hash, None, headers=HEADERS)
        response = urllib.request.urlopen(request, context=CTX)
        url_resp = response.read()
        return (True, json.loads(url_resp))
    except HTTPError as e_err:
        return (False, e_err)
    except URLError as u_err:
        return (False, u_err)


def load_lylocora_files()->None:
    lof = [
        LORA_DIR+f for f 
        in os.listdir(LORA_DIR[:-1:]) 
        if os.path.isfile(LORA_DIR+f) 
        and f[-(f[::-1].find('.')):].lower() in ['ckpt','pt','safetensors']]
    with ThreadPoolExecutor(8) as executor:
        status_bar = statbar(int(len(lof)),"Hashing LORAs")
        futures = [executor.submit(get_hash, file, True) for file in lof]
        for _ in as_completed(futures):
            status_bar.update(n=1)
        status_bar.close()
    lyf = [
        LYCO_DIR+f for f 
        in os.listdir(LYCO_DIR[:-1:]) 
        if os.path.isfile(LYCO_DIR+f) 
        and f[-(f[::-1].find('.')):].lower() in ['ckpt','pt','safetensors']]
    with ThreadPoolExecutor(8) as executor:
        status_bar = statbar(int(len(lyf)),"Hashing LYCOs")
        futures = [executor.submit(get_hash, file, False) for file in lyf]
        for _ in as_completed(futures):
            status_bar.update(n=1)
        status_bar.close()

if __name__ == "__main__":

    load_lylocora_files()
    
    with open(SAVE_PATH+"lora_list.txt","wt") as fi:
        for x in LORA_FILES:
            fi.write(fr"{x[0]},{x[1]}{chr(10)}")
            
    with open(SAVE_PATH+"lyco_list.txt","wt") as fi:
        for x in LYCO_FILES:
            fi.write(fr"{x[0]},{x[1]}{chr(10)}")

    for LF in LORA_FILES + LYCO_FILES:
        try:
            res_bool, res = get_model_info_by_hash(LF[1])
            if res_bool:
                f=f_split(LF[0])
                with open(f"{SAVE_PATH}{f[1]}.json","wt",encoding='utf-8') as fi:
                    json.dump(res, fi, ensure_ascii=False, indent=4, allow_nan=True)
        except Exception as e:
            print(e)
            pass
