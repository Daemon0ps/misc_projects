import sys
import ssl
import urllib.request
import urllib
from urllib.error import HTTPError
from time import sleep
from random import randint
from tqdm import tqdm

save_path = "./test/"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows; U; Win98; en-US; rv:1.6) Gecko/20040206 Firefox/0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Charset": "ISO-8859-1,utf-8;q=0.7,*;q=0.3",
    "Accept-Encoding": "none",
    "Accept-Language": "en-US,en;q=0.8",
    "Connection": "keep-alive",
}
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# open a yearbook, click to open the first yearbook image in a new tab, get the base_url from the image displayed, then put this address in here

BASE_URL = "https://yb.cmcdn.com/yearbooks/x/x/x/x/xxxxxxxxxxxxxxxxxxxxxxxxxxxx/1100/"


def yb_img(i):
    try:
        sleep(randint(1, 5))
        req_url = BASE_URL + str(i).zfill(4) + ".jpg"
        request = urllib.request.Request(req_url, None, headers=headers)
        response = urllib.request.urlopen(request, context=ctx)
        url_file = response.read()
        with open(save_path + str(i).zfill(4) + ".jpg", "wb") as fi:
            fi.write(url_file)
    except KeyboardInterrupt:
        sys.exit()
    except HTTPError as h_err:
        if h_err.find("404"):
            sys.exit()
    except Exception as e:
        print(e)
        pass


if __name__ == "__main__":
    try:
        status_bar = tqdm(desc="Yearbook Images")
        for i in range(1, 1000):
            yb_img(i)
            status_bar.update(n=1)
    except KeyboardInterrupt:
        sys.exit()
    except Exception as e:
        print(e)
        pass
