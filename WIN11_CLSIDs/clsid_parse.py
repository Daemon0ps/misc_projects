#just prepping for subprocess calls from SimpleGui. And I am practicing my readability formatting.
import codecs
import unicodedata
clsid_b_read = []
clsid_1 = []
clsid_1_str = []
clsid_2 = []
clsid_2_str = []

with open('./clsid_b.txt','rb') as fi: # Open as bytes
    clsid_b_read = codecs.decode(unicodedata.normalize('NFKD', codecs.decode(fi.read())).encode('ascii', 'ignore')).split(r'|')

clsid_1 = str(
            clsid_b_read[0]
            ).replace(chr(92),chr(47)
            ).replace(chr(13)+chr(10),chr(10)
            ).split(chr(10))

clsid_1_str = str(
                clsid_b_read[1]
                ).replace(chr(92),chr(47)
                ).replace(chr(13)+chr(10),chr(10)
                ).split(chr(10))
                
clsid_2 = str(
            clsid_b_read[2]
            ).replace(chr(92),chr(47)
            ).replace(chr(13)+chr(10),chr(10)
            ).split(chr(10))
                
clsid_2_str = str(
                clsid_b_read[3]
                ).replace(chr(92),chr(47)
                ).replace(chr(13)+chr(10),chr(10)
                ).split(chr(10))

clsid_1_map = list(
                map(
                    lambda x: x[:(x[::1].find('}')+1)],
                    clsid_1))

clsid_all=[
    [clsid_1_str[i],clsid_1[i]] 
    for i,x in enumerate(clsid_1)
    if clsid_1[i] not in clsid_2
    ] + [
    [clsid_2_str[i],clsid_2[i]] 
    for i,x in enumerate(clsid_2)
    if clsid_2[i] not in clsid_1_map]

clsid_all = list(map(
    lambda i: [clsid_all[i][0],clsid_all[i][1]]
                if clsid_all[i] not in clsid_1_map 
                else [
                    clsid_all[i][0],
                    clsid_1[
                        clsid_1_map.index(clsid_all[i])]],
                [i for i,_ in enumerate(clsid_all)]))

clsid_write =   f"{chr(34)}{chr(10)}".join(
                r'\t%SystemRoot%/explorer.exe "shell:::'.join(
                s for s in i)
                for i in [x 
                for x in clsid_all])+chr(34)

with open('./clsid_shell_cmds.txt','wb')as fi:
    fi.write(
            codecs.encode(
                str(clsid_write.replace(
                    chr(47),chr(92))
            ),encoding='utf-8'))


#                                                    how i originally wrote this, because, payload limits in discord give me bad habits.
# with open('./clsid_b.txt','rb') as fi:
#     clsid_b_read = codecs.decode(unicodedata.normalize('NFKD', codecs.decode(fi.read())).encode('ascii', 'ignore')).split(r'|')
# clsid_1=str(clsid_b_read[0]).replace(chr(92),chr(47)).replace(chr(13)+chr(10),chr(10)).split(chr(10))
# clsid_1_str=str(clsid_b_read[1]).replace(chr(92),chr(47)).replace(chr(13)+chr(10),chr(10)).split(chr(10))
# clsid_2=str(clsid_b_read[2]).replace(chr(92),chr(47)).replace(chr(13)+chr(10),chr(10)).split(chr(10))
# clsid_2_str=str(clsid_b_read[3]).replace(chr(92),chr(47)).replace(chr(13)+chr(10),chr(10)).split(chr(10))
# clsid_all=[[clsid_1_str[i],clsid_1[i]]for i,x in enumerate(clsid_1)if clsid_1[i]not in clsid_2]+[[clsid_2_str[i],clsid_2[i]]for i,x in enumerate(clsid_2)if clsid_2[i]not in clsid_1_map]
# clsid_all=list(map(lambda i:[clsid_all[i][0],clsid_all[i][1]]if clsid_all[i]not in clsid_1_map else [clsid_all[i][0],clsid_1[clsid_1_map.index(clsid_all[i])]],[i for i,_ in enumerate(clsid_all)]))
# clsid_write=f"{chr(34)}{chr(10)}".join(r'\t%SystemRoot%/explorer.exe "shell:::'.join(s for s in i)for i in [x for x in clsid_all])+chr(34)
# with open('./clsid_shell_cmds.txt','wb')as fi:
#     fi.write(codecs.encode(str(clsid_write.replace(chr(47),chr(92))),encoding='utf-8'))
