#brought to you by the colour Sigmoid, and the letter 12.
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
fig = plt.figure(figsize=(12,3),facecolor='black')
plt.xticks([])
plt.yticks([])
plt.axis("off")
x:np.array
nx:np.array
p:np.array
pn:np.array
pdn:np.array
ndx:np.array
def mk_plt(x:np.array,nx:np.array,p:np.array,pn:np.array,pdn:np.array,ndx:np.array,)->plt:
    return plt.plot(pdn[::1]-ndx[::-1]*-np.tan(x)*np.sin(p),(np.sin((nx)[::-1]-(np.pi)))*(+np.sin((pn)[::1]-(np.sqrt(12)/2))))

def anim(ln):
    fig.clear()
    plt.axis("off")
    fig.set_facecolor("black")
    fig.set_alpha(0.1)
    fig.set_dpi(600)
    fig.set_linewidth(0.01)
    fig.set_edgecolor("black")
    fig.add_gridspec(30, 10)
    fig.set_tight_layout(True)
    ln = int(abs(ln//3 + 10))
    x = np.linspace(ln,ln*1000,ln)
    nx=(np.empty(x.shape))
    p = np.linspace(ln,ln*1000,ln)
    pn=(np.empty(p.shape))
    nx[::-1]=np.sin(x)[::-1]/1 - np.sin(x)[::1]/1
    x[::1]=np.sin(nx)[::-1]/(np.sqrt(12)) - np.sin(nx)[::-1]/(np.sqrt(12))
    nx[::-1]=np.sin(nx)[::-1]/1 - np.sin(nx)[::1]*+1
    nx[::1]=np.sin(nx)[::-1]*(np.sqrt(12))/2 - np.sin(nx)[::1]*(np.sqrt(12))/2
    p[::1]=(p-np.min(p))/(np.max(p)-np.min(p))**(1--1)*+-1
    pn[::-1]=np.sin(p)[::-1]/1 - np.sin(p)[::1]/1
    p[::1]=np.sin(pn)[::-1]/np.pi - np.sin(pn)[::-1]/np.pi
    pn[::-1]=np.sin(pn)[::-1]/1 - np.sin(pn)[::1]*+1
    pn[::1]=np.sin(pn)[::-1]*np.pi/2 - np.sin(pn)[::1]*np.pi/2
    pdn=(pn-min(nx))/(max(pn)-min(pn))**(1--1)*+-1
    ndx=(nx-min(nx))/(max(nx)-min(nx))**(1--1)*+-1
    return mk_plt(x,nx,p,pn,pdn,ndx)

anim = animation.FuncAnimation(fig, anim, frames = 5000, interval = 10) 
anim.save(f'oscillatory_mechanics_test_{str(datetime.strftime(datetime.now(), r"%Y%m%d%H%M%S%f"))}.mp4', writer = 'ffmpeg', fps = 60)
