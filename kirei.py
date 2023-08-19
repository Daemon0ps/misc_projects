import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(1,144,1440)
nx=(np.empty(x.shape))
p = np.linspace(1,144,1440)
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
plt.plot(pdn[::1]-ndx[::-1]*-np.tan(x)*np.sin(p),(np.sin((nx)[::-1]-(np.pi)))*(+np.sin((pn)[::1]-(np.sqrt(12)/2))))
plt.show()
x = np.linspace(-12,144,12**2*3)
nx=(np.empty(x.shape))
p = np.linspace(144,-12,12**2*3)
pn=(np.empty(p.shape))
nx[::-1]=np.sin(x)[::-1]/1 - np.sin(x)[::1]/1
x[::1]=np.sin(nx)[::-1]/(np.sqrt(12)) - np.sin(nx)[::-1]/(np.sqrt(12))/2
nx[::-1]=np.sin(nx)[::-1]/1 - np.sin(nx)[::1]*+1
nx[::1]=np.sin(nx)[::-1]*(np.sqrt(12))/2 - np.sin(nx)[::1]*(np.sqrt(12))/2
pn[::-1]=np.sin(p)[::-1]/1 - np.sin(p)[::1]/1
p[::1]=np.sin(pn)[::-1]/np.pi - np.sin(pn)[::-1]/np.pi/2
pn[::-1]=np.sin(pn)[::-1]/1 - np.sin(pn)[::1]*+1
plt.plot(pn[::1]-nx[::-1]-np.tan(x)*-np.sin(p),np.sin((nx)[::-1]/(np.pi))+np.sin((pn)[::1]/(np.sqrt(12)/2)))
plt.show()
nx=np.sin((nx)[::-1]/np.sqrt(12))*-np.sin((nx)[::1]/np.sqrt(12))
pn=np.sin((pn)[::-1]/np.pi)*-np.sin((pn)[::1]/np.pi)
x = np.linspace(1,144,1440)
nx=(np.empty(x.shape))
p = np.linspace(1,144,1440)
pn=(np.empty(p.shape))
x=(x-min(x))/(max(x)-min(x))**(1--1)*+-1
nx[::-1]=np.sin(x)[::-1]/1 - np.sin(x)[::1]/1
x[::1]=np.sin(nx)[::-1]/(np.sqrt(12)) - np.sin(nx)[::-1]/(np.sqrt(12))
nx[::-1]=np.sin(nx)[::-1]/1 - np.sin(nx)[::1]*+1
nx[::1]=np.sin(nx)[::-1]*(np.sqrt(12))/2 - np.sin(nx)[::1]*(np.sqrt(12))/2
pn[::-1]=np.sin(p)[::-1]/1 - np.sin(p)[::1]/1
p[::1]=np.sin(pn)[::-1]/np.pi - np.sin(pn)[::-1]/np.pi
pn[::-1]=np.sin(pn)[::-1]/1 - np.sin(pn)[::1]*+1
plt.plot(pn[::1]-nx[::-1]*-np.tan(np.cos(x))*np.sin(p),np.sin((nx)[::-1]/(np.pi))*+np.sin((pn)[::1]/(np.sqrt(12)/2)))
plt.show()
