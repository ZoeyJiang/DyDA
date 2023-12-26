import pickle
import os
# import math
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
import numpy as np
# x, y = np.random.rand(10), np.random.rand(10)
# z = (np.random.rand(9000000)+np.linspace(0,1, 9000000)).reshape(3000, 3000)
# plt.imshow(z+10, extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)),
#     cmap=cm.hot, norm=LogNorm())
# plt.colorbar()
# plt.show()

# matplotlib.use('TkAgg')
file="/Users/yeshuai/Documents/GraduateDesign/papers2020-10-22/ADVENT/entrophy_map_pic/npy/kits/wrong/case_00082.npz"
f=np.load(file)#[3,262,122,122]
f=f['softmax'][:,::-1,...]
print(f.shape)
for fi in range(f.shape[1]):
    print(fi)
    # print(f[:,fi,...].shape)
    res=np.zeros(f.shape[-2:]).astype('float32') #[122,122]
    slide=f[:,fi,...]
    for x in range(slide.shape[-2]):
        for y in range(slide.shape[-1]):
            for z in range(3):
                if(slide[z,x,y]==0):
                    slide[z, x, y] = 0.01
                res[x,y]+=-slide[z,x,y]*np.log(slide[z,x,y])
    print(res)
    print(res.max())
    print(res.min())
    # plt.imshow(res,vmax=1.15,vmin=0)
    plt.imshow(res)
    plt.colorbar()
    plt.show()



