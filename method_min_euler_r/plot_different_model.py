import numpy as np
import matplotlib.pyplot as plt
import os

path = os.path.dirname(__file__)

#8
data_8 = np.load(path+"\\save\\data_8X8.npy")
FB_8 = data_8[0]
utility_8 = data_8[1]

#16
data_16 = np.load(path+"\\save\\data_16X16.npy")
FB_16 = data_16[0]
utility_16 = data_16[1]

#32
data_32 = np.load(path+"\\save\\data_32X32.npy")
FB_32 = data_32[0]
utility_32 = data_32[1]

#64
data_64 = np.load(path+"\\save\\data_64X64.npy")
FB_64 = data_64[0]
utility_64 = data_64[1]

len_x = list(range(len(utility_8)))

plt.subplot(1,2,1)
plt.plot(len_x, FB_8,label = "8X8",color = "red")
plt.plot(len_x, FB_16,label = "16X16",color = "blue")
plt.plot(len_x, FB_32,label = "32X32",color = "green")
plt.plot(len_x, FB_64,label = "64X64",color = "yellow")
plt.legend()
plt.xlabel("train iter")
plt.ylabel("FB loss")

plt.subplot(1,2,2)
plt.plot(len_x, utility_8,label = "8X8",color = "red")
plt.plot(len_x, utility_16,label = "16X16",color = "blue")
plt.plot(len_x, utility_32,label = "32X32",color = "green")
plt.plot(len_x, utility_64,label = "64X64",color = "yellow")
plt.legend()
plt.xlabel("train iter")
plt.ylabel("utility")
plt.show()