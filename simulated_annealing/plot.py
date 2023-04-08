import numpy as np
from matplotlib import pyplot as plt



obj_0999 = np.loadtxt('obj38_0.999.txt', dtype=int)
obj_09 = np.loadtxt('obj38_0.9.txt', dtype=int)
obj_07 = np.loadtxt('obj38_0.7.txt', dtype=int)
obj_log = np.loadtxt('obj38_log.txt', dtype=int)

obj_b1 = np.loadtxt('obj38_b1.txt', dtype=int)
obj_b2 = np.loadtxt('obj38_b2.txt', dtype=int)
obj_b3 = np.loadtxt('obj38_b3.txt', dtype=int)





temp_0999 = np.loadtxt('temp_0.999.txt', dtype=float)
temp_09 = np.loadtxt('temp_0.9.txt', dtype=int)
temp_07 = np.loadtxt('temp_0.7.txt', dtype=int)
temp_log = np.loadtxt('temp_log.txt', dtype=float)

temp_b1 = np.loadtxt('temp_b1.txt', dtype=float)
temp_b2 = np.loadtxt('temp_b2.txt', dtype=float)
temp_b3 = np.loadtxt('temp_b3.txt', dtype=float)





n1 = len(temp_0999)
n2 = len(temp_09)
n3 = len(temp_07)
n4 = len(temp_log)
n5 = len(temp_b2)
n6 = len(temp_b3)
n7 = len(temp_b1)



a1 = np.arange(0, n1)
a2 = np.arange(0, n2)
a3 = np.arange(0, n3)
a4 = np.arange(0, n4)
a5 = np.arange(0, n5)
a6 = np.arange(0, n6)
a7 = np.arange(0, n7)



plt.subplot(2, 1, 1)
# plt.plot(a1, temp_0999, "r-")
# plt.plot(a2, temp_09, "k-")
# plt.plot(a3, temp_07, "b-")
# plt.plot(a4, temp_log, "g-")
plt.plot(a7, temp_b1, "r-")
plt.plot(a5, temp_b2, "g-")
plt.plot(a6, temp_b3, "b-")



plt.title("Temperature V/S no of iterations")
plt.xlabel("no of iterations")
plt.ylabel("Temperature")
# plt.legend(["cooling rate:0.999  Final temp:" + str(temp_0999[-1]), "cooling rate:0.9 Final temp:" + str(temp_09[-1]), "cooling rate:0.7 Final temp:" + str(temp_07[-1])])
plt.legend(["Init Temp: 1500000  Final temp:" + str(temp_b1[-1]), "Init Temp: 150000 Final temp:" + str(temp_b2[-1]), "Init Temp: 1500 Final temp:" + str(temp_b3[-1])])

plt.subplot(2, 1, 2)
# plt.plot(a1, obj_0999, "r-")
# plt.plot(a2, obj_09, "k-")
# plt.plot(a3, obj_07, "b-")
# plt.plot(a4, obj_log, "g-")
plt.plot(a7, obj_b1, "r-")
plt.plot(a5, obj_b2, "g-")
plt.plot(a6, obj_b3, "b-")


plt.title("Objective Value V/S no of iterations")
plt.xlabel("no of iterations")
plt.ylabel("Objective Value")
plt.legend(["Init Temp: 1500000 Final obj:" + str(obj_b1[-1]), "Init Temp: 150000 Final obj:" + str(obj_b2[-1]), "Init Temp: 1500 Final obj:" + str(obj_b3[-1])])
plt.pause(0.0001)

plt.show()
