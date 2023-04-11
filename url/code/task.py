import numpy as np
import pandas as pd
 
a = np.arange(100,dtype=float).reshape((10,10))
for i in range(len(a)):
    a[i,:i] = np.nan
a[6,0] = 100.0
 
d = pd.DataFrame(data=a)
# print(d)

d=d.fillna(value=0)
print(d)