import numpy as np

x = np.random.randn(10,1,28,29)
print(x.shape)
out = x.reshape(-1,10)
print(out.shape)

print(x.T.shape)