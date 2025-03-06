import numpy as np

a = np.array([1, 2, 3, 4, 5])

mask = (a >= 3)
a[mask] = 10

a[~mask] = 2*a[~mask]

print(a)
