import numpy as np

a = [1, 2, 3, 4, 5]
print(np.pad(a, (3, 3), 'constant', constant_values=(4, 6)))
#[4 4 4 1 2 3 4 5 6 6 6]

print(np.pad(a, (4, 2), 'edge'))
#[1 1 1 1 1 2 3 4 5 5 5 5 5]