import numpy as np
import matplotlib.pyplot as plt


x = np.stack(np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100)), axis=-1)

input = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
values = np.array([0, 7, 2, 6])

dist = np.abs(x[:,:,None,:] - input[None, None, :, :])
weights = np.prod(1 - dist, axis=-1)

output = np.sum(weights * values, axis=-1)

plt.imshow(output)
plt.colorbar()
plt.show()
