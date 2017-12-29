import numpy as np
from PIL import Image
import scipy.misc as smp
import utils

k_vals = [10]
			
def compress():
	images, _ = utils.read_from_csv(False, './data/fashion-mnist_train.csv')
	for k in k_vals:
		print 'k:', k
		compressed = np.zeros(images.shape)
		for i in range(0, images.shape[0]):
			if i % 1000 == 0:
				print i
			if i == 1000:
				break
			img = np.reshape(images[i], (28,28))
			u, d, v = np.linalg.svd(img, full_matrices=True, compute_uv=True)
			d = np.diag(d)

			uk = u[:,:k] # first k columns of U (m x m becomes m x k)
			dk = d[:k,:k] # first k rows and columns of d (m x n becomes k x k)
			vk = v[:k,] # first k rows of Vt (n x n becomes k x n)
		
			k_approx = np.matmul(np.matmul(uk, dk), vk)
			k_approx_rescaled = (k_approx - np.min(k_approx))/(np.max(k_approx) - np.min(k_approx)) # rescale values to fit between 0 and 1
			shrunk = np.uint8(k_approx_rescaled*255)
			#print(shrunk.shape)
			#im = Image.fromarray(shrunk)
			#im.show()
			to_save = np.reshape(shrunk, (1, 784))
			print to_save
			compressed[i:] = to_save
			
		np.savetxt(str(k) + '.csv', compressed, delimiter=',')


compress()
