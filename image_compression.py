import numpy as np
from PIL import Image
import scipy.misc as smp
import utils

k_vals = [1, 5, 10, 20]

def compress():
	images, _ = utils.read_from_csv(False, './data/fashion-mnist_train.csv')
	print images.shape
	img = np.reshape(images[0], (28,28))
	print img.shape
"""
	u, d, v = np.linalg.svd(img, full_matrices=True, compute_uv=True)
	d = np.diag(d)

	for k in k_vals:
		uk = u[:,:k] # first k columns of U (m x m becomes m x k)
		dk = d[:k,:k] # first k rows and columns of d (m x n becomes k x k)
		vk = v[:k,] # first k rows of Vt (n x n becomes k x n)
		
		print uk.shape
		print dk.shape
		print vk.shape

		k_approx = np.matmul(np.matmul(uk, dk), vk)
		k_approx_rescaled = (k_approx - np.min(k_approx))/(np.max(k_approx) - np.min(k_approx)) # rescale values to fit between 0 and 1
		im = Image.fromarray(np.uint8(k_approx_rescaled*255))
		im.show()"""


compress()