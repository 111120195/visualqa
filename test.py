import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def l_matrix(d, p):
	l = np.empty(shape=(d, p))
	for k in range(d):
		for j in range(p):
			l[k][j] = (1 - j / p) - (k / d) * (1 - 2 * j / p)
	return l


class A(object):
	def __init__(self):
		super(A).__init__()
		self.x = 1

	def foo(self):
		self.y = 2


if __name__ == '__main__':
	a = A()
	print(a.y)
	# p = l_matrix(10, 6)
	# sns.heatmap(l_matrix(100, 60))
	# plt.show()
