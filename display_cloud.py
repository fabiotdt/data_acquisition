import open3d as o3d
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def get_pts(infile):
	data = np.loadtxt(infile, delimiter=' ')
	return data[12:,0], data[12:,1], data[12:,2] #returns X,Y,Z points skipping the first 12 lines
	
def plot_ply(infile):
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	x,y,z = get_pts(infile)
	print(x)
	ax.scatter(x, y, z, c='r', marker='o')
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	plt.show()	
	
if __name__ == '__main__':
	infile = '../zed_data/marg_9.ply'
	plot_ply(infile)
