import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from matplotlib._png import read_png

# table_image = read_png('table.png')

fig = plt.figure()
ax = Axes3D(fig)
ax.set_proj_type('ortho')

def table():
  global ax
  w = 5
  h = 3
  table_edges = np.array([[-w,-h,0],[-w,h,0],[w,h,0],[w,-h,0]])
  table_verts = [list(zip(table_edges[:,0],table_edges[:,1],table_edges[:,2]))]
  table_poly = Poly3DCollection(table_verts)
  table_poly.set_alpha(0.3)
  table_poly.set_edgecolor('black')
  table_poly.set_linewidth(5)
  table_poly.set_linestyle('solid')
  ax.add_collection3d(table_poly)
  ax.auto_scale_xyz([-w, w], [-w, w], [-w + h, w + h])
  # ax.add_collection3d(table_poly)
  # x, y = np.ogrid[
  #   -table_image.shape[0]/2:table_image.shape[0]/2,
  #   -table_image.shape[1]/2:table_image.shape[1]/2]
  # ax.plot_surface(x, y, np.array([[-1]]), facecolors=table_image, rstride=2, cstride=2, alpha=0.5)
  # ax.auto_scale_xyz(
  #   [-table_image.shape[0], table_image.shape[0]],
  #   [-table_image.shape[0], -table_image.shape[0]],
  #   [-table_image.shape[0], table_image.shape[0]]
  # )
  # ax.plot_surface(x, y, np.array([[-1]]), facecolors=table_image,  rstride=8, cstride=8)

def point(point, color='black', label=None):
  global ax
  if label is not None:
    text(label, point, color=color)
  ax.scatter3D(
    point[0], point[1], point[2],
    color=color
  )

def vector(vector, point=np.array([0,0,0]), color='black', label=None):
  global ax
  if label is not None:
    text(label, point, color=color)
  ax.quiver(
    point[0], point[1], point[2],
    vector[0], vector[1], vector[2],
    color=color
  )

def points(points, color='black'):
  global ax
  ax.scatter3D(
    points[:, 0], points[:, 1], points[:, 2],
    color=color
  )

def vectors(vectors, point=np.array([0,0,0]), color='black'):
  global ax
  ax.quiver(
    point[0], point[1], point[2],
    vectors[:, 0], vectors[:, 1], vectors[:, 2],
    color=color
  )

def text(text, point, color='black'):
  ax.text(point[0] + 0.1, point[1] + 0.1, point[2] + 0.1, text, color=color)

def show():
  global plt
  plt.show()