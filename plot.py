import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

fig = plt.figure()
ax = Axes3D(fig)

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

def point(point, color='black'):
  global ax
  ax.scatter3D(
    point[0], point[1], point[2],
    color=color
  )

def vector(vector, point=np.array([0,0,0]), color='black'):
  global ax
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

def show():
  global plt
  plt.show()