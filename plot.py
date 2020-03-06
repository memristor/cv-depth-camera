import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from matplotlib._png import read_png
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d

fig = plt.figure()
ax = Axes3D(fig)
ax.set_proj_type('ortho')

def table(table_size, texture=False):
  global ax
  w, h = table_size
  # table_edges = np.array([[-w/2,-h/2,0],[-w/2,h/2,0],[w/2,h/2,0],[w/2,-h/2,0]])
  # table_verts = [list(zip(table_edges[:,0],table_edges[:,1],table_edges[:,2]))]
  # table_poly = Poly3DCollection(table_verts)
  # table_poly.set_alpha(0.2)
  # table_poly.set_edgecolor('black')
  # table_poly.set_linewidth(5)
  # table_poly.set_linestyle('solid')
  # ax.add_collection3d(table_poly)
  # ax.grid(False)
  # ax.add_collection3d(table_poly)
  corner_nw = np.array([-w/2,h/2,0])
  corner_ne = np.array([w/2,h/2,0])
  corner_sw = np.array([-w/2,-h/2,0])
  corner_se = np.array([w/2,-h/2,0])
  line(corner_nw, corner_ne)
  line(corner_ne, corner_se)
  line(corner_se, corner_sw)
  line(corner_sw, corner_nw)
  ax.auto_scale_xyz([-w/2, w/2], [-w/2, w/2], [-w/2, w/2])
  if texture:
    table_image = read_png('table.png')
    x, y = np.ogrid[
      -table_image.shape[0]/2:table_image.shape[0]/2,
      -table_image.shape[1]/2:table_image.shape[1]/2]
    ax.plot_surface(
      x, y, np.array([[0]]),
      facecolors=table_image,
      rstride=1, cstride=1, alpha=1, linewidth=0
    )

def point(point, color='black', label=None):
  global ax
  if label is not None:
    text(label, point, color=color)
  ax.scatter3D(
    point[0], point[1], point[2],
    color=color
  )

def line(point_a, point_b, color='black', label=None):
  global ax
  if label is not None:
    text(label, (point_a + point_b) / 2, color=color)
  ax.plot(
    np.array([point_a[0], point_b[0]]),
    np.array([point_a[1], point_b[1]]),
    color=color
  )

def vector(vector, point=np.array([0,0,0]), scale=1, color='black', label=None):
  global ax
  if label is not None:
    text(label, point, color=color)
  ax.quiver(
    point[0], point[1], point[2],
    vector[0] * scale, vector[1] * scale, vector[2] * scale,
    color=color
  )

def points(points, color='black'):
  global ax
  ax.scatter3D(
    points[:, 0], points[:, 1], points[:, 2],
    color=color
  )

def vectors(vectors, point=np.array([0,0,0]), scale=1, color='black'):
  global ax
  ax.quiver(
    point[0], point[1], point[2],
    vectors[:, 0] * scale, vectors[:, 1] * scale, vectors[:, 2] * scale,
    color=color
  )

def text(text, point, color='black'):
  global ax
  ax.text(point[0] + 0.3, point[1] + 0.3, point[2] + 0.3, text, color=color)

def cylinder(point, radius=0.36, height=1.15, resolution=15, color='black'):
  global ax

  x_center = point[0]
  y_center = point[1]
  elevation=point[2]

  x = np.linspace(x_center-radius, x_center+radius, resolution)
  z = np.linspace(elevation, elevation+height, resolution)
  X, Z = np.meshgrid(x, z)

  Y = np.sqrt(radius**2 - (X - x_center)**2) + y_center # Pythagorean theorem

  ax.plot_surface(X, Y, Z, linewidth=0, color=color)
  ax.plot_surface(X, (2*y_center-Y), Z, linewidth=0, color=color)

  # floor = Circle((x_center, y_center), radius, color=color)
  # ax.add_patch(floor)
  # art3d.pathpatch_2d_to_3d(floor, z=elevation, zdir="z")

  ceiling = Circle((x_center, y_center), radius, color=color, alpha=1)
  ax.add_patch(ceiling)
  art3d.pathpatch_2d_to_3d(ceiling, z=elevation+height, zdir="z")

def show():
  global plt
  # plt.axis('off')
  plt.show()