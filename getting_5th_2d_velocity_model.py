import time
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

import gmsh

# plt.rcdefaults()
# plt.style.use('seaborn-paper')
# plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

from ttcrpy.tmesh import Mesh2d

gmsh.initialize()

gmsh.clear()
mesh_size = 5

mesh_size2 = 0.2*mesh_size

# points for the surface
p1 = gmsh.model.geo.addPoint(8.00, 0.0, 3.00, meshSize=mesh_size) # ground zero values
p2 = gmsh.model.geo.addPoint(8.70, 0.0, 2.61, meshSize=mesh_size)
p3 = gmsh.model.geo.addPoint(21.70, 0.0, 2.57, meshSize=mesh_size)
p4 = gmsh.model.geo.addPoint(26.45, 0.0, 2.12, meshSize=mesh_size)
p5 = gmsh.model.geo.addPoint(28.20, 0.0, 2.33, meshSize=mesh_size)
p6 = gmsh.model.geo.addPoint(32.45, 0.0, 1.9, meshSize=mesh_size)
p7 = gmsh.model.geo.addPoint(34.95, 0.0, 2.07, meshSize=mesh_size)
p8 = gmsh.model.geo.addPoint(38.45, 0.0, 1.75, meshSize=mesh_size)
p9 = gmsh.model.geo.addPoint(39.70, 0.0, 2.02, meshSize=mesh_size)
p10 = gmsh.model.geo.addPoint(42.95, 0.0, 1.62, meshSize=mesh_size)
p11 = gmsh.model.geo.addPoint(49.70, 0.0, 2.18, meshSize=mesh_size)
p12 = gmsh.model.geo.addPoint(56.70, 0.0, 1.89, meshSize=mesh_size)
p13 = gmsh.model.geo.addPoint(65.45, 0.0, 1.87, meshSize=mesh_size)
p14 = gmsh.model.geo.addPoint(74.95, 0.0, 1.02, meshSize=mesh_size)
p15 = gmsh.model.geo.addPoint(86.95, 0.0, 0.76, meshSize=mesh_size)
p16 = gmsh.model.geo.addPoint(91.95, 0.0, 0.89, meshSize=mesh_size)
p17 = gmsh.model.geo.addPoint(105.95, 0.0, 0.18, meshSize=mesh_size)
p18 = gmsh.model.geo.addPoint(125.95, 0.0, 0.10, meshSize=mesh_size)

# points for the interface between the layers, we use a smaller
# mesh_size to get a denser distribution along the interface

#Yellow layer below

p19 = gmsh.model.geo.addPoint(13.20, 0.0, 2.80, meshSize=mesh_size2)
p20 = gmsh.model.geo.addPoint(21.45, 0.0, 2.83, meshSize=mesh_size2)
p21 = gmsh.model.geo.addPoint(26.45, 0.0, 2.60, meshSize=mesh_size2)
p22 = gmsh.model.geo.addPoint(28.20, 0.0, 2.85, meshSize=mesh_size2)
p23 = gmsh.model.geo.addPoint(30.95, 0.0, 2.58, meshSize=mesh_size2)
p24 = gmsh.model.geo.addPoint(47.95, 0.0, 2.83, meshSize=mesh_size2)
p25 = gmsh.model.geo.addPoint(55.70, 0.0, 2.84, meshSize=mesh_size2)
p26 = gmsh.model.geo.addPoint(65.70, 0.0, 2.84, meshSize=mesh_size2)
p27 = gmsh.model.geo.addPoint(87.70, 0.0, 2.81, meshSize=mesh_size2)
p28 = gmsh.model.geo.addPoint(105.95, 0.0, 2.77, meshSize=mesh_size2)
p29 = gmsh.model.geo.addPoint(125.95, 0.0, 2.78, meshSize=mesh_size2)


# bottom example
p30 = gmsh.model.geo.addPoint(8.00, 0.0, 3.00, meshSize=mesh_size2)
p31 = gmsh.model.geo.addPoint(125.95, 0.0, 3.00, meshSize=mesh_size2)
#p11 = gmsh.model.geo.addPoint(15.0, 0.0, 6.0, meshSize=mesh_size)
#p12 = gmsh.model.geo.addPoint(0.0, 0.0, 6.0, meshSize=mesh_size)

# Get curved surface using BSplines
surf_tag = gmsh.model.geo.addBSpline([p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18], tag=10)
gmsh.model.geo.addLine(p18, p29, tag=11)
refl_tag = gmsh.model.geo.addBSpline([p29, p28, p27, p26, p25, p24, p23, p22, p21, p20, p19], tag=12)
gmsh.model.geo.addLine(p19, p1, tag=13)

gmsh.model.geo.addLine(p29, p31, tag=14)
gmsh.model.geo.addLine(p31, p30, tag=15)
gmsh.model.geo.addLine(p30, p19, tag=16)

gmsh.model.geo.addCurveLoop([10, 11, 12, 13], tag=21)
gmsh.model.geo.addCurveLoop([-12, 14, 15, 16], tag=22)

gmsh.model.geo.addPlaneSurface([21], tag=31)
gmsh.model.geo.addPlaneSurface([22], tag=32)

gmsh.model.geo.synchronize()



# create physical entities to be able to retrieve node
# coordinates at the surface & interface, and to assign
# velocity
pl1 = gmsh.model.addPhysicalGroup(1, [surf_tag])
pl2 = gmsh.model.addPhysicalGroup(1, [refl_tag])

ps1 = gmsh.model.addPhysicalGroup(2, [31])
ps2 = gmsh.model.addPhysicalGroup(2, [32])



gmsh.model.setPhysicalName(1, pl1, "surface")
gmsh.model.setPhysicalName(1, pl2, "reflector")
gmsh.model.setPhysicalName(2, ps1, "layer_1")
gmsh.model.setPhysicalName(2, ps2, "layer_2")


# We can then generate a 2D mesh...
gmsh.model.mesh.generate(2)



slowness = []
triangles = []

surface = []  # node coordinates at the surface
reflector = []  # node coordinates at the interface

for dim, tag in gmsh.model.getEntities():

    if dim == 1:
        # get the nodes on the BSplines
        elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim, tag)
        physicalTags = gmsh.model.getPhysicalGroupsForEntity(dim, tag)
        if len(physicalTags) == 0:
            continue
        name = gmsh.model.getPhysicalName(dim, physicalTags[0])

        if name == 'surface':
            for tag in np.unique(elemNodeTags[0]):
                node = gmsh.model.mesh.getNode(tag)
                surface.append(node[0])
        elif name == 'reflector':
            # get coordinates of the nodes on the interface
            for tag in np.unique(elemNodeTags[0]):
                node = gmsh.model.mesh.getNode(tag)
                reflector.append(node[0])

    elif dim == 2:
        # assign velocity
        elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim, tag)
        physicalTags = gmsh.model.getPhysicalGroupsForEntity(dim, tag)
        name = gmsh.model.getPhysicalName(dim, physicalTags[0])

        for n in range(len(elemTags[0])):
            t = elemNodeTags[0][3 * n:(3 * n + 3)]
            triangles.append(t)
            print("T", t)

            if name == "layer_1":
                slowness.append(1/3201)  # velocity of 1
                #slowness.append(1.0)  # velocity of 1
            else:
                slowness.append(1/3623)  # velocity of 3
                #slowness.append(0.333)  # velocity of 1



help(gmsh.model.mesh.getNode)

slowness = np.array(slowness)  # convert to numpy array
triangles = np.array(triangles)
surface = np.array(surface)
reflector = np.array(reflector)

uniqueTags = np.unique(triangles)  # we want to store the nodes only once

equiv = np.empty((int(1 + uniqueTags.max()),))
nodes = []
for n, tag in enumerate(uniqueTags):
    equiv[tag] = n
    node = gmsh.model.mesh.getNode(tag)
    nodes.append(node[0])

for n1 in range(triangles.shape[0]):
    for n2 in range(triangles.shape[1]):
        triangles[n1, n2] = equiv[triangles[n1, n2]]  # change the tag for the corresponding index

nodes = np.array(nodes)  # convert to numpy array
nodes = np.c_[nodes[:, 0], nodes[:, 2]]  # keep only x & z

# sort coordinates (for later plotting)
ind = np.argsort(surface[:, 0])
surface = surface[ind, :]
ind = np.argsort(reflector[:, 0])
reflector = reflector[ind, :]

gmsh.finalize()  # we're done with gmsh



print(surface)

#exit()


V = 1./slowness  # velocity is more intuitive than slowness

fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(111)

tpc = ax.tripcolor(nodes[:, 0], nodes[:, 1], triangles, V, cmap='coolwarm', edgecolors='w')
ax.plot(surface[:, 0], surface[:, 2]-0.1, 'kv', label='Receiver')
ax.legend(fontsize=12)
cbar = plt.colorbar(tpc, ax=ax)
cbar.ax.set_ylabel('Velocity', fontsize=14)

ax.invert_yaxis()
#ax.set_aspect('equal', 'box')

plt.xlabel('Distance', fontsize=14)
plt.ylabel('Depth', fontsize=14)
plt.tight_layout()



plt.savefig('5th_shot_new_mmodel_mesh.pdf', bbox_inches='tight')
plt.show()