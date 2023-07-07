import pandas as pd
import numpy as np
import time
import ttcrpy.rgrid as rg
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#Most recent

file1 = open('new-line6-converted.txt', 'r')
#file1 = open('29-sept_fixed_inconsistency.dat.txt', 'r')
#file1 = open('4_october_columns_added.dat', 'r')
Lines = file1.readlines()

list_of_files = []
list_of_sx = []
list_of_sy = []
list_of_gx = []
list_of_gy = []
list_of_xs = []
list_of_ys = []
n=0
first_time = True
n_636 = 0

first_sx = 0
first_sy = 0
first_gx = 0
first_gy = 0
for line in Lines:
    content = line.strip()

    if content != '' and content[0] == 's' and first_time is True:
        #print(content)
        content_split = content.split(' ')
        #print(content_split)
        #content_split_by_equals = content_split('=')
        sx = content_split[0].split('=')[1]
        sy = content_split[1].split('=')[1]
        #gx = content_split[2].split('=')[1]
        #gx = content_split[2].split('=')[1]
        gx = float(sx) + 264 + (n * 12.5)
        gy = content_split[3].split('=')[1]
        #print(sx, sy, gx, gy)
        first_sx = float(sx)
        first_sy = float(sy)
        first_gx = float(gx)
        first_gy = float(gy)
        list_of_sx.append(float(0))
        list_of_sy.append(float(0))
        list_of_gx.append(float(0))
        list_of_gy.append(float(0))
        # list_of_xs.append(float(sx))
        # list_of_xs.append(float(gx))
        # list_of_ys.append(float(sy))
        # list_of_ys.append(float(gy))
        n +=1
        first_time = False

    elif content != '' and content[0] == 's' and first_time is False:
        # print(content)
        content_split = content.split(' ')
        # print(content_split)
        # content_split_by_equals = content_split('=')
        sx = float(content_split[0].split('=')[1])
        sy = float(content_split[1].split('=')[1])
        # gx = content_split[2].split('=')[1]
        # gx = content_split[2].split('=')[1]
        #new_sx = sx - first_sx
        gx = float(float(sx) + 264 + (n * 12.5))
        gy = float(content_split[3].split('=')[1])
        # print(sx, sy, gx, gy)
        #first_sx = sx
        #first_sy = sy
        #first_gx = gx
        #first_gy = gy
        list_of_sx.append(float(sx-first_sx))
        list_of_sy.append(float(sy-first_sy))
        list_of_gx.append(float(gx-first_gx))
        list_of_gy.append(float(gy-first_gy))
        # list_of_xs.append(float(sx))
        # list_of_xs.append(float(gx))
        # list_of_ys.append(float(sy))
        # list_of_ys.append(float(gy))
        n += 1
        # print(list_of_gx)
        # print(list_of_gy)
        #break


    if n%636 ==0:
        #print ("HEHE")
        n=0

    elif content != '' and content[0] == 't':
        content_split = content.split(' ')
        #print(content_split)
        trace = int(content_split[0].split('=')[1])


        if trace   == 636:
            #print(trace)
            n_636 +=1
            #break

print(n_636)
print("SMALL n", n)

#exit()
# numpy_sx = np.array(list_of_sx[: : 5])
# numpy_sy = np.array(list_of_sy[: : 5])
# numpy_gx = np.array(list_of_gx[: : 5])
# numpy_gy = np.array(list_of_gy[: : 5])
# numpy_xs = np.array(list_of_xs[: : 5])
# numpy_ys = np.array(list_of_ys[: : 5])


numpy_sx = np.array(list_of_sx)
numpy_length_sx = len(numpy_sx)
numpy_sy = np.zeros(numpy_length_sx)
#numpy_sy = np.array(list_of_sy)

numpy_gx = np.array(list_of_gx)
numpy_length_gx = len(numpy_gx)
numpy_gy = np.zeros(numpy_length_gx)
#numpy_gy = np.array(list_of_gy)

#intersect = np.intersect1d(numpy_sx, numpy_gx)
#print(intersect)

print("NUMPY SIZE:", numpy_sx.size)
#print(numpy_sy)

print("MIN SX:",np.min(numpy_sx))
print("MAX SX:",np.max(numpy_sx))
print("MIN SY:",np.min(numpy_sy))
print("MAX SY:",np.max(numpy_sy))
print("MIN GX:",np.min(numpy_gx))
print("MAX GX:",np.max(numpy_gx))
print("MIN GY:",np.min(numpy_gy))
print("MAX GY:",np.max(numpy_gy))

diff_x = np.max(numpy_sx) - np.min(numpy_sx)
diff_y = np.max(numpy_sy) - np.min(numpy_sy)
diff_x = diff_x*2
diff_y = diff_y*2

print("DIFFF X",diff_x)
print("DIFFF Y",diff_y)

#grid_x = np.arange(min(min(numpy_gx), min(numpy_sx)), max(max(numpy_gy)+10000, max(numpy_sy)+10000), 5000)
#grid_y = np.arange(min(min(numpy_gx), min(numpy_sx)), max(max(numpy_gy)+10000, max(numpy_sy)+10000), 5000)
#grid_y = np.arange(min(numpy_gx), max(numpy_gy)+10000, 5000)

#grid_x = np.arange(265294.0,5177423.0, 5000)
#grid_y = np.arange(265294.0,19152946.0, 5000)


#2361

initial_grid_number = 2351

#full_grid_begin = 636 * (2261)
full_grid_begin = 636 * (2161)
#full_grid_end   = 636 * (2361)
full_grid_end   = 636 * (2361)
full_grid_increment = full_grid_begin + 636


increment_in_int_x = 299
increment_in_floats_x = float(increment_in_int_x)

increment_in_int_y = 50
increment_in_floats_y = float(increment_in_int_y)

big_numpy_sx = numpy_sx[full_grid_begin:full_grid_end]
big_numpy_sy = numpy_sy[full_grid_begin:full_grid_end]
big_numpy_gx = numpy_gx[full_grid_begin:full_grid_end]
big_numpy_gy = numpy_gy[full_grid_begin:full_grid_end]


# grid_x = np.arange(265294.0,19152946.0+increment_in_floats, increment_in_int)
# grid_y = np.arange(265294.0,19152946.0+increment_in_floats, increment_in_int)

#min_grid = min(np.min(new_numpy_sx), np.min(new_numpy_sy), np.min(new_numpy_gx), np.min(new_numpy_gy))
#max_grid = max(np.max(new_numpy_sx), np.max(new_numpy_sy), np.max(new_numpy_gx), np.max(new_numpy_gy))


min_grid_x = min(np.min(big_numpy_sx), np.min(big_numpy_gx))
max_grid_x = max(np.max(big_numpy_sx), np.max(big_numpy_gx))

min_grid_y = min(np.min(big_numpy_sy), np.min(big_numpy_gy))
max_grid_y = max(np.max(big_numpy_sy), np.max(big_numpy_gy))



print(min_grid_x)
print(max_grid_x)
print(min_grid_y)
print(max_grid_y)

print("X DIFF:", (max_grid_x-min_grid_x))
print("Y DIFF:", (max_grid_y-min_grid_y))
#print("X pieces:", (max_grid_x-min_grid_x)/140) # for last 10
print("X pieces:", (max_grid_x-min_grid_x)/299) #for last 200
print("Y pieces:", (max_grid_y-min_grid_y)/100)
print("Y pieces MODIFIED:", (max_grid_y-min_grid_y+3000.0)/50)
print("JUST DEPTH:", 3000.0/50)


#exit()

print()
#min max for sy and gy are the same, so going with + 3000 depth
grid_x = np.arange(min_grid_x, max_grid_x + increment_in_floats_x, increment_in_int_x)
grid_y = np.arange(0.0, 3000.0 + increment_in_floats_y, increment_in_int_y)

print("GX:", len(grid_x))
# print(grid_x)
print("GY:", len(grid_y))

print("BIG GRID:",big_numpy_sx[0], big_numpy_sy[0])
# print(list_of_gx)
# print(np.max(list_of_gx))
# print(list_of_gy)
# exit()



# print(numpy_sx)

# fig = plt.figure()
# plt.scatter(numpy_sx, numpy_sy)
# plt.grid()
# plt.show()

grid = rg.Grid2d(grid_x, grid_y)

Vmin = 1500  # m/s
Vmax = 5500
Nx = len(grid_x) - 1
Ny = len(grid_y) - 1
print(Ny)

min_depth = 300  # metre (m)
max_depth = 3000

gradient = (Vmax - Vmin) / max_depth  # gradient or time at the maximum depth, as that will have the highest velocity

# fill array
slowness_y = np.empty((Ny,))
slowness_x = np.empty((Nx,))
print("SLOW:", len(slowness_y))
for n in range(Ny):
    z = 2 * int(grid_y[n] / 2) + 1  # manipulating the y coordinates according to example
    slowness_y[n] = 1.0 / (Vmin + z * gradient)
    print(z)
    print(slowness_y[n])
    # if (n >20):
    #   break
# repeat for all x & y locations

slowness_y = np.tile(slowness_y, Ny)
slowness_x = np.tile(slowness_x, Ny)
print("LL", len(slowness_y))

print(slowness_y)
# exit()
# Assign slowness to grid
grid.set_slowness(slowness_y)

#grid.to_vtk({'Velocity': 1. / slowness_y}, 'FBIG_inc_sources_' + str(loop_iterator) + 'example_r1')
grid.to_vtk({'Velocity': 1. / slowness_y}, 'FBIG_inc_sources_example_r1')

#max 2361
loop_counter = 200
loop_iterator = 0
#exit()
# begin = 0
# end = 636

#full_grid_begin = 636 * (2161)
full_grid_begin = 636 * (2161 + 90)

begin = full_grid_begin
end = full_grid_begin+636

#exit()
# first run 0-40
#second run 60-70
#third run
#for loop_iterator in range(0, loop_counter, 10):
for loop_iterator in range(90, loop_counter, 10):

    ##src is first cordinate
    new_numpy_sx = numpy_sx[begin:end]
    new_numpy_sy = numpy_sy[begin:end]
    new_numpy_gx = numpy_gx[begin:end]
    new_numpy_gy = numpy_gy[begin:end]
    src = np.array([[new_numpy_sx[0], new_numpy_sy[0]]])

    #mid_sx = int(numpy_sx.size / 2)
    #mid_sy = int(numpy_sy.size / 2)
    #print(mid_sx)
    print("SRC:", src)
    #print("MID:", numpy_sx[mid_sx], numpy_sy[mid_sy])

    #src_mid = np.array([[numpy_sx[mid_sx], numpy_sy[mid_sy]]])

    # src is last cordinate
    # src = np.array([[numpy_sx[-1], numpy_sy[-1]]])

    rcv = np.array([[new_numpy_gx, new_numpy_gy]])
    # src = np.array([numpy_sx, numpy_sy])
    # rcv = np.array([numpy_gx, numpy_gy])
    rcv = rcv.T
    rcv = rcv[:, :, 0]
    # print(rcv)
    print(src.shape)
    print(rcv.shape)
    print("RCV:", rcv)

    # exit()

    # first_source = src[0]
    # print(first_source.shape)
    ref = time.time()
    tt, rays = grid.raytrace(src, rcv, return_rays=True)
    compute_time1 = time.time() - ref

    print(compute_time1)
    # exit()

    plt.figure(figsize=(4.5, 3))
    plt.plot(tt, 'o')
    plt.xlabel('Receiver no', fontsize=14)
    plt.ylabel('Traveltime', fontsize=14)
    plt.savefig('FBIG_inc_'+str(loop_iterator)+'_tt_fsm.pdf', bbox_inches='tight')

    # Save raypaths
    grid.to_vtk({'raypaths for shot no 1': rays}, 'FBIG_inc_sources_'+str(loop_iterator)+'_grid_rays')

    begin =  begin + (636*10)
    end =  end + (636*10)

    # begin = 636 * (initial_grid_number+1)
    # end =   636 * (initial_grid_number+1)

    print("LOOP:", loop_iterator)
    #loop_iterator = loop_iterator + 10

    # tt_mid, rays_mid = grid.raytrace(src_mid, rcv, return_rays=True)
    # grid.to_vtk({'raypaths for shot no 1': rays_mid}, 'Fifth_inc_sources_mid_grid_rays')

    # Ignore SPM

    # SPM part
    # grid_spm = rg.Grid2d(grid_x, grid_y, method='SPM')
    #
    # # Compute traveltimes and raypaths
    # ref = time.time()
    # tt, rays = grid_spm.raytrace(src, rcv, slowness=slowness, return_rays=True)
    # compute_time2 = time.time() - ref
    #
    # # Check traveltime values
    # plt.figure(figsize=(4.5,3))
    # plt.plot(tt, 'o')
    # plt.xlabel('Receiver no', fontsize=14)
    # plt.ylabel('Traveltime', fontsize=14)
    # plt.savefig('Fifth_all_sources_spm.pdf', bbox_inches='tight')
    #
    # # Save raypaths
    # grid.to_vtk({'raypaths for shot no 1': rays}, 'Fifth_all_sources_rays_spm')


