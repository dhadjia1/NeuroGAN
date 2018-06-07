
import sys
import os
import numpy as np
from collections import namedtuple
from sklearn.decomposition import PCA

Point = namedtuple('Point', ['int_label', 'typ', 'xyz', 'radius', 'parent'])
PC_Transform = namedtuple('PC_Transform', ['PointCloud', 'PointCloud_reference', 'rot'])

class CellCloud(object):
    def __init__(self, group=None, identifier=None, points=None):
        self.group = group
        self.identifier = identifier
        self.points = points
      

    def rotate(self, reference_cloud, rotation, typ=3):
        self.reference_cloud = reference_cloud
        self.rotation = rotation
        

        c, s = np.cos(rotation), np.sin(rotation)
        self.rotation_matrix = np.array( ((c, -s), (s,c)) )

        new_points = []
        for point in self.points:
            if point[1] != typ:
                continue
            (x,y,z) = point[2]
            new_point = np.dot(self.rotation_matrix, np.array([x, y]))
            new_points.append(new_point)
        new_points = np.asarray(new_points)
        self.rotated_xy_center = new_points - np.mean(new_points, axis=0)
       

        self.rotated_ub, self.rotated_lb = self.bounds(self.rotated_xy_center,axis=0)
        self.rotated_eigenvalues = self.eigenvalues
        self.rotated_eigenvectors = np.matmul(self.rotation_matrix, self.eigenvectors)

    def bounds(self, xy, axis=0):
        return np.max(xy, axis=axis), np.min(xy, axis=axis)

    def setup(self, typ):
        xpts, ypts = [], []
        for point in self.points:
            if point[1] == typ:
                (x,y,z) = point[2]
                xpts.append(x), ypts.append(y)
        return np.array([xpts, ypts]).T
        
    def center_pc(self, xy=None, mod=True, typ=3):
        if xy is None:
            xy = self.setup(typ)
        xy_center = xy - np.mean(xy, axis=0)
        if mod:
            self.xy_center = xy_center
        return xy_center
    
    def pca_bounding(self,typ=3):
 
        xpts, ypts = [], []
        for point in self.points:
            if point[1] == typ:
                (x,y,z) = point[2]
                xpts.append(x), ypts.append(y)
        xy = np.array([xpts, ypts])
        xy = np.transpose(xy)
        
        pca = PCA(n_components=2)
        xy_transform = pca.fit_transform(xy)
        xy_center = xy - np.mean(xy, axis=0)
        
        self.xy_center = xy_center
        self.eigenvalues = pca.explained_variance_
        self.eigenvectors = pca.components_

        self.ub, self.lb = self.bounds(self.xy_center, axis=0)

def get_rotation_from_eigenvectors(ev1, ev2):

    def get_rotation(x, y):
        return np.dot(x,y) / ( np.sqrt(np.dot(x,x)) * np.sqrt(np.dot(y,y)) )

    x1, x2 = ev1[0,0], ev1[1,0]
    y1, y2 = ev2[0,0], ev2[1,0]
    z1 = get_rotation(np.array([x1, x2]), np.array([y1, y2]))

    x1, x2 = ev1[0,1], ev1[1,1]
    y1, y2 = ev2[0,1], ev2[1,1]
    z2 = get_rotation(np.array([x1, x2]), np.array([y1, y2]))
   
    return (np.arccos(z1), np.arccos(z2))
    
def read_file(lab, fn, fpath):
    fy = open(fpath, 'r')
    cells = []
    for line in fy.readlines():
        if '#' in line:
            continue
        line = line.strip('\n').split()
        if line == []:
            continue
        int_label = int(line[0])
        typ = int(line[1])
        x, y, z, = float(line[2]), float(line[3]), float(line[4])
        r, p = float(line[5]), float(line[6])
        cells.append(Point(int_label, typ, (x,y,z), r, p))
    fy.close()

    return CellCloud(group=lab, identifier=fn, points=cells)
    



def read_swc_files(database_loc):
    clouds = []
    labs = [name for name in os.listdir(database_loc) if os.path.isdir(database_loc)]
    abs_path = [database_loc + '/' + lab for lab in labs]
    number_files = 0
    for (i, path) in enumerate(abs_path):
        lab = labs[i]
        path = path + '/CNG version'
        for fn in os.listdir(path):
            if fn.endswith(".swc"):
                number_files += 1
                fpath = path + '/' + fn
                try:
                    clouds.append(read_file(lab, fn, fpath))
                except:
                    continue
    cloud_dictionary = {}
    for cloud in clouds:
        if cloud.group in cloud_dictionary.keys():
            cloud_dictionary[cloud.group].append(cloud)
        else:
            cloud_dictionary[cloud.group] = [cloud]
    return cloud_dictionary, labs, number_files

def reach(cloud_dictionary):
    xul, yul, xll, yll = [], [], [], []
    for _, clist in cloud_dictionary.items():
        for pc in clist:
            xy_center = pc.center_pc(mod=False)
            ub, lb = pc.bounds(xy_center)
            xu, yu = ub
            xl, yl = lb
            xul.append(xu), yul.append(yu), xll.append(xl), yll.append(yl)

    xumax, yumax = max(xul), max(yul)
    xlmin, ylmin = min(xll), min(yll)
    
    return xumax, yumax, xlmin, ylmin

def coord_to_img(coordinates, cache=None):
    # https://stackoverflow.com/questions/12670736/how-to-convert-python-list-of-points-to-numpy-image-array
    
    x, y = [int(i[0]) for i in coordinates], [int(i[1]) for i in coordinates]
    if len(x) == 0 or len(y) == 0:
        return None, 0
    max_x, max_y = max(x), max(y)
    min_x, min_y = min(x), min(y)

    x[:] = [xa - min_x for xa in x]
    max_x = max_x - min_x
    y[:] = [ya - min_y for ya in y]
    max_y = max_y - min_y
       
    if cache is not None:
        max_x, max_y = cache
        max_x, max_y = int(max_x), int(max_y)
    image = np.zeros((max_y + 1, max_x + 1))

    for i in range(len(coordinates)):
        image[max_y - y[i], x[i]] = 1
    return image, 1

# Development, pls ignore
# if __name__ == '__main__':
    # import time

    # loc = '/home/dhadjia1/soltesz-lab/ca1/test/dendrite_generation/database'
    # print('...reading files in directory')
    # tic = time.time()
    # cloud_dictionary, lab_keys = read_swc_files(loc)
    # num_clouds = 0
    # num_points = 0
    # #for k,v in cloud_dictionary.iteritems():
    # #    num_clouds += len(v)
    # #    for cloud in v:
    # #        num_points += len(cloud.points)

    # #toc = time.time()
    # #elapsed = float(toc - tic)
    # #print('Reading %d files with a total of %d points was completed in %0.3f seconds' % (num_clouds, num_points, elapsed))

    # COI_one = cloud_dictionary[lab_keys[0]][0]
    # COI_one.pca_bounding()


    # COI_two = cloud_dictionary[lab_keys[0]][1]
    # COI_two.pca_bounding()

    # ev1, ev2 = COI_one.eigenvectors, COI_two.eigenvectors
    # (a1, a2) = get_rotation_from_eigenvectors(ev1, ev2)
    # assert( (a1 - a2) <1e-5 or (np.pi - a2 - a1) < 1e-5)

    # PCT = PC_Transform(COI_two, COI_one, a1)
    # PCT[0].rotate(PCT[1], PCT[2])

    # fig1, ax1 = plot_bounds(PCT[0].reference_cloud.ub, PCT[0].reference_cloud.lb, first=True, show=False)
    # plot_pc_with_eigenvector(PCT[0].reference_cloud.xy_center, PCT[0].reference_cloud.eigenvalues, PCT[0].reference_cloud.eigenvectors, fig=fig1, ax=ax1, first=False, show=False)

    # fig2, ax2 = plot_bounds(PCT[0].ub, PCT[0].lb, first=True, show=False)
    # plot_pc_with_eigenvector(PCT[0].xy_center, PCT[0].eigenvalues, PCT[0].eigenvectors, fig=fig2, ax=ax2, first=False, show=False)

    # ax1.set_title('PC1')
    # ax2.set_title('PC2')

    # fig3, ax3 = plot_bounds(PCT[0].rotated_ub, PCT[0].rotated_lb, first=True, show=False)
    # ax3.set_title('PC2 - rotated')
    # plot_pc_with_eigenvector(PCT[0].rotated_xy_center, PCT[0].rotated_eigenvalues, PCT[0].rotated_eigenvectors, fig=fig3, ax=ax3, first=False, show=True)


    

    
        
         
