import func_triangulate
import numpy
import camera
from PIL import Image

# Read right, left stereo image
im1 = numpy.array(Image.open('stereo/left.jpg'))
im2 = numpy.array(Image.open('stereo/right.jpg'))

# Read 2D point in image from already save-file
points2D = [numpy.loadtxt('2D/00'+str(i+1)+'.corners').T for i in range(3)]

# Read real 3D point from already save-file
points3D = numpy.loadtxt('3D/p3d').T

# Read corresponding point between two images from already save-file
corr = numpy.genfromtxt('2D/nview-corners',dtype='int',missing='*')

# Read camera matrix
P = [camera.Camera(numpy.loadtxt('2D/00'+str(i+1)+'.P')) for i in range(3)]

# index for correspond points in two images
ndx = (corr[:,0]>=0) & (corr[:,1]>=0)

x1 = points2D[0][:,corr[ndx,0]]
x1 = numpy.vstack( (x1,numpy.ones(x1.shape[1])) )

x2 = points2D[1][:,corr[ndx,1]]
x2 = numpy.vstack( (x2,numpy.ones(x2.shape[1])) )

# xt is a true 3D point
xt = points3D[:,ndx]
xt = numpy.vstack( (xt,numpy.ones(xt.shape[1])) )

# xest is a estimation of 3D point
xest = func_triangulate.triangulate(x1,x2,P[0].P,P[1].P)

print 'Estimation 3D point'
print xest[:,:3]
print 'True 3D point'
print xt[:,:3]