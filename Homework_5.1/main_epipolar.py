import numpy
import pylab
from PIL import Image
import camera
import func_epipolar

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

# Compute fundamental matrix
F = func_epipolar.compute_fundamental(x1,x2)

# Compute epipole
e = func_epipolar.compute_epipole(F)

# Show image1
pylab.figure()
pylab.imshow(im1)
for i in range(5):
    func_epipolar.plot_epipolar_line(im1,F,x2[:,i],e,False)
pylab.axis('off')

# Show image2
pylab.figure()
pylab.imshow(im2)
for i in range(5):
    pylab.plot(x2[0,i],x2[1,i],'o')
pylab.axis('off')

pylab.show()