import numpy
import pylab

# Compute fundamental matrix
# x1, x2 = corresponding points (3*n arrays)
def compute_fundamental(x1,x2):

    n = x1.shape[1]
    
    A = numpy.zeros((n,9))
    for i in range(n):
        A[i] = [x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i],
                x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i]*x2[2,i],
                x1[2,i]*x2[0,i], x1[2,i]*x2[1,i], x1[2,i]*x2[2,i] ]
            
    # compute linear least square solution
    U,S,V = numpy.linalg.svd(A)
    F = V[-1].reshape(3,3)

    # constrain F
    # make rank 2 by zeroing out last singular value
    U,S,V = numpy.linalg.svd(F)
    S[2] = 0
    F = numpy.dot(U,numpy.dot(numpy.diag(S),V))
    
    return F/F[2,2]

# Compute epipole by using fundamental matrix F
def compute_epipole(F):
      
    U,S,V = numpy.linalg.svd(F)
    e = V[-1]
    return e/e[2]
    
# Plot epipolar line on the original image
def plot_epipolar_line(im,F,x,epipole=None,show_epipole=True):
    
    m,n = im.shape[:2]
    line = numpy.dot(F,x)
    
    t = numpy.linspace(0,n,100)
    lt = numpy.array([(line[2]+line[0]*tt)/(-line[1]) for tt in t])

    ndx = (lt>=0) & (lt<m) 
    pylab.plot(t[ndx],lt[ndx],linewidth=2)
    
    if show_epipole:
        if epipole is None:
            epipole = compute_epipole(F)
        pylab.plot(epipole[0]/epipole[2],epipole[1]/epipole[2],'r*')