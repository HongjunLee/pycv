import numpy

# Compute the least square triangulation of a point pair
def triangulate_point(x1,x2,P1,P2):
        
    M = numpy.zeros((6,6))
    M[:3,:4] = P1
    M[3:,:4] = P2
    M[:3,4] = -x1
    M[3:,5] = -x2

    U,S,V = numpy.linalg.svd(M)
    X = V[-1,:4]

    return X / X[3]

# Write triangulation function for two view points in x1, x2
def triangulate(x1,x2,P1,P2):
        
    n = x1.shape[1]
    
    X = [ triangulate_point(x1[:,i],x2[:,i],P1,P2) for i in range(n)]
    return numpy.array(X).T