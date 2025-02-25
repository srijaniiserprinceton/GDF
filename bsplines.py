import numpy as np
import bspline
import bspline.splinelab as splinelab
import matplotlib.pyplot as plt; plt.ion()

class bsplines:
    def __init__(self, knots, p=3):
        self.knots = knots
        self.p = p
        self.gen_bsp_basis()

    def gen_bsp_basis(self):
        self.knots_aug = splinelab.augknt(self.knots, self.p)  # add endpoint repeats as appropriate for spline order p
        self.B = bspline.Bspline(self.knots_aug, self.p)            # create spline basis of order p on knots k

    def eval_bsp_basis(self, x):
        # bsp = np.array([self.B(i) for i in x]).T # <------ these are the basis functions
        bsp = self.B.collmat(x).T                  # <------ these are the basis functions
        # making a small adjustment in the right most B-spline
        bsp[-1,-1] = bsp[0,0]

        return bsp

if __name__=='__main__':
    p = 3              # order of spline (as-is; 3 = cubic)
    nknots = 20        # number of knots to generate (here endpoints count only once)
    knots = np.linspace(0,1,nknots)  # create a knot vector without endpoint repeats

    bsp = bsplines(knots, p)
    
    x_min = np.min(bsp.knots)
    x_max = np.max(bsp.knots)
    x = np.linspace(x_min, x_max, num=1000)
    bsp_basis = bsp.eval_bsp_basis(x)

    # plotting the B-spline components
    for b in bsp_basis:
        plt.plot(x,b)
    plt.title('B-spline basis elements')
