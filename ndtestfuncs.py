#!/usr/bin/env python
""" some n-dimensional test functions for optimization in Python.
    Example:

    import numpy as np
    from ... import ndtestfuncs  # this file, ndtestfuncs.py

    funcnames = "ackley ... zakharov"
    dim = 8
    x0 = e.g. np.zeros(dim)
    for func in ndtestfuncs.getfuncs( funcnames ):
        fmin, xmin = myoptimizer( func, x0 ... )
        # calls func( x ) with x a 1d numpy array or array-like of any dim >= 2

These are the n-dim Matlab functions by A. Hedar (2005), translated to Python-numpy.
http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO.htm
    ackley.m dp.m griew.m levy.m mich.m perm.m powell.m power.m
    rast.m rosen.m schw.m sphere.m sum2.m trid.m zakh.m
    + ellipse nesterov powellsincos

--------------------------------------------------------------------------------
    All functions appearing in this work are fictitious;
    any resemblance to real-world functions, living or dead, is purely coincidental.
--------------------------------------------------------------------------------

Notes
-----

The values of most of these functions increase as O(dim) or O(dim^2),
so the convergence curves for dim 5 10 20 ... are not comparable,
and `ftol_abs` depends on func() scaling.
Better would be to scale function values to min 1, max 100 in all dimensions.
Similarly, `xtol_abs` depends on `x` scaling;
`x` should be scaled to -1 .. 1 in all dimensions.

Results from any optimizer depend of course on `ftol_abs xtol_abs maxeval ...`
plus hidden or derived parameters, e.g. BOBYQA rho.
Methods like Nelder-Mead that track sets of points, starting with `x0 + initstep I`,
are sensitive to `initstep` / restart initsteps.

Some functions have many local minima or saddle points (more in higher dimensions ?),
making the final fmin very sensitive to the starting x0.
*Always* look at a few points near a purported xmin -- cf. nearmin.py .


See also
--------

http://en.wikipedia.org/wiki/Test_functions_for_optimization  -- 2d plots
http://www.scipy.org/NumPy_for_Matlab_Users

nlopt/test/... runs and plots BOBYQA PRAXIS SBPLX ... on these ndtestfuncs
    from several random startpoints, in several dimensions
Nd-testfuncs-python.md
F = Funcmon(func): wrap func() to monitor and plot F.fmem F.xmem F.cost
"""
    # zillions of papers and methods for derivative-free / noisy optimization


#...............................................................................
from __future__ import division
import numpy as np
from numpy import abs, cos, exp, mean, pi, prod, sin, sqrt, sum

try:
    from opt.testfuncs.powellsincos import Powellsincos
except ImportError:
    Powellsincos = None

__version__ = "2014-10-28 oct denis-bz-py@t-online.de"


#...............................................................................
def ackley( x, a=20, b=0.2, c=2*pi ):
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    n = len(x)
    s1 = sum( x**2 )
    s2 = sum( cos( c * x ))
    return -a*exp( -b*sqrt( s1 / n )) - exp( s2 / n ) + a + exp(1)

#...............................................................................
def dixonprice( x ):  # dp.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange( 2, n+1 )
    x2 = 2 * x**2
    return sum( j * (x2[1:] - x[:-1]) **2 ) + (x[0] - 1) **2

#...............................................................................
def griewank( x, fr=4000 ):
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange( 1., n+1 )
    s = sum( x**2 )
    p = prod( cos( x / sqrt(j) ))
    return s/fr - p + 1

#...............................................................................
def levy( x ):
    x = np.asarray_chkfinite(x)
    n = len(x)
    z = 1 + (x - 1) / 4
    return (sin( pi * z[0] )**2
        + sum( (z[:-1] - 1)**2 * (1 + 10 * sin( pi * z[:-1] + 1 )**2 ))
        +       (z[-1] - 1)**2 * (1 + sin( 2 * pi * z[-1] )**2 ))

#...............................................................................
michalewicz_m = 2  # orig 10: ^20 tiny, underflow

def michalewicz( x ):  # mich.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange( 1., n+1 )
    return - sum( sin(x) * sin( j * x**2 / pi ) ** (2 * michalewicz_m) )

#...............................................................................
def perm( x, b=.5 ):
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange( 1., n+1 )
    xbyj = np.fabs(x) / j
    return mean([ mean( (j**k + b) * (xbyj ** k - 1) ) **2
            for k in j/n ])
    # original overflows at n=100 --
    # return sum([ sum( (j**k + b) * ((x / j) ** k - 1) ) **2
    #       for k in j ])

#...............................................................................
def powell( x ):
    x = np.asarray_chkfinite(x)
    n = len(x)
    n4 = ((n + 3) // 4) * 4
    if n < n4:
        x = np.append( x, np.zeros( n4 - n ))
    x = x.reshape(( 4, -1 ))  # 4 rows: x[4i-3] [4i-2] [4i-1] [4i]
    f = np.empty_like( x )
    f[0] = x[0] + 10 * x[1]
    f[1] = sqrt(5) * (x[2] - x[3])
    f[2] = (x[1] - 2 * x[2]) **2
    f[3] = sqrt(10) * (x[0] - x[3]) **2
    return sum( f**2 )

#...............................................................................
def powersum( x, b=[8,18,44,114] ):  # power.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    s = 0
    for k in range( 1, n+1 ):
        bk = b[ min( k - 1, len(b) - 1 )]  # ?
        s += (sum( x**k ) - bk) **2
    return s

#...............................................................................
def rastrigin( x ):  # rast.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    return 10*n + sum( x**2 - 10 * cos( 2 * pi * x ))

#...............................................................................
def rosenbrock( x ):  # rosen.m
    """ http://en.wikipedia.org/wiki/Rosenbrock_function """
        # a sum of squares, so LevMar (scipy.optimize.leastsq) is pretty good
    x = np.asarray_chkfinite(x)
    x0 = x[:-1]
    x1 = x[1:]
    return (sum( (1 - x0) **2 )
        + 100 * sum( (x1 - x0**2) **2 ))

#...............................................................................
def schwefel( x ):  # schw.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    return 418.9829*n - sum( x * sin( sqrt( abs( x ))))

#...............................................................................
def sphere( x ):
    x = np.asarray_chkfinite(x)
    return sum( x**2 )

#...............................................................................
def sum2( x ):
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange( 1., n+1 )
    return sum( j * x**2 )

#...............................................................................
def trid( x ):
    x = np.asarray_chkfinite(x)
    return sum( (x - 1) **2 ) - sum( x[:-1] * x[1:] )

#...............................................................................
def zakharov( x ):  # zakh.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange( 1., n+1 )
    s2 = sum( j * x ) / 2
    return sum( x**2 ) + s2**2 + s2**4

#...............................................................................
    # not in Hedar --

def ellipse( x ):
    x = np.asarray_chkfinite(x)
    return mean( (1 - x) **2 )  + 100 * mean( np.diff(x) **2 )

#...............................................................................
def nesterov( x ):
    """ Nesterov's nonsmooth Chebyshev-Rosenbrock function, Overton 2011 variant 2 """
    x = np.asarray_chkfinite(x)
    x0 = x[:-1]
    x1 = x[1:]
    return abs( 1 - x[0] ) / 4 \
        + sum( abs( x1 - 2*abs(x0) + 1 ))


#-------------------------------------------------------------------------------
allfuncs = [
    ackley,
    dixonprice,
    ellipse,
    griewank,
    levy,
    michalewicz,  # min < 0
    nesterov,
    perm,
    powell,
    # powellsincos,  # many local mins
    powersum,
    rastrigin,
    rosenbrock,
    schwefel,  # many local mins
    sphere,
    sum2,
    trid,  # min < 0
    zakharov,
    ]

if Powellsincos is not None:  # try import
    _powellsincos = {}  # dim -> func
    def powellsincos( x ):
        x = np.asarray_chkfinite(x)
        n = len(x)
        if n not in _powellsincos:
            _powellsincos[n] = Powellsincos( dim=n )
        return _powellsincos[n]( x )

    allfuncs.append( powellsincos )

#...............................................................................
allfuncnames = " ".join( sorted([ f.__name__ for f in allfuncs ]))
name_to_func = { f.__name__ : f for f in allfuncs }

    # bounds from Hedar, used for starting random_in_box too --
    # getbounds evals ["-dim", "dim"]
ackley._bounds       = [-15, 30]
dixonprice._bounds   = [-10, 10]
ellipse._bounds      =  [-2, 2]
griewank._bounds     = [-600, 600]
levy._bounds         = [-10, 10]
michalewicz._bounds  = [0, pi]
nesterov._bounds     = [-2, 2]
perm._bounds         = ["-dim", "dim"]  # min at [1 2 .. n]
powell._bounds       = [-4, 5]  # min at tile [3 -1 0 1]
powellsincos._bounds = [ "-20*pi*dim", "20*pi*dim"]
powersum._bounds     = [0, "dim"]  # 4d min at [1 2 3 4]
rastrigin._bounds    = [-5.12, 5.12]
rosenbrock._bounds   = [-2.4, 2.4]  # wikipedia
schwefel._bounds     = [-500, 500]
sphere._bounds       = [-5.12, 5.12]
sum2._bounds         = [-10, 10]
trid._bounds         = ["-dim**2", "dim**2"]  # fmin -50 6d, -200 10d
zakharov._bounds     = [-5, 10]

#...............................................................................
def getfuncs( names ):
    """ for f in getfuncs( "a b ..." ):
            y = f( x )
    """
    if names == "*":
        return allfuncs
    funclist = []
    for nm in names.split():
        if nm not in name_to_func:
            raise ValueError( "getfuncs( \"%s\" ) not found" % names )
        funclist.append( name_to_func[nm] )
    return funclist

def getbounds( funcname, dim ):
    """ "ackley" or ackley -> [-15, 30] """
    funcname = getattr( funcname, "__name__", funcname )
    func = getfuncs( funcname )[0]
    b = func._bounds[:]
    if isinstance( b[0], basestring ):  b[0] = eval( b[0] )
    if isinstance( b[1], basestring ):  b[1] = eval( b[1] )
    return b

def funcnames_minus( minus="powersum sphere sum2 trid zakharov " ):
    return " ".join( sorted([ f.__name__ for f in allfuncs
            if f.__name__ not in minus.split() ]))


#-------------------------------------------------------------------------------
if __name__ == "__main__":  # standalone test --
    import sys

    dims = [2, 4, 10, 100]
    nstep = 11  # 11: 0 .1 .2 .. 1
    seed = 0

        # to change these params in sh or ipython, run this.py  a=1  b=None  c=[3] ...
    for arg in sys.argv[1:]:
        exec( arg )
    np.set_printoptions( 1, threshold=100, edgeitems=10, linewidth=120, suppress=True,
        formatter = dict( float = lambda x: "%.3g" % x ))  # float arrays %.3g
    np.random.seed(seed)

    #...........................................................................
        # each func( line [0 0 0 ...] .. upper bound ) --
        # cmp matlab, anyone ?
    steps = np.linspace( 0, 1, nstep )
    for dim in dims:
        print "\n# ndtestfuncs dim %d  along the diagonal 0 .. high corner --" % dim

        for func in allfuncs:
            funcname = func.__name__
            hibound = getbounds( funcname, dim )[1]
            corner = hibound * np.ones(dim)

            Y = np.array([ func( step * corner ) for step in steps ])
            jmin = Y.argmin()
            print "%-12s %dd  0 .. %4.3g: min %6.3g  at %4.2g \tY %s" % (
                    funcname, dim, hibound, Y[jmin], steps[jmin], Y )

    # see plot-ndtestfuncs.py
