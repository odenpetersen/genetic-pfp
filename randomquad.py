""" random quadratic:
    randomquad(x): |ax - b|^2 / 2
    randomquad.gradient(x): A x - B, A random posdef, B normal
        A x - B = a' (ax - b)
        a = d r: d diagonal, r reflect / flip
    ~ Su Boyd Candes, Diff eq modeling Nesterov accelerated gradient, 2014, p. 7

    Randomquad( seed: int / RandomState / env SEED  / 0,
            eigmin=.001, eigmax=1: eigenvalues linspace( eigmin, eigmax, n )
"""
    # how realistic is completely random b, ~ what *real* problems ?
    # x = A^-1 b = sum ((b . e_i) / lambda_i) e_i  is very sensitive to b

from __future__ import division
import os
import numpy as np

__version__ = "2015-01-29 jan  denis-bz-py at t-online.de"


# randomquad = Randomquad() below: randomquad(x), randomquad.gradient(x)

#...............................................................................
class Randomquad( object ):
    __doc__ = globals()["__doc__"]

    def __init__( s, seed=None, eigmin=.001, eigmax=1 ):
        s.eigmin, s.eigmax = eigmin, eigmax
        s.__name__ = "randomquad"
        s.reset( seed )

    #...........................................................................
    def __call__( s, x ):
        """ |a x - b|^2 / 2 """
        x = np.asarray_chkfinite(x)
        if len(x) != len(s.u):
            s._initrandom( len(x) )
        ax_b = s._ax_b( x )
        return ax_b .dot(ax_b) / 2  # = (x' A x  - 2 B x  + b'b) / 2  >= 0

    def gradient( s, x ):
        """ A x - B = a' (ax - b), a = d r """
        x = np.asarray_chkfinite(x)
        if len(x) != len(s.u):
            s._initrandom( len(x) )
        ax_b = s._ax_b( x )
        ax_b *= s.d
        return s._flip( ax_b )

    def _flip( s, x ):
        """ flip / reflect I - 2 u u': u -> - u, u \perp fixed """
            # Householder transform
        return x - 2 * s.u .dot(x) * s.u

    def _ax_b( s, x ):
        """ a x - b, A = a'a """
        y = s._flip( x )
        y *= s.d
        y -= s.b
        return y

    def _A( s, n ):
        """ -> A = a'a """
        if n != len(s.u):
            s._initrandom( n )
        a = (np.eye(n) - 2 * np.outer( s.u, s.u )) * s.d[:,np.newaxis]
        return a.T.dot(a)

    def reset( s, seed ):
        if seed is None:
            seed = int( os.getenv( "SEED", 0 ))
        if not isinstance(seed, np.random.RandomState):
            seed = np.random.RandomState( seed=seed )
        s.randomstate = seed
        s.u = []  # if len(x) != len(u): _initrandom

    def _initrandom( s, n ):
        """ init random u d B b on first call each size """
            # bug: calls 3d 4d 3d 4d ... todo: dict n -> u d B b
        s.u = s.randomstate.normal( size=n )
        s.u /= np.sqrt( s.u .dot(s.u) )
        s.d = np.sqrt( np.linspace( s.eigmin, s.eigmax, n ))  # eigvals d^2
        s.B = s.randomstate.normal( scale=5, size=n )  # grad A x - B = a' (a x - b)
        s.b = s._flip( s.B )
        s.b /= s.d
        s.xmin = s._flip( s.b / s.d )


randomquad = Randomquad()  # randomquad(x), randomquad.gradient(x)


#...............................................................................
if __name__ == "__main__":
    import sys

    def avmax( x ):
        absx = np.fabs(x)
        return "|av| %.3g  max %.3g" % (absx.mean(), absx.max())

#...............................................................................
    nn = 2 ** np.arange( 3, 3+1 )  # 4, 9+1 )
    nseed = 1

        # to change these params in sh or ipython, run this.py  a=1  b=None  c=[3] ...
    for arg in sys.argv[1:]:
        exec( arg )

    np.set_printoptions( threshold=100, edgeitems=3, linewidth=120,
        formatter = dict( float = lambda x: "%.2g" % x ))  # float arrays %.2g

    def eig( A ):
        print "A:\n", A * 100
        evals, evecs = np.linalg.eigh( A )
        print "evals:", evals
        print "evecs:\n", evecs * 100
        return evals, evecs

#...............................................................................
    As = []
    for n in nn:
      for seed in range(nseed):
        print "\nrandomquad( n %d  seed %d ) --" % (n, seed)
        randomquad = Randomquad( seed=seed )
        x = np.zeros(n)
        f = randomquad( x )
        g = randomquad.gradient( x )
        print "f(0): %.3g" % f
        print "grad: %s  %s" % (avmax(g), g)
        xmin = randomquad.xmin
        print "xmin: %s  %s" % (avmax(xmin), xmin)  # 128: max 6600
        fxmin = randomquad( xmin )
        print "f(xmin): %.3g " % fxmin
        As.append( randomquad._A( n ))

    eig( As[0] )
