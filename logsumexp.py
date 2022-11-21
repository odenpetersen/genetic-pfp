#!/usr/bin/env python
""" logsumexp ~ Su Boyd Candes, Diff eq modeling Nesterov accelerated gradient, 2014, p. 7

    Logsumexp( seed: int / RandomState / env SEED  / 0 )
"""

from __future__ import division
import os
import numpy as np

__version__ = "2015-01-31 jan  denis-bz-py at t-online.de"


# logsumexp = Logsumexp() below: logsumexp(x), logsumexp.gradient(x)

#...............................................................................
class Logsumexp( object ):
    __doc__ = globals()["__doc__"]

    def __init__( s, seed=None, m=1, n=1, rho=20 ):
        s.rho = rho
        s.__name__ = "logsumexp"
        s.reset( seed )
        s._initrandomAb( n, m )

    #...........................................................................
    def __call__( s, x ):
        ax_b = s._ax_b( x )
        exp = np.exp( ax_b / s.rho )
        return s.rho * np.log( np.sum( exp ))

    def gradient( s, x ):
        ax_b = s._ax_b( x )
        exp = np.exp( ax_b / s.rho )
        return s.A.T.dot( exp ) / np.sum( exp )

    def _ax_b( s, x ):
        x = np.asarray_chkfinite(x)
        if len(x) != s.A.shape[1]:
            s._initrandomAb( len(x) )
        return s.A.dot(x) - s.b

    def reset( s, seed ):
        if seed is None:
            seed = int( os.getenv( "SEED", 0 ))
        if not isinstance(seed, np.random.RandomState):
            seed = np.random.RandomState( seed=seed )
        s.randomstate = seed
        s._initrandomAb( 1, 1 )

    def _initrandomAb( s, n, m=0 ):
        """ init random A b on first call each size """
            # bug: calls 3d 4d 3d 4d ... todo: dict n -> A b
        if m == 0:
            m = 4 * n  # 200 x 50
        s.A = s.randomstate.normal( size=(m, n) )
        s.b = s.randomstate.normal( scale=np.sqrt(2), size=m )


logsumexp = Logsumexp()  # logsumexp(x), logsumexp.gradient(x)


#...............................................................................
if __name__ == "__main__":
    import sys

    def avmax( x ):
        absx = np.fabs(x)
        return "|av| %.3g  max %.3g" % (absx.mean(), absx.max())

#...............................................................................
    nn = [50]
    nseed = 3

        # to change these params in sh or ipython, run this.py  a=1  b=None  c=[3] ...
    for arg in sys.argv[1:]:
        exec( arg )

    np.set_printoptions( threshold=100, edgeitems=3, linewidth=120,
        formatter = dict( float = lambda x: "%.2g" % x ))  # float arrays %.2g

#...............................................................................
    for n in nn:
      for seed in range(nseed):
        print("\nlogsumexp( n %d  seed %s ) --" % (n, seed))
        logsumexp = Logsumexp( seed=seed )
        x = np.zeros(n)
        f = logsumexp( x )
        g = logsumexp.gradient( x )
        print("f(0): %.3g" % f)
        print("grad: %s  %s" % (avmax(g), g))
