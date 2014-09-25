N-dimensional test functions for optimization, in Python.
---------------------------------------------------------

These are the n-dim Matlab functions by A. Hedar (2005), translated to Python-numpy.  
http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO.htm  
    ackley dp griew levy mich perm powell power rast rosen schw sphere sum2 trid zakh .m
    + ellipse nesterov powellsincos

http://imgur.com/Iu9H0jk
is a plot of 3 no-derivative algorithms, [NLOpt](http://ab-initio.mit.edu/wiki/index.php/NLopt) LN_BOBYQA LN_PRAXIS LN_SBPLX,
on these functions, in 8d, from 10 random start points.
This plot shows, yet again, that

- comparing methods -- which is "best" -- is *really* difficult
- random effects, here the 10 random start points x0, can easily mask method differences.  
(A rule of thumb: [Wendel's theorem](http://en.wikipedia.org/wiki/Wendel%27s_theorem)
suggests that one should take at least 2*dim random x0.)


#### Ideas wanted

- better ways to plot methods / treatments across many test cases
- how to find test functions similar to a given real one --
    for optimization, a gallery of function landscapes ?
- how to plot convergence paths in 10d.

Comments welcome.

cheers  
-- denis 8 Sept 2014
