# OpticalFlow

OpticalFlow is an application to compute a halfway alignment for  a pair of signals defined on a surface. This application was used for texture interpolation in:

https://dl.acm.org/citation.cfm?id=2925967 

and photometric tracking in:

https://dl.acm.org/citation.cfm?id=3073679


If you have Intel MKL installed in your machine uncomment the  the line "#define EIGEN_USE_MKL_ALL" in OpticalFlow.cpp to accelerate the Cholesky factorization routines.
 
To compile in Linux, install PNG and OpenGL libraries. Then run the provided Makefile.

In Windows, copy /include and /lib from 4Windows.zip to the main directory. After the code is compiled copy the content of /dll to /x64/Release.

In /Examples we provide testing data for both the two different types of supported input : per-vertex signals, and uv texture maps.

Align per-vertex sampled signals by running:

OpticalFlow.exe --in A.ply B.ply

Align uv texture maps by running:

OpticalFlow.exe --mesh mesh.ply --in A.png B.png

Add the parameter --out result.ply (for the first case) or --out result.png (for the later) to run the application in default configuration and skip the user interface.

For an explanatory video on the usage of the interface please visit,

http://www.cs.jhu.edu/~fpradan1/code/

