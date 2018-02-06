OpticalFlow is an application to compute a halfway alignment for  a pair of signals defined on a surface. This application was used for texture interpolation in 

https://dl.acm.org/citation.cfm?id=2925967 

and photometric tracking in

https://dl.acm.org/citation.cfm?id=3073679


If you have Intel MKL installed in your machine uncomment the  the line "#define EIGEN_USE_MKL_ALL" in OpticalFlow.cpp to accelerate the Cholesky factorization routines.
 
Compile in Linux using the makefile provided.

In Windows, copy /include and /lib from 4Windows.zip to the main directory. After the code is compiled copy the content of /dll to /x64/Release.


Run the Examples:


OpticalFlow.exe --in A.ply B.ply --out result.ply --verbose

OpticalFlow.exe --mesh mesh.ply --in A.png B.png --out result.png --verbose