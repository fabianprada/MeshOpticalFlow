/*
Copyright (c) 2018, Michael Kazhdan and Fabian Prada
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution.

Neither the name of the Johns Hopkins University nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
*/


#undef ARRAY_DEBUG

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/timeb.h>
#ifndef WIN32
#include <sys/time.h>
#endif // WIN32
#include <algorithm>
#include <Misha/CmdLineParser.h>
#include <Misha/Algebra.h>
#include <Misha/Ply.h>
#include <Misha/PNG.h>
#include "Src/Texture.inl"
#include "Src/Subdivide.inl"

cmdLineParameter< char* > In( "in" ) , Texture( "texture" ) , Out( "out" );
cmdLineParameter< float > SubdivideEdgeLength( "eLength" , 0.006f );
cmdLineReadable Verbose( "verbose" );
cmdLineReadable* params[] = { &In , &Texture , &Out , &SubdivideEdgeLength , &Verbose , NULL };

void ShowUsage( const char* ex )
{
	printf( "Usage %s:\n" , ex );
	printf( "\t --%s <input mesh>\n" , In.name );
	printf( "\t --%s <input texture file>\n" , Texture.name );
	printf( "\t --%s <output mesh>]\n" , Out.name );
	printf( "\t --%s <diagonal fraction edge length subdivision> [%f]\n" , SubdivideEdgeLength.name, SubdivideEdgeLength.value);
	printf( "\t --%s\n" , Verbose.name );
}

template< class Real >
int _Execute( void )
{
	std::vector< PlyColorVertex< float > > vertices;
	std::vector< TriangleIndex > triangles;
	int file_type;

	unsigned char* texture = NULL;
	int w , h;
	{
		char* ext = GetFileExtension( Texture.value );
		if( !strcasecmp( ext , "png" ) ) texture = PNGReadColor( Texture.value , w , h );
		else{ fprintf( stderr , "[ERROR] Unrecognized image extension: %s\n" , ext ) ; return EXIT_FAILURE; }
		delete[] ext;
	}

	std::vector< PlyTexturedFace< float > > _triangles;
	std::vector< TriangleIndexWithData< Point2D< Real > > > __triangles;
	std::vector< PlyVertex< float > > _vertices;

	PlyReadPolygons( In.value , _vertices , _triangles , PlyVertex< float >::ReadProperties , NULL , PlyVertex< float >::ReadComponents , PlyTexturedFace< float >::ReadProperties , NULL , PlyTexturedFace< float >::ReadComponents , file_type );
	__triangles.resize( _triangles.size() );
	vertices.resize( _vertices.size() );

	for( int i=0 ; i<_triangles.size() ; i++ ) for( int j=0 ; j<3 ; j++ )
	{
		__triangles[i][j] = _triangles[i][j];
		__triangles[i].data[j] = _triangles[i].texture(j);
	}
	for( int i=0 ; i<vertices.size() ; i++ ) vertices[i].point = _vertices[i].point;

	if( SubdivideEdgeLength.set )
	{
		 Point3D< Real > minCorner = vertices[0].point;
		 Point3D< Real > maxCorner = vertices[0].point;
		 for (int v = 0; v < vertices.size(); v++) {
			 for (int c = 0; c < 3; c++) {
				 minCorner[c] = std::min<Real>(minCorner[c], vertices[v].point[c]);
				 maxCorner[c] = std::max<Real>(maxCorner[c], vertices[v].point[c]);
			 }
		 }
		 Real  diagonalLength = Point3D< Real >::Length(maxCorner - minCorner);
		 SubdivideEdgeLength.value = SubdivideEdgeLength.value * diagonalLength;
		 Subdivide(__triangles, vertices, (Real)SubdivideEdgeLength.value);
	}
		
	triangles.resize( __triangles.size() );
	for( int i=0 ; i<triangles.size() ; i++ ) for( int j=0 ; j<3 ; j++ )
	{
		triangles[i][j] = __triangles[i][j];
		// Assuming texture mapping is seamless so that it doesn't make a difference which wedge we sample from
		vertices[ triangles[i][j] ].color = SampleTexture( texture , w , h , __triangles[i].data[j] );			
	}
	if( Verbose.set ) printf( "Vertices / Triangles: %d / %d\n" , vertices.size() , triangles.size() );

	if( Out.set ) PlyWriteTriangles( Out.value , vertices , triangles , PlyColorVertex< float >::WriteProperties , PlyColorVertex< float >::WriteComponents , file_type );

	return EXIT_SUCCESS;
}
int main( int argc , char* argv[] )
{
	cmdLineParse( argc-1 , argv+1 , params );
	if( !In.set || !Texture.set )
	{
		ShowUsage( argv[0] );
		return EXIT_FAILURE;
	}
	return _Execute< float >();
}
