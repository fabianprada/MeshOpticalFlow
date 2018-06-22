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


template< class Real , class Vertex > int Subdivide( std::vector< TriangleIndexWithData< Point2D< Real > > >& triangles , std::vector< Vertex >& vertices , Real edgeLength );
template< class Real , class Vertex > int Subdivide( std::vector< TriangleIndex >& triangles , std::vector< Vertex >& vertices , Real edgeLength );

template< class Real , class Vertex >
void SampleTextureToVertices
(
	const std::vector< TriangleIndexWithData< Point2D< Real > > >& triangles ,
	const std::vector< Vertex >& vertices , 
	const unsigned char* texture , int tWidth , int tHeight , 
	std::vector< Point3D< Real > >& colors ,
	bool bilinear
);

template< class Real >
typename FEM::RiemannianMesh< Real >::SamplePoint* GetTextureSource
(
	const std::vector< TriangleIndexWithData< Point2D< Real > > >& triangles ,
	const FEM::RiemannianMesh< Real >& mesh ,
	ConstPointer( typename FEM::RiemannianMesh< Real >::Edge ) edges ,
	int tWidth , int tHeight ,
	int padRadius
);

template< class Real >
typename FEM::RiemannianMesh< Real >::SamplePoint* GetTextureSource
(
	const FEM::RiemannianMesh< Real >& mesh ,
	ConstPointer( typename FEM::RiemannianMesh< Real >::Edge ) edges ,
	const std::vector< Point2D< Real > >& triangleTextures ,
	int tWidth , int tHeight ,
	int padRadius
);

/////////////////////////////////////////////////////

template< class Real >
Point3D< Real > Sample( const unsigned char* texture , int tWidth , int tHeight , Point2D< Real > p , bool bilinear )
{
	p[1] = 1 - p[1];
	p[0] = std::min< Real >( 1. , std::max< Real >( 0. , p[0] ) );
	p[1] = std::min< Real >( 1. , std::max< Real >( 0. , p[1] ) );
	p[0] *= tWidth-1 , p[1] *= tHeight-1;
	int x0 = (int)floor( p[0] ) , y0 = (int)floor( p[1] );
	if( bilinear )
	{
		Real dx = p[0] - x0 , dy = p[1] - y0;
		int x1 = std::min< int >( x0+1 , tWidth-1 ) , y1 = std::min< int >( y0+1 , tHeight-1 );
		return
			Point3D< Real >( (Real)( texture[3*(tWidth*y0+x0)] ) , (Real)( texture[3*(tWidth*y0+x0)+1] ) , (Real)( texture[3*(tWidth*y0+x0)+2] ) ) * (Real)( (1.-dx) * (1.-dy) ) +
			Point3D< Real >( (Real)( texture[3*(tWidth*y0+x1)] ) , (Real)( texture[3*(tWidth*y0+x1)+1] ) , (Real)( texture[3*(tWidth*y0+x1)+2] ) ) * (Real)( (   dx) * (1.-dy) ) +
			Point3D< Real >( (Real)( texture[3*(tWidth*y1+x1)] ) , (Real)( texture[3*(tWidth*y1+x1)+1] ) , (Real)( texture[3*(tWidth*y1+x1)+2] ) ) * (Real)( (   dx) * (   dy) ) +
			Point3D< Real >( (Real)( texture[3*(tWidth*y1+x0)] ) , (Real)( texture[3*(tWidth*y1+x0)+1] ) , (Real)( texture[3*(tWidth*y1+x0)+2] ) ) * (Real)( (1.-dx) * (   dy) ) ;
	}
	else return Point3D< Real >( (Real)( texture[3*(tWidth*y0+x0)] ) , (Real)( texture[3*(tWidth*y0+x0)+1] ) , (Real)( texture[3*(tWidth*y0+x0)+2] ) );
}

template< class Real , class Vertex >
int _Subdivide( std::vector< TriangleIndex >& triangles , std::vector< Vertex >& vertices , Real edgeLength )
{
	int subdivideCount = 0;
	std::unordered_map< long long , int > edgeMap;
	std::vector< TriangleIndex > _triangles;
	std::vector< Vertex > _vertices = vertices;

	for( int i=0 ; i<triangles.size() ; i++ )
	{
		int eCount = 0;
		int e[] = { -1 , -1 , -1 };
		for( int j=0 ; j<3 ; j++ ) 
		{
			int i1 = triangles[i][j] , i2 = triangles[i][(j+1)%3];
			Real l2 = Point3D< Real >::SquareNorm( vertices[i1].point - vertices[i2].point );
			if( l2>edgeLength*edgeLength )
			{
				long long key = EdgeKey( i1 , i2 );
				int idx;
				std::unordered_map< long long , int >::iterator iter = edgeMap.find( key );
				if( iter==edgeMap.end() )
				{
					idx = (int)_vertices.size();
					edgeMap[key] = idx;
					_vertices.push_back( ( vertices[i1] + vertices[i2] ) / 2 );
					subdivideCount++;
				}
				else idx = iter->second;
				e[j] = idx;
				eCount++;
			}
		}
		if( eCount==0 ) _triangles.push_back( triangles[i] );
		else if( eCount==1 )
		{
			for( int j=0 ; j<3 ; j++ ) if( e[j]!=-1 )
			{
				_triangles.push_back( TriangleIndex( triangles[i][ j     ] , e[j] , triangles[i][(j+2)%3] ) );
				_triangles.push_back( TriangleIndex( triangles[i][(j+1)%3] , triangles[i][(j+2)%3] , e[j] ) );
			}
		}
		else if( eCount==2 )
		{
			for( int j=0 ; j<3 ; j++ ) if( e[j]==-1 )
			{
				_triangles.push_back( TriangleIndex( e[(j+1)%3] , triangles[i][(j+2)%3] , e[(j+2)%3] ) );
				_triangles.push_back( TriangleIndex( triangles[i][j] , triangles[i][(j+1)%3] , e[(j+2)%3] ) );
				_triangles.push_back( TriangleIndex( triangles[i][(j+1)%3] , e[(j+1)%3] , e[(j+2)%3] ) );
			}
		}
		else if( eCount==3 )
		{
			for( int j=0 ; j<3 ; j++ ) _triangles.push_back( TriangleIndex( triangles[i][j] , e[j] , e[(j+2)%3] ) );
			_triangles.push_back( TriangleIndex( e[0] , e[1] , e[2] ) );
		}
	}
	triangles = _triangles;
	vertices = _vertices;
	return subdivideCount;
}
template< class Real , class Vertex >
int Subdivide( std::vector< TriangleIndex >& triangles , std::vector< Vertex >& vertices , Real edgeLength )
{
	int count = 0;
	while( true )
	{
		int _count = _Subdivide( triangles , vertices , edgeLength );
		if( _count ) count += _count;
		else return count;
	}
}
template< class Real , class Vertex >
int _Subdivide( std::vector< TriangleIndexWithData< Point2D< Real > > >& triangles , std::vector< Vertex >& vertices , Real edgeLength )
{
	int subdivideCount = 0;
	std::unordered_map< long long , int > edgeMap;
	std::vector< TriangleIndexWithData< Point2D< Real > > > _triangles;
	std::vector< Vertex > _vertices = vertices;

	for( int i=0 ; i<triangles.size() ; i++ )
	{
		int eCount = 0;
		int e[] = { -1 , -1 , -1 };
		Point2D< Real > texture[3];
		for( int j=0 ; j<3 ; j++ ) 
		{
			int i1 = triangles[i][j] , i2 = triangles[i][(j+1)%3];
			Real l2 = Point3D< Real >::SquareNorm( vertices[i1].point - vertices[i2].point );
			if( l2>edgeLength*edgeLength )
			{
				long long key = EdgeKey( i1 , i2 );
				int idx;
				std::unordered_map< long long , int >::iterator iter = edgeMap.find( key );
				if( iter==edgeMap.end() )
				{
					idx = (int)_vertices.size();
					edgeMap[key] = idx;
					_vertices.push_back( ( vertices[i1] + vertices[i2] ) / 2 );
					subdivideCount++;
				}
				else idx = iter->second;
				e[j] = idx;
				texture[j] = ( triangles[i].data[j] + triangles[i].data[(j+1)%3] ) / 2;
				eCount++;
			}
		}
		if( eCount==0 ) _triangles.push_back( triangles[i] );
		else if( eCount==1 )
		{
			for( int j=0 ; j<3 ; j++ ) if( e[j]!=-1 )
			{
				_triangles.push_back( TriangleIndexWithData< Point2D< Real > >( triangles[i][ j     ] , e[j] , triangles[i][(j+2)%3] , triangles[i].data[ j     ] , texture[j] , triangles[i].data[(j+2)%3] ) );
				_triangles.push_back( TriangleIndexWithData< Point2D< Real > >( triangles[i][(j+1)%3] , triangles[i][(j+2)%3] , e[j] , triangles[i].data[(j+1)%3] , triangles[i].data[(j+2)%3] , texture[j] ) );
			}
		}
		else if( eCount==2 )
		{
			for( int j=0 ; j<3 ; j++ ) if( e[j]==-1 )
			{
				_triangles.push_back( TriangleIndexWithData< Point2D< Real > >( e[(j+1)%3] , triangles[i][(j+2)%3] , e[(j+2)%3] , texture[(j+1)%3] , triangles[i].data[(j+2)%3] , texture[(j+2)%3] ) );
				_triangles.push_back( TriangleIndexWithData< Point2D< Real > >( triangles[i][j] , triangles[i][(j+1)%3] , e[(j+2)%3] , triangles[i].data[j] , triangles[i].data[(j+1)%3] , texture[(j+2)%3] ) );
				_triangles.push_back( TriangleIndexWithData< Point2D< Real > >( triangles[i][(j+1)%3] , e[(j+1)%3] , e[(j+2)%3] , triangles[i].data[(j+1)%3] , texture[(j+1)%3] , texture[(j+2)%3] ) );
			}
		}
		else if( eCount==3 )
		{
			for( int j=0 ; j<3 ; j++ ) _triangles.push_back( TriangleIndexWithData< Point2D< Real > >( triangles[i][j] , e[j] , e[(j+2)%3] , triangles[i].data[j] , texture[j] , texture[(j+2)%3] ) );
			_triangles.push_back( TriangleIndexWithData< Point2D< Real > >( e[0] , e[1] , e[2] , texture[0] , texture[1] , texture[2] ) );
		}
	}
	triangles = _triangles;
	vertices = _vertices;
	return subdivideCount;
}

template< class Real , class Vertex >
int Subdivide( std::vector< TriangleIndexWithData< Point2D< Real > > >& triangles , std::vector< Vertex >& vertices , Real edgeLength )
{
	int count = 0;
	while( true )
	{
		int _count = _Subdivide( triangles , vertices , edgeLength );
		if( _count ) count += _count;
		else return count;
	}
}

template< class Real >
struct PointData
{
	int count;
	Point3D< Real > p , c;
};
template< class Real >
unsigned long long PointKey( Point3D< Real > p )
{
	for( int j=0 ; j<3 ; j++ ) p[j] = std::max< Real >( 0 , std::min< Real >( 1 , p[j] ) );
	unsigned long long x , y , z;
	x = (unsigned long long)( p[0] * ( 1<<21 ) );
	y = (unsigned long long)( p[1] * ( 1<<21 ) );
	z = (unsigned long long)( p[2] * ( 1<<21 ) );
	return ( x ) | ( y<<21 ) | ( z<<42 );
}

template< class Real , class Vertex >
void SampleTextureToVertices
(
	const std::vector< TriangleIndexWithData< Point2D< Real > > >& triangles ,
	const std::vector< Vertex >& vertices , 
	const unsigned char* texture , int tWidth , int tHeight , 
	std::vector< Point3D< Real > >& colors ,
	bool bilinear
)
{
	std::vector< int > count( vertices.size() );
	colors.resize( vertices.size() );
	for( int i=0 ; i<vertices.size() ; i++ ) colors[i] = Point3D< Real >();
	for( int i=0 ; i<triangles.size() ; i++ ) for( int j=0 ; j<3 ; j++ ) colors[ triangles[i][j] ] += Sample( texture , tWidth , tHeight , triangles[i].data[j] , bilinear ) , count[ triangles[i][j] ]++;
	for( int i=0 ; i<vertices.size() ; i++ ) colors[i] /= count[i];
}
template< class Real >
Point2D< Real > BarycentricCoordinate( const Point2D< Real > v[3] , Point2D< Real > p )
{
	// solve for (s,t) such that:
	//		p-v[0] = s * ( v[1]-v[0] ) + t * ( v[2]-v[0] )
	//		       = s * w1 + t * w2
	Point2D< Real > w1 = v[1]-v[0] , w2 = v[2]-v[0];
	SquareMatrix< Real , 2 > M;
	M(0,0) = w1[0] , M(1,0) = w2[0];
	M(0,1) = w1[1] , M(1,1) = w2[1];
	return M.inverse() * ( p-v[0] );
}

template< class Real >
void RasterizeTriangle( Point2D< Real > v0 , Point2D< Real > v1 , Point2D< Real > v2 , int tIdx , FEM::SamplePoint< Real >* samplePoints , int width , int height )
{
	Point2D< Real > v[] = { v0 , v1 , v2 };
	for( int j=0 ; j<3 ; j++ ) v[j][0] *= (width-1) , v[j][1] *= (height-1);
	// Sort the points from highest to lowest
	int map[3];
	if( v0[1]<=v1[1] && v0[1]<=v2[1] )
	{
		map[0] = 0;
		if( v1[1]<=v2[1] ) map[1] = 1 , map[2] = 2;
		else               map[1] = 2 , map[2] = 1;
	}
	else if( v1[1]<=v0[1] && v1[1]<=v2[1] )
	{
		map[0] = 1;
		if( v0[1]<=v2[1] ) map[1] = 0 , map[2] = 2;
		else               map[1] = 2 , map[2] = 0;
	}
	else
	{
		map[0] = 2;
		if( v0[1]<=v1[1] ) map[1] = 0 , map[2] = 1;
		else               map[1] = 1 , map[2] = 0;
	}
	Point2D< Real > w[] = { v[ map[0] ] , v[ map[1] ] , v[ map[2] ] };
	int yStart = (int)ceil( w[0][1] ) , yEnd = (int)floor( w[2][1] );
	yStart = std::max< int >( 0 , std::min< int >( height-1 , yStart ) );
	yEnd   = std::max< int >( 0 , std::min< int >( height-1 , yEnd   ) );

	Point2D< Real > source , slopes[2];
	source = w[0] , slopes[0] = w[1]-w[0] , slopes[1] = w[2]-w[0];
	for( int y=yStart ; y<=yEnd ; y++ )
	{
		if( y>=w[1][1] ) source = w[2] , slopes[0] = w[1]-w[2] , slopes[1] = w[0]-w[2];
		if( slopes[0][1]==0 || slopes[1][1]==0 ) continue;
		// source[1] + t * slopes[i][1] = y
		// => t = ( y - source[1] ) / slopes[i][1]
		// => x = sources[0] + ( y - source[1] ) * slopes[i][0] / slopes[i][1]
		Real xIntercepts[] = { source[0] + ( (Real)y-source[1] ) * slopes[0][0] / slopes[0][1] , source[0] + ( (Real)y-source[1] ) * slopes[1][0] / slopes[1][1] };
		int xStart , xEnd;
		if( xIntercepts[0]<=xIntercepts[1] ) xStart = (int)ceil( xIntercepts[0] ) , xEnd = (int)floor( xIntercepts[1] );
		else                                 xStart = (int)ceil( xIntercepts[1] ) , xEnd = (int)floor( xIntercepts[0] );
		xStart = std::max< int >( 0 , std::min< int >( width-1 , xStart ) );
		xEnd   = std::max< int >( 0 , std::min< int >( width-1 , xEnd   ) );
		Point2D< Real > b[2];
		b[0] = BarycentricCoordinate( v , Point2D< Real >( (Real)xStart , (Real)y ) );
		b[1] = BarycentricCoordinate( v , Point2D< Real >( (Real)xEnd , (Real)y ) );
		for( int x=xStart ; x<=xEnd ; x++ )
		{
			Real s = (Real)( x - xStart ) / (Real)( xEnd-xStart );
			if( xStart==xEnd ) s = (Real)0.;
			Point2D< Real > _b = b[0] * ( (Real)1.-s ) + b[1] * s;
			FEM::SamplePoint< Real >& p = samplePoints[ y*width + x ];
			if( p.tIdx==-1 || ( _b[0]>=0 && _b[1]>=1 &&  _b[0]+_b[1]<=1 ) ) p.tIdx = tIdx , p.p = _b;
		}
	}
}

template< class Real >
void RemapSamplePoint( const FEM::RiemannianMesh< Real >& mesh , ConstPointer( FEM::EdgeXForm< Real > ) edges , FEM::SamplePoint< Real >& p )
{
	if( p.p[0]>=0 && p.p[1]>=0 && p.p[0]+p.p[1]<=1 ) return;
	else
	{
		FEM::HermiteSamplePoint< Real > _p;
		_p.tIdx = p.tIdx , _p.p = Point2D< Real >( (Real)1./3 , (Real)1./3 ) , _p.v = p.p - _p.p;
		mesh.exp( edges , _p );
		p = _p;
	}
}

template< class Real >
typename FEM::RiemannianMesh< Real >::SamplePoint* GetTextureSource
(
	const std::vector< TriangleIndexWithData< Point2D< Real > > >& triangles ,
	const FEM::RiemannianMesh< Real >& mesh ,
	ConstPointer( FEM::EdgeXForm< Real > ) edges ,
	int tWidth , int tHeight ,
	int padRadius
)
{

	FEM::SamplePoint< Real >* samplePoints = new FEM::SamplePoint< Real >[ tWidth * tHeight ];
	for( int i=0 ; i<tWidth*tHeight ; i++ ) samplePoints[i].tIdx = -1;

	for( int i=0 ; i<triangles.size() ; i++ ) RasterizeTriangle( triangles[i].data[0] , triangles[i].data[1] , triangles[i].data[2] , i , samplePoints , tWidth , tHeight );

	if( padRadius>0 )
	{
		std::vector< int > updateIndex( tWidth * tHeight );
		for( int r=0 ; r<padRadius ; r++ )
		{
			// Mark all pixels that can be updated
			for( int i=0 ; i<tWidth ; i++ ) for( int j=0 ; j<tHeight ; j++ )
			{
				int idx = j*tWidth + i;
				updateIndex[idx] = -1;
				if( samplePoints[idx].tIdx==-1 )
				{
					for( int ii=-1 ; ii<=1 ; ii++ ) if( i+ii>=0 && i+ii<tWidth  && samplePoints[ j*tWidth + (i+ii) ].tIdx!=-1 ) updateIndex[idx] = samplePoints[ j*tWidth + (i+ii) ].tIdx;
					for( int jj=-1 ; jj<=1 ; jj++ ) if( j+jj>=0 && j+jj<tHeight && samplePoints[ (j+jj)*tWidth + i ].tIdx!=-1 ) updateIndex[idx] = samplePoints[ (j+jj)*tWidth + i ].tIdx;
				}
			}
			for( int i=0 ; i<tWidth ; i++ ) for( int j=0 ; j<tHeight ; j++ )
			{
				int idx = j*tWidth + i;
				int t = updateIndex[idx];
				if( t!=-1 )
				{
					Point2D< Real > v[] = { triangles[t].data[0] , triangles[t].data[1] , triangles[t].data[2] };
					Point2D< Real > p( (Real)i/(tWidth-1) , (Real)j/(tHeight-1) );
					samplePoints[idx].tIdx = updateIndex[idx];
					samplePoints[idx].p = BarycentricCoordinate( v , p );
				}
			}
		}
	}

	for( int i=0 ; i<tWidth ; i++ ) for( int j=0 ; j<tHeight ; j++ )
	{
		FEM::SamplePoint< Real > p = samplePoints[ j*tWidth + i ];
		if( p.tIdx!=-1 )
		{
			RemapSamplePoint( mesh , edges , p );
			samplePoints[ j*tWidth + i ] = p;
		}
	}
	return samplePoints;
}
template< class Real >
FEM::SamplePoint< Real >* GetTextureSource
(
	const FEM::RiemannianMesh< Real >& mesh ,
	ConstPointer( FEM::EdgeXForm< Real > ) edges ,
	const std::vector< Point2D< Real > >& triangleTextures ,
	int tWidth , int tHeight ,
	int padRadius
)
{

	FEM::SamplePoint< Real >* samplePoints = new FEM::SamplePoint< Real >[ tWidth * tHeight ];
	for( int i=0 ; i<tWidth*tHeight ; i++ ) samplePoints[i].tIdx = -1;

	for( int i=0 ; i<mesh.tCount ; i++ ) RasterizeTriangle( triangleTextures[3*i] , triangleTextures[3*i+1] , triangleTextures[3*i+2] , i , samplePoints , tWidth , tHeight );

	if( padRadius>0 )
	{
		std::vector< int > updateIndex( tWidth * tHeight );
		for( int r=0 ; r<padRadius ; r++ )
		{
			// Mark all pixels that can be updated
			for( int i=0 ; i<tWidth ; i++ ) for( int j=0 ; j<tHeight ; j++ )
			{
				int idx = j*tWidth + i;
				updateIndex[idx] = -1;
				if( samplePoints[idx].tIdx==-1 )
				{
					for( int ii=-1 ; ii<=1 ; ii++ ) if( i+ii>=0 && i+ii<tWidth  && samplePoints[ j*tWidth + (i+ii) ].tIdx!=-1 ) updateIndex[idx] = samplePoints[ j*tWidth + (i+ii) ].tIdx;
					for( int jj=-1 ; jj<=1 ; jj++ ) if( j+jj>=0 && j+jj<tHeight && samplePoints[ (j+jj)*tWidth + i ].tIdx!=-1 ) updateIndex[idx] = samplePoints[ (j+jj)*tWidth + i ].tIdx;
				}
			}
			for( int i=0 ; i<tWidth ; i++ ) for( int j=0 ; j<tHeight ; j++ )
			{
				int idx = j*tWidth + i;
				int t = updateIndex[idx];
				if( t!=-1 )
				{
					Point2D< Real > v[] = { triangleTextures[3*t] , triangleTextures[3*t+1] , triangleTextures[3*t+2] };
					Point2D< Real > p( (Real)i/(tWidth-1) , (Real)j/(tHeight-1) );
					samplePoints[idx].tIdx = updateIndex[idx];
					samplePoints[idx].p = BarycentricCoordinate( v , p );
				}
			}
		}
	}

	for( int i=0 ; i<tWidth ; i++ ) for( int j=0 ; j<tHeight ; j++ )
	{
		FEM::SamplePoint< Real > p = samplePoints[ j*tWidth + i ];
		if( p.tIdx!=-1 )
		{
			RemapSamplePoint( mesh , edges , p );
			samplePoints[ j*tWidth + i ] = p;
		}
	}
	return samplePoints;
}
