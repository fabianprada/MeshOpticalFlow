#ifndef TEXTURE_INCLUDED
#define TEXTURE_INCLUDED

template< class Real >
Point3D< Real > SampleTexture( const unsigned char* texture , int tWidth , int tHeight , Point2D< Real > p , bool bilinear=true )
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
void SampleTextureToVertices
(
	const std::vector< TriangleIndexWithData< Point2D< Real > > >& triangles ,
	const std::vector< Vertex >& vertices , 
	const unsigned char* texture , int tWidth , int tHeight , 
	std::vector< Point3D< Real > >& colors ,
	bool bilinear=true
)
{
	std::vector< int > count( vertices.size() );
	colors.resize( vertices.size() );
	for( int i=0 ; i<vertices.size() ; i++ ) colors[i] = Point3D< Real >();
	for( int i=0 ; i<triangles.size() ; i++ ) for( int j=0 ; j<3 ; j++ ) colors[ triangles[i][j] ] += SampleTexture( texture , tWidth , tHeight , triangles[i].data[j] , bilinear ) , count[ triangles[i][j] ]++;
	for( int i=0 ; i<vertices.size() ; i++ ) colors[i] /= count[i];
}


#endif // TEXTURE_INCLUDED