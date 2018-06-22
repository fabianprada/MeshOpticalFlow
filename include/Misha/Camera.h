#ifndef CAMERA_INCLUDED
#define CAMERA_INCLUDED
#include <Misha/Geometry.h>

class Camera
{
	void _setRight( void )
	{
		right = Point3D< double >::CrossProduct( forward , up );
		right /= Length( right );
	}
	void rotatePoint( Point3D< double > axis , double angle , Point3D< double > center )
	{
		Point3D< double > p , r , f , u;
		Point3D< double > v[3];
		double c , s;
		double d[3];

		v[2] = axis/Length( axis );

		v[0] = Point3D< double >::CrossProduct( v[2] , Point3D< double >( 1 , 0 , 0 ) );
		if( Point3D< double >::SquareNorm( v[0] )<.001) v[0] = Point3D< double >::CrossProduct( v[2] , Point3D< double >( 0 , 1 , 0 ) );
		v[0] /= Length( v[0] );
		v[1] = Point3D< double >::CrossProduct( v[2] , v[0] );
		v[1] /= Length( v[1] );

		c = cos(angle);
		s = sin(angle);

		p = position-center;
		for( int j=0 ; j<3 ; j++ ) d[j] = Point3D< double >::Dot( p , v[j] );

		position = v[2]*d[2] + v[0]*(d[0]*c+d[1]*s) + v[1]*(-d[0]*s+d[1]*c) + center;

		for( int j=0 ; j<3 ; j++ )
		{
			r[j] = Point3D< double >::Dot(   right , v[j] );
			f[j] = Point3D< double >::Dot( forward , v[j] );
			u[j] = Point3D< double >::Dot(      up , v[j] );
		}

		r = v[2]*r[2]+v[0]*(r[0]*c+r[1]*s)+v[1]*(-r[0]*s+r[1]*c);
		f = v[2]*f[2]+v[0]*(f[0]*c+f[1]*s)+v[1]*(-f[0]*s+f[1]*c);
		u = v[2]*u[2]+v[0]*(u[0]*c+u[1]*s)+v[1]*(-u[0]*s+u[1]*c);

		forward	= f / Length(f);
		right	= r / Length(r);
		up		= u / Length(u);

		_setRight();
	}

public:
	Point3D< double > position , forward , up , right;

	Camera( void )
	{
		position = Point3D< double >( 0 , 0 , 0 );
		forward  = Point3D< double >( 0 , 0 , 1 );
		up       = Point3D< double >( 0 , 1 , 0 );
		_setRight();
	}
	Camera( Point3D< double > p , Point3D< double > f , Point3D< double > u )
	{
		position = p , forward = f , up = u;
		_setRight();
	}
	void draw( void )
	{
		glMatrixMode( GL_MODELVIEW );        
		glLoadIdentity();
		gluLookAt(
			position[0] , position[1] , position[2] ,
			position[0]+forward[0] , position[1]+forward[1] , position[2]+forward[2] ,
			up[0] , up[1] , up[2]
		);
	}

	void translate( Point3D< double > t ){ position += t; }
	void rotateUp     ( double angle , Point3D< double > p=Point3D< double >() ){ rotatePoint( up      , angle , p ); }
	void rotateRight  ( double angle , Point3D< double > p=Point3D< double >() ){ rotatePoint( right   , angle , p ); }
	void rotateForward( double angle , Point3D< double > p=Point3D< double >() ){ rotatePoint( forward , angle , p ); }

	Point2D< double > project( Point3D< double > p , bool orthographic )
	{
		p -= position;
		double x = Point3D< double >::Dot( p , right ) , y = Point3D< double >::Dot( p , up ) , z = Point3D< double >::Dot( p , forward );
		if( orthographic ) return Point2D< double >( x , y );
		else               return Point2D< double >( x/z , y/1 );
	}
};
#endif // CAMERA_INCLUDED
