#include <math.h>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <atomic>

#undef SUPPORT_LINEAR_PROGRAM

/////////
// FEM //
/////////
template< class Real >
inline Real FEM::Area( const SquareMatrix< Real , 2 >& tensor , const Point2D< Real > tri[3] )
{
	return Point2D< Real >::Dot( tri[2]-tri[0] , tensor * Rotate90( tensor , tri[1] - tri[0] ) ) / (Real)2;
}
template< class Real >
inline Point2D< Real > FEM::Rotate90( const SquareMatrix< Real , 2 >& tensor , Point2D< Real > v )
{
	Point2D< Real > w = tensor.inverse() * Point2D< Real >( -v[1] , v[0] );
	Real vNorm2 = Point2D< Real >::Dot( tensor * v , v ) , wNorm2 = Point2D< Real >::Dot( tensor * w , w );
	if( wNorm2 ) return w * (Real)sqrt( vNorm2 / wNorm2 );
	else         return w;
}

template< class Real >
inline SquareMatrix< Real , 6 > FEM::TraceForm( const SquareMatrix< Real , 2 >& tensor , const Point2D< Real > dirs[3] )
{
	SquareMatrix< Real , 6 > tForm;

	// The linear operator that takes the values of the matrix along the three directions
	// and returns the best-fit linear transform.
	Matrix< Real , 6 , 4 > lFit = LinearFit( dirs );

	// Given the vectors {v0,v1,v2} and {w0,w1,w2}, we would like to:
	// -- Compute the best fit linear operators: L_v = L(v0,v1,v2) and L_w = L(w0,w1,w2)
	// -- Compute the bilinear form: B_{v,w}(x,y) = tensor( L_v * x )[ L_w * y ] = x^t * L_v^t * tensor * L_w * y
	// -- Compute the corresponding linear operator: L_{v,w} = tensor^{-1} * B_{v,w} = tensor^{-1} * L_v^t * tensor * L_w
	// -- Take its trace: Tr( L_{v,w} )
	SquareMatrix< Real , 2 > iTensor = tensor.inverse();
	for( int i=0 ; i<6 ; i++ ) for( int j=0 ; j<6 ; j++ )
	{
		SquareMatrix< Real , 2 > L_v , L_w;
		Real *L_v_ptr = &L_v(0,0) , *L_w_ptr = &L_w(0,0);
		for( int k=0 ; k<4 ; k++ ) L_v_ptr[k] = lFit(i,k) , L_w_ptr[k] = lFit(j,k);
		SquareMatrix< Real , 2 > L_vw = iTensor * L_v.transpose() * tensor * L_w;
		tForm(i,j) = L_vw(0,0) + L_vw(1,1);
	}
	return tForm;
}
template< class Real >
inline SquareMatrix< Real , 6 > FEM::LinearFitEvaluation( const Point2D< Real > dirs[3] )
{
	// The linear operator that takes the values of the matrix along the three directions
	// and returns the best-fit linear transform.
	Matrix< Real , 6 , 4 > lFit = LinearFit( dirs );

	// Given the vectors {v0,v1,v2} , we would like to:
	// -- Compute the best fit linear operators: L_v = L(v0,v1,v2)
	// -- Compute the difference between the predicted and actual values: dv_{j} = ( L_v * directions[j] - v_{j} )
	SquareMatrix< Real , 6 > lfEvaluation;

	for( int i=0 ; i<6 ; i++ )
	{
		SquareMatrix< Real , 2 > L_v;
		Real *L_v_ptr = &L_v(0,0);
		for( int k=0 ; k<4 ; k++ ) L_v_ptr[k] = lFit(i,k);
		for( int j=0 ; j<3 ; j++ )
		{
			Point2D< Real > temp = L_v * dirs[j];
			lfEvaluation( i , j*2+0 ) = temp[0] , lfEvaluation( i , j*2+1 ) = temp[1];
		}
	}

	return lfEvaluation;
}
template< class Real > inline SquareMatrix< Real , 6 > FEM::LinearFitResidual( const Point2D< Real > dirs[3] ){ return LinearFitEvaluation( dirs ) - SquareMatrix< Real , 6 >::Identity(); }

template< class Real >
inline SquareMatrix< Real , 6 > FEM::MCTraceForm( const SquareMatrix< Real , 2 >& tensor , const Point2D< Real > dirs[3] , int quadratureType )
{
	SquareMatrix< Real , 6 > tForm;

	Point3D< Real > weights( 1 , 1 , 1 );
	CircularQuadratureWeights( tensor , dirs , 3 , &weights[0] , quadratureType );
	weights /= (Real)( M_PI );

	for( int i=0 ; i<3 ; i++ )
	{
		weights[i] /= Point2D< Real >::Dot( dirs[i] , tensor * dirs[i] );
		for( int j=0 ; j<2 ; j++ ) for( int k=0 ; k<2 ; k++ ) tForm(2*i+j,2*i+k) = tensor(j,k) * weights[i];
	}
	return tForm;
}

#ifdef SUPPORT_LINEAR_PROGARM
#include <Eigen/Dense>
#define IL_STD
#include <ilcplex/ilocplex.h>
template< class Real >
void FEM::TraceWeights( const Point2D< Real >* directions , int dirCount , Real* weights )
{
	{
		IloEnv env;
		IloModel model( env );
		IloNumVarArray var( env , dirCount , 0.0 , IloInfinity , ILOFLOAT );
		IloExpr ojbective( env );
		for( int i=0 ; i<dirCount ; i++ ) ojbective += 1. * var[i];
		model.add( IloMinimize( env , ojbective ) );
		IloExpr constraint1( env ) , constraint2( env ) , constraint3( env );
		for( int i=0 ; i<dirCount ; i++ ) constraint1 += ( directions[i][0] * directions[i][0] ) * var[i] , constraint2 += ( directions[i][1] * directions[i][1] ) * var[i] , constraint3 += ( directions[i][0] * directions[i][1] ) * var[i];
		model.add( constraint1==1. ) , model.add( constraint2==1. ) , model.add( constraint3==0. );

		IloCplex cplex( model );
//		cplex.setParam(IloCplex::Param::Simplex::Display, 2);
//		cplex.setParam(IloCplex::Param::RootAlgorithm, IloCplex::Network);
		cplex.setOut( env.getNullStream() );
		if( !cplex.solve() )
		{
			IloNumArray x( env ) ; cplex.getValues( x , var );
			fprintf( stderr , "[ERROR] Failed to solve linear program\n" );
			std::vector< Real > angles( dirCount );
			for( int i=0 ; i<dirCount ; i++ ) angles[i] = atan2( directions[i][1] , directions[i][0] );
//			std::sort( angles.begin() , angles.end() , []( Real a1 , Real a2 ){ return a1<a2; } );
			fprintf( stderr , "\tAngles: " ) ; for( int i=0 ; i<dirCount ; i++ ) fprintf( stderr , " %g" , angles[i] * 360. / 2. / M_PI ) ; fprintf( stderr , "\n" );
			fprintf( stderr , "\tWeights: " ) ; for( int i=0 ; i<dirCount ; i++ ) fprintf( stderr , " %g" , x[i] ) ; fprintf( stderr , "\n" );
			exit( 0 );
		}

		constraint1.end();
		constraint2.end();
		constraint3.end();
		ojbective.end();
		IloNumArray x( env ) ; cplex.getValues( x , var );
		for( int i=0 ; i<dirCount ; i++ ) weights[i] = x[i];
		env.end();
		return;
	}

	// Want the set of weights such that:
	//		\sum_i w[i] * dir[i] * dir[i]^t = g^{-1}
	// with w[i]>=0.
	// For uniformly distributed directions on the circle, we expect w[i] = 2 / n
	// For uniformly distributed directions, we expect w[i] = 2 / ( n * || dir[i] ||^2 )
	Eigen::MatrixXd M( dirCount+3 , dirCount+3 );
	Eigen::VectorXd b( dirCount+3 ) , x( dirCount+3 );

	SquareMatrix< Real , 2 > I = SquareMatrix< Real , 2 >::Identity();
	static const int Indices[][2] = { { 0 , 0 } , { 0 , 1 } , { 1 , 1 } };

	// Construct the equality constraints
	for( int i=0 ; i<3 ; i++ )
	{
		for( int j=0 ; j<dirCount ; j++ ) M( dirCount+i , j ) = M( j , dirCount+i ) = directions[j][ Indices[i][0] ] * directions[j][ Indices[i][1] ];
		b[ dirCount+i ] = I( Indices[i][0] ,  Indices[i][1] );
	}
	for( int i=0 ; i<dirCount ;  i++ ) b[i] = 0;

	// Add the optimization constraints
	// E = \sum_{i \neq j} ( w[i] - w[j] )^2
	//   = \sum_{i \neq j} w^2[i] + w*2[j] - 2 * w[i] * w[j]
//	for( int i=0 ; i<dirCount ; i++ ) for( int j=0 ; j<dirCount ; j++ ) M(i,j) = -1.;
	for( int i=0 ; i<dirCount ; i++ ) for( int j=0 ; j<dirCount ; j++ ) M(i,j) =  0.;
	for( int i=0 ; i<dirCount ; i++ ) M(i,i) = dirCount;
	for( int i=0 ; i<3 ; i++ ) for( int j=0 ; j<3 ; j++ ) M( i+dirCount , j+dirCount ) = 0;

	// Solve the linear sytem
	x = Eigen::FullPivLU< Eigen::MatrixXd >( M ).solve( b );

	if( dirCount<20 )
	{
		Eigen::MatrixXd M( dirCount , dirCount );
		M *= 0;
		for( int i=0 ; i<3 ; i++ ) for( int j=0 ; j<dirCount ; j++ ) M(i,j) = directions[j][ Indices[i][0] ] * directions[j][ Indices[i][1] ];
		Eigen::MatrixXd K = Eigen::FullPivLU< Eigen::MatrixXd >( M ).kernel();
		printf( "Kernel dim: %d x %d\n" , K.cols() , K.rows() );
std::cout << K << std::endl;
	}

	for( int i=0 ; i<dirCount ; i++ ) weights[i] = (Real)x[i];
}
#endif // SUPPORT_LINEAR_PROGAM
template< class Real >
inline Point3D< Real > FEM::TraceWeights( const SquareMatrix< Real , 2 >& tensor , const Point2D< Real > directions[3] )
{
	// Given a linear operator L, a tensor g, and directions v[3], we want to find the weights w[3] such that:
	//		Tr( L ) = \sum w[i] * < v[i] , L( v[i] ) >_g
	// Re-writing the RHS we get:
	//		Tr( L ) = \sum w[i] * < v[i] , ( g * L ) * v[i] >
	//		        = \sum w[i] * Tr( v[i]^t * g * L * v[i] )
	//		        = Tr( L * ( \sum w[i] * v[i] * v[i]^t ) * g )
	// Thus, the weights w[3] must satisfy:
	//		\sum w[i] * v[i] * v[i]^t = g^{-1}
	static const int Indices[][2] = { { 0 , 0 } , { 0, 1 } , { 1 , 1 } };
	SquareMatrix< Real , 2 > tensor_inverse = tensor.inverse();
	SquareMatrix< Real , 2 > M[3];
	for( int i=0 ; i<3 ; i++ ) for( int j=0 ; j<2 ; j++ ) for( int k=0 ; k<2 ; k++ ) M[i](j,k) = directions[i][j] * directions[i][k];
	SquareMatrix< Real , 3 > A;
	Point3D< Real > b;
	for( int i=0 ; i<3 ; i++ )
	{
		b[i] = tensor_inverse( Indices[i][0] , Indices[i][1] );
		for( int j=0 ; j<3 ; j++ ) A(i,j) = M[i]( Indices[j][0] , Indices[j][1] );
	}
	return A.inverse() * b;
}
template< class Real >
inline Matrix< Real , 6 , 4 > FEM::LinearFit( const Point2D< Real > v[3] )
{
// Given directions v[3] and values w[3], solve for the linear operator L minimizing:
//	E(L) = \sum_i ||L(v[i]) - w[i]||_g^2
//       = \sum_i ( L(v[i]) - w[i] )^t * g * ( L(v[i]) - w[i] )
//       = \sum_i v[i]^t * L^t * g * L * v[i] - 2 * v[i]^t * L^t * g*  w[i] + ...
//       = \sum_i Tr( L^t * g * L * v[i] * v[i]^t ) - 2 Tr( L^t * g * w[i] * v[i]^t ) + ...
// Setting V = \sum_i v[i] * v[i]^t and W = \sum_i w[i] * v[i]^t, this gives:
//       = Tr( L^t * g * L * V ) - 2 * Tr( L^t * g * W )
//       = [ Tr( L^t * g * L * V ) + Tr( L * V * L^t * g ) ] / 2 - 2 * Tr( L^t * g * W )
//       = [ Tr( L^t * g * L * V ) + Tr( g * L * V^t * L^t ) ] / 2 - 2 * Tr( L^t * g * W )
//       = [ Tr( L^t * g * L * V ) + Tr( L^T * g * L * V^t ] / 2 - 2 * Tr( L^t * g * W )
// Differentiating with respect to to L gives:
// g * L * ( V + V^t )/2 = g * W
// Or equivalently:
//	    L = W * V^{-1}
	auto OuterProduct = [] ( Point2D< Real > v1 , Point2D< Real > v2 )
	{
		SquareMatrix< Real , 2 > M;
		for( int i=0 ; i<2 ; i++ ) for( int j=0 ; j<2 ; j++ ) M(i,j) = v1[i] * v2[j];
		return M;
	};

	// Compute the sum of the outer products of the drections and invert
	SquareMatrix< Real , 2 > V;
	for( int i=0 ; i<3 ; i++ ) V += OuterProduct( v[i] , v[i] );
	V = V.inverse();

	Matrix< Real , 6 , 4 > fitMatrix;
	for( int i=0 ; i<3 ; i++ ) for( int j=0 ; j<2 ; j++ )
	{
		SquareMatrix< Real , 2 > W = OuterProduct( v[i] , j==0 ? Point2D< Real >(1,0) : Point2D< Real >(0,1) ) * V;
		memcpy( &fitMatrix(2*i+j,0) , &W(0,0) , sizeof(Real)*4 );
	}
	return fitMatrix;
}
template< class Real > void FEM::CircularQuadratureWeights( const SquareMatrix< Real , 2 >& tensor , const Point2D< Real >* dirs , int dirCount , Real* weights , int quadratureType )
{
	if( quadratureType & QUADRATURE_ANGULAR )
	{
		struct IndexedAngle { int idx ; Real angle; };
		Point2D< Real > x( 1 , 0 ) , y = Rotate90( tensor , x );
		x = tensor * x , y = tensor * y;
		std::vector< IndexedAngle > iAngles( 2*dirCount );
		for( int i=0 ; i<dirCount ; i++ )
		{
			weights[i] = 0;
			iAngles[2*i].idx = iAngles[2*i+1].idx = i;
			iAngles[2*i].angle = atan2( Point2D< Real >::Dot( y , dirs[i] ) , Point2D< Real >::Dot( x , dirs[i] ) ) , iAngles[2*i+1].angle = iAngles[2*i].angle + (Real)M_PI;
		}
		for( int i=0 ; i<2*dirCount ; i++ )
		{
			while( iAngles[i].angle<0       ) iAngles[i].angle += (Real)( 2. * M_PI );
			while( iAngles[i].angle>2.*M_PI ) iAngles[i].angle -= (Real)( 2. * M_PI );
		}
		std::sort( iAngles.begin() , iAngles.end() , [] ( const IndexedAngle& i1 , const IndexedAngle& i2 ){ return i1.angle<i2.angle; } );
		for( int i=0 ; i<2*dirCount ; i++ )
		{
			Real a1 , a2;
			if( i==0            ) a1 = ( iAngles[i].angle + iAngles[2*dirCount-1].angle - 2. * M_PI ) / 2;
			else                  a1 = ( iAngles[i].angle + iAngles[i-1         ].angle             ) / 2;
			if( i==2*dirCount-1 ) a2 = ( iAngles[i].angle + iAngles[0           ].angle + 2. * M_PI ) / 2;
			else                  a2 = ( iAngles[i].angle + iAngles[i+1         ].angle             ) / 2;
			weights[ iAngles[i].idx ] += a2 - a1;
		}
	}
	else for( int i=0 ; i<dirCount ; i++ ) weights[i] = (Real)( 2. * M_PI  / dirCount );
	if( quadratureType & QUADRATURE_SQUARE_LENGTH )
	{
		Real sum = 0;
		for( int i=0 ; i<dirCount ; i++ )
		{
			Real l = Point2D< Real >::Dot( dirs[i] , tensor * dirs[i] );
			weights[i] *= l , sum += l;
		}
		for( int i=0 ; i<dirCount ; i++ ) weights[i] /= sum;
	}
}

template< class Real > SquareMatrix< Real , 2 > FEM::MakeConformal( const SquareMatrix< Real , 2 >& sourceTensor , const SquareMatrix< Real , 2 >& targetTensor ){ return targetTensor * (Real)sqrt( sourceTensor.determinant() / targetTensor.determinant() ); }
template< class Real > SquareMatrix< Real , 2 > FEM::MakeAuthalic ( const SquareMatrix< Real , 2 >& sourceTensor , const SquareMatrix< Real , 2 >& targetTensor ){ return sourceTensor * (Real)sqrt( targetTensor.determinant() / sourceTensor.determinant() ); }
template< class Real >
SquareMatrix< Real , 2 > FEM::TensorRoot( const SquareMatrix< Real , 2 >& tensor )
{
	// Code borrowed from: https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
	SquareMatrix< Real , 2 > root = tensor;
	Real det = tensor.determinant();
	if( det<0 ) fprintf( stderr , "[ERROR] Negative determinant: %g\n" , det ) , exit( 0 );
	Real s = (Real)sqrt( det );
	Real disc = (Real)( tensor.trace() + 2. * s );
	if( disc<0 ) fprintf( stderr , "[ERROR] Negative discriminant: %g\n" , disc ) , exit( 0 );
	root(0,0) += s , root(1,1) += s;
	return root / (Real)sqrt( disc );
}
////////////////////////
// FEM::RightTriangle //
////////////////////////
template< class Real > inline Real FEM::RightTriangle< Real >::EdgeDot( const SquareMatrix< Real , 2 >& tensor , int e1 , int e2 ){ return Point2D< Real >::Dot( Edges[e1] , tensor * Edges[e2] ); }
template< class Real > inline Real FEM::RightTriangle< Real >::SquareEdgeLength( const SquareMatrix< Real , 2 >& tensor , int e ){ return EdgeDot( tensor , e , e ); }
template< class Real > inline Real FEM::RightTriangle< Real >::EdgeLength( const SquareMatrix< Real , 2 >& tensor , int e ){ return (Real)sqrt( SquareEdgeLength( tensor , e ) ); }
template< class Real > inline Real FEM::RightTriangle< Real >::Dot( const SquareMatrix< Real , 2 >& tensor , Point2D< Real > v1 , Point2D< Real > v2 ){ return Point2D< Real >::Dot( v1 , tensor * v2 ); }
template< class Real > inline Real FEM::RightTriangle< Real >::SquareLength( const SquareMatrix< Real , 2 >& tensor , Point2D< Real > v ){ return Dot( tensor , v , v ); }
template< class Real > inline Real FEM::RightTriangle< Real >::Length( const SquareMatrix< Real , 2 >& tensor , Point2D< Real > v ){ return (Real)sqrt( SquareLength( tensor , v ) ); }
template< class Real >
inline Real FEM::RightTriangle< Real >::Angle( const SquareMatrix< Real , 2 >& tensor , int v )
{
	int v1 = (v+1)%3 , v2 = (v+2)%3;
	return (Real)acos( - EdgeDot( tensor , v1 , v2 ) / sqrt( SquareEdgeLength( tensor , v1 ) * SquareEdgeLength( tensor , v2 ) ) );
}
template< class Real >
inline Point2D< Real > FEM::RightTriangle< Real >::Center( const SquareMatrix< Real , 2 >& tensor , int dualType )
{
	static const double SQRT_THREE_QUARTERS = sqrt( 3./4 );
	switch( dualType )
	{
	case DUAL_BARYCENTRIC:
	case DUAL_CIRCUMCENTER_PROJECTED_BARYCENTRIC:
	case DUAL_ISOGON_PROJECTED_BARYCENTRIC:
		return Point2D< Real >( (Real)1./3 , (Real)1./3 );
	case DUAL_INCENTRIC:
		{
			Real lengths[] =
			{
				(Real)sqrt( Point2D< Real >::Dot( RightTriangle< Real >::Edges[0] , tensor * RightTriangle< Real >::Edges[0] ) ) ,
				(Real)sqrt( Point2D< Real >::Dot( RightTriangle< Real >::Edges[1] , tensor * RightTriangle< Real >::Edges[1] ) ) ,
				(Real)sqrt( Point2D< Real >::Dot( RightTriangle< Real >::Edges[2] , tensor * RightTriangle< Real >::Edges[2] ) )
			};
			Real lSum = lengths[0] + lengths[1] + lengths[2];
			return Point2D< Real >( lengths[1] / lSum , lengths[2] / lSum );
		}
	case DUAL_CIRCUMCENTRIC:
		{
			Real maxDet = 0;
			Point2D< Real > c;
			for( int j=0 ; j<3 ; j++ )
			{
				Point2D< Real > c1 = RightTriangle< Real >::EdgeMidpoints[(j+1)%3] , c2 = RightTriangle< Real >::EdgeMidpoints[(j+2)%3];
				Point2D< Real > v1 = Rotate90( tensor , RightTriangle< Real >::Edges[(j+1)%3] ) , v2 = Rotate90( tensor , RightTriangle< Real >::Edges[(j+2)%3] );
				// Solve for s and t such that:
				//		c1 + s * v1 = c2 + t * v2
				// =>	c1 - c2 = - s * v1 + t * v2
				// =>	c1 - c2 = ( -v1 | v2 ) * ( s , t )^t
				SquareMatrix< Real , 2 > M;
				M( 0 , 0 ) = -v1[0] , M( 0 , 1 ) = -v1[1];
				M( 1 , 0 ) =  v2[0] , M( 1 , 1 ) =  v2[1];
				Real det = (Real)fabs( M.determinant() );
				if( det>maxDet )
				{
					Point2D< Real > x = M.inverse() * ( c1 - c2 );
					c = ( c1 + v1 * x[0] + c2 + v2 * x[1] ) / 2;
					maxDet = det;
				}
			}
			return c;
		}
	case DUAL_ISOGONIC:
		{
			Point2D< Real > eVertices[3];
			// Solve for the positive value of t such that:
			//		|| Corners[j+1] - ( midPoint + t * perpDir ) ||^2 = || Corners[j+1] - Corners[j+2] ||^2
			// <=>	t^2 || perpDir ||^2 = || Corners[j+1] - Corners[j+2] ||^2 - || Corners[j+1] - midPoint ||^2
			// <=>	t^2 || Corners[j+1] - Corners[j+2] ||^2 = || Corners[j+1] - Corners[j+2] ||^2 - || ( Corners[j+1] - Corners[j+2] )/2 ||^2
			// <=>	t^2 || Corners[j+1] - Corners[j+2] ||^2 = 3/4 || Corners[j+1] - Corners[j+2] ||^2
			// <=>	t = sqrt( 3/4 )
			for( int j=0 ; j<3 ; j++ ) eVertices[j] = RightTriangle< Real >::EdgeMidpoints[j] - Rotate90( tensor ,  RightTriangle< Real >::Edges[j] ) * (Real)SQRT_THREE_QUARTERS;
			Real maxDet = 0;
			Point2D< Real > c;
			for( int j=0 ; j<3 ; j++ )
			{
				Point2D< Real > c1 = eVertices[(j+1)%3] , c2 = eVertices[(j+2)%3];
				Point2D< Real > v1 = RightTriangle< Real >::Corners[(j+1)%3] - c1 , v2 = RightTriangle< Real >::Corners[(j+2)%3] - c2;

				// Solve for s and t such that:
				//		c1 + s * v1 = c2 + t * v2
				// =>	c1 - c2 = - s * v1 + t * v2
				// =>	c1 - c2 = ( -v1 | v2 ) * ( s , t )^t
				SquareMatrix< Real , 2 > M;
				M( 0 , 0 ) = -v1[0] , M( 0 , 1 ) = -v1[1];
				M( 1 , 0 ) =  v2[0] , M( 1 , 1 ) =  v2[1];
				Real det = (Real)fabs( M.determinant() );
				if( det>maxDet )
				{
					Point2D< Real > x = M.inverse() * ( c1 - c2 );
					c = ( c1 + v1 * x[0] + c2 + v2 * x[1] ) / 2;
					maxDet = det;
				}
			}
			return c;
		}
	default:
		fprintf( stderr , "[ERROR] Unrecognized dual type: %d\n" , dualType ) , exit( 0 );
	}
}
template< class Real >
inline Point3D< Real > FEM::RightTriangle< Real >::SubTriangleAreas( const SquareMatrix< Real , 2 >& tensor , Point2D< Real > center )
{
	Point2D< Real > triangle[3];
	triangle[2] = center;
	Point3D< Real > areas;
	for( int i=0 ; i<3 ; i++ )
	{
		triangle[0] = RightTriangle< Real >::Corners[(i+1)%3] , triangle[1] = RightTriangle< Real >::Corners[(i+2)%3];
		areas[i] = Area( tensor , triangle );
	}
	return areas;
}
template< class Real >
inline Point3D< Real > FEM::RightTriangle< Real >::CenterAreas( const SquareMatrix< Real , 2 >& tensor , int dualType ){ return SubTriangleAreas( tensor , Center( tensor , dualType ) ); }

template< class Real >
inline Point2D< Real > FEM::RightTriangle< Real >::EdgeReflect( const SquareMatrix< Real , 2 >& tensor , int e , Point2D< Real > p )
{
	Point2D< Real > c = Corners[(e+1)%3] , v = p - c ,  perp = Rotate90( tensor , Edges[e] );
	return c + v - ( 2 * Point2D< Real >::Dot( perp , tensor * v ) / Point2D< Real >::Dot( perp , tensor * perp ) ) * perp;
}

template< class Real >
inline SquareMatrix< Real , 3 > FEM::RightTriangle< Real >::GetScalarMassMatrix( const SquareMatrix< Real , 2 >& tensor , bool lump )
{
	SquareMatrix< Real , 3 > mass;
	SetScalarMassMatrix( tensor , mass , lump );
	return mass;
}
template< class Real >
inline SquareMatrix< Real , 3 > FEM::RightTriangle< Real >::GetScalarStiffnessMatrix( const SquareMatrix< Real , 2 >& tensor )
{
	SquareMatrix< Real , 3 > stiffness;
	SetScalarStiffnessMatrix( tensor , stiffness );
	return stiffness;
}
template< class Real >
inline void FEM::RightTriangle< Real >::SetScalarMassMatrix( const SquareMatrix< Real , 2 >& tensor , SquareMatrix< Real , 3 >& massMatrix , bool lump )
{
	// F_0(x,y) = (1-x-y) ; grad( F_0 ) = ( -1 , -1 )
	// F_1(x,y) = x       ; grad( F_1 ) = (  1 ,  0 )
	// F_2(x,y) = y       ; grad( F_2 ) = (  0 ,  1 )

	// < F_0 , F_0 > = \int_0^1 \int_0^{1-y}( 1 + x^2 + y^2 - 2*x - 2*y + 2*x*y ) dx dy
	//               = \int_0^1 [ (1-y) + 1/3*(1-y)^3 + (1-y)*y^2 - (1-y)^2 - 2*(1-y)*y + (1-y)^2*y ] dy
	//               = \int_0^1 [ (1-y) + 1/3*(1-3*y+3*y^2-y^3) + (y^2-y^3) - (1-2*y+y^2) - 2*(y-y^2) + (y-2*y^2+y^3) ] dy
	//               = [ 1 - 1/2  +  1/3 - 1/2 + 1/3 - 1/12  +  1/3 - 1/4  -  1 + 1 - 1/3  -  1 + 2/3  +  1/2 - 2/3 + 1/4 ]
	//               = [ 1/3 - 1/2 + 1/3 - 1/12 ]
	//               = 1/12
	// < F_0 , F_1 > = \int_0^1 \int_0^{1-y}( x - x^2 - x*y ) dx dy
	//               = \int_0^1 [ 1/2*(1-y)^2 - 1/3*(1-y)^3 - 1/2*(1-y)^2*y ] dy
	//               = \int_0^1 [ 1/2*(1-2*y+y^2) - 1/3*(1-3*y+3*y^2-y*3) - 1/2*(y-2*y^2+y^3) ] dy
	//               = [ 1/2 - 1/2 + 1/6 - 1/3 + 1/2 - 1/3 + 1/12 - 1/4 + 1/3 - 1/8 ] dy
	//               = [ 1/6 + 1/4 - 1/3 + 1/12 - 1/8 ]
	//               = 1/24
	// < F_1 , F_1 > = \int_0^1 \int_0^{1-y}( x^2 ) dx dy
	//               = \int_0^1 [ 1/3*(1-y)^3 ] dy
	//               = \int_0^1 [ 1/3*(1-3*y+3*y^2-y^3) ] dy
	//               = [ 1/3 - 1/2 + 1/3 - 1/12 ]
	//               = 1/12
	// < F_1 , F_2 > = \int_0^1 \int_0^{1-y}( x*y ) dx dy
	//               = \int_0^1 [ 1/2*(1-y)^2*y ] dy
	//               = \int_0^1 [ 1/2*(y-2*y^2+y^3) ] dy
	//               = [ 1/4 - 1/3 + 1/8 ]
	//               = 1/24

	if( !tensor.determinant() )
	{
		fprintf( stderr , "[WARNING] Vanishing metric tensor determinant\n" );
		massMatrix *= 0;
		return;
	}
	massMatrix(0,0) = massMatrix(1,1) = massMatrix(2,2) = ( lump ? 1./6 : 1./12 );
	massMatrix(1,0) = massMatrix(0,1) = massMatrix(1,2) = massMatrix(2,1) = massMatrix(0,2) = massMatrix(2,0) = ( lump ? 0. : 1./24 );

	massMatrix *= (Real)sqrt( tensor.determinant() );
}
template< class Real >
inline void FEM::RightTriangle< Real >::SetScalarStiffnessMatrix( const SquareMatrix< Real , 2 >& tensor , SquareMatrix< Real , 3 >& stiffnessMatrix )
{
	if( !tensor.determinant() )
	{
		fprintf( stderr , "[WARNING] Vanishing metric tensor determinant\n" );
		stiffnessMatrix *= 0;
		return;
	}
	SquareMatrix< Real , 2 > iTensor = tensor.inverse();
	Point2D< Real > grad[3];
	grad[0][0] = -1. , grad[0][1] = -1.;
	grad[1][0] =  1. , grad[1][1] =  0.;
	grad[2][0] =  0. , grad[2][1] =  1.;
	for( int i=0 ; i<3 ; i++ ) for( int j=0 ; j<3 ; j++ ) stiffnessMatrix(i,j) = Point2D< Real >::Dot( grad[i] , iTensor * grad[j] ) / 2.;

	stiffnessMatrix *= (Real)sqrt( tensor.determinant() );
}
template< class Real >
inline Point2D< Real > FEM::RightTriangle< Real >::Gradient( const SquareMatrix< Real , 2 >& tensor , const Real values[3] ){ return tensor.inverse() * Point2D< Real >( values[1]-values[0] , values[2]-values[0] ); }



/////////////////////////
// FEM::RiemannianMesh //
/////////////////////////
template< class Real >
inline int FEM::RiemannianMesh< Real >::vCount( void ) const
{
	int count = 0;
	for( unsigned int i=0 ; i<tCount ; i++ ) for( int j=0 ; j<3 ; j++ ) count = std::max< int >( count , triangles[i][j] );
	return count+1;
}

template< class Real >
FEM::RiemannianMesh< Real > FEM::RiemannianMesh< Real >::subdivide( void ) const
{
#define EDGE_KEY( i1 , i2 ) ( (i1)>(i2) ? ( ( (long long) (i1) )<<32 ) | ( (long long) (i2) ) : ( ( (long long) (i2) )<<32 ) | ( (long long) (i1) ) )

	RiemannianMesh mesh;
	mesh.tCount = tCount * 4;
	mesh.triangles = AllocPointer< TriangleIndex >( tCount*4 );
	if( g ) mesh.g = AllocPointer< SquareMatrix< Real , 2 > >( tCount*4 );

	int vertexCount = vCount();
	std::unordered_map< long long , int > vMap;
	for( int i=0 ; i<tCount ; i++ )
	{
		long long keys[] = { EDGE_KEY( triangles[i][1] , triangles[i][2] ) , EDGE_KEY( triangles[i][2] , triangles[i][0] ) , EDGE_KEY( triangles[i][0] , triangles[i][1] ) };
		int eIndex[3];
		for( int j=0 ; j<3 ; j++ )
			if( vMap.find( keys[j] )==vMap.end() ) vMap[ keys[j] ] = eIndex[j] = vertexCount++;
			else eIndex[j] = vMap[ keys[j] ];
		mesh.triangles[4*i+0] = TriangleIndex( eIndex[0] , eIndex[1] , eIndex[2] );
		mesh.triangles[4*i+1] = TriangleIndex( triangles[i][0] , eIndex[2] , eIndex[1] );
		mesh.triangles[4*i+2] = TriangleIndex( eIndex[2] , triangles[i][1] , eIndex[0] );
		mesh.triangles[4*i+3] = TriangleIndex( eIndex[1] , eIndex[0] , triangles[i][2] );
		if( g ) mesh.g[4*i] = mesh.g[4*i+1] = mesh.g[4*i+2] = mesh.g[4*i+3] = g[i] / (Real)4;
	}
	return mesh;
#undef EDGE_KEY
}

template< class Real >
Pointer( FEM::EdgeXForm< Real > ) FEM::RiemannianMesh< Real >::getEdgeXForms( void ) const
{
	Pointer( EdgeXForm< Real > ) edgeXForms = NewPointer< EdgeXForm< Real > >( tCount*3 );
	setEdgeXForms( edgeXForms );
	return edgeXForms;
}
template< class Real >
void FEM::RiemannianMesh< Real >::_setEdgeXForm( int edge , Pointer( EdgeXForm< Real > ) edgeXForms ) const
{
	static const Point2D< Real > tVerts[] = { Point2D< Real >( (Real)0 , (Real)0 ) , Point2D< Real >( (Real)1 , (Real)0 ) , Point2D< Real >( (Real)0 , (Real)1 ) };
	int oEdge = edgeXForms[edge].oppositeEdge;
	if( oEdge==-1 ) fprintf( stderr , "[ERROR] Boundary edge (TriangleMesh::unfold)\n" ) , exit( 0 );

	// The two triangles on this edge
	int tIdx[] = { edge/3 , oEdge/3 };

	// The end-points of the edge
	int  v[] = { ( edge+1)%3 , ( edge+2)%3 };
	int ov[] = { (oEdge+1)%3 , (oEdge+2)%3 };

	// The direction of the edge
	Point2D< Real > edgeDir = tVerts[ v[1] ] - tVerts[ v[0] ] , oEdgeDir = -( tVerts[ ov[1] ] - tVerts[ ov[0] ] );
#if 1
	// Normalize both directions in case the metrics don't agree on the shared edge
	edgeDir /= (Real)sqrt( Point2D< Real >::Dot( edgeDir , g[tIdx[0]] * edgeDir ) );
	oEdgeDir /= (Real)sqrt( Point2D< Real >::Dot( oEdgeDir , g[tIdx[1]] * oEdgeDir ) );
#endif
	// The perpendicular direction to the edge
	Point2D< Real > edgePerpDir = Rotate90( g[ tIdx[0] ] , edgeDir ) , oEdgePerpDir = Rotate90( g[ tIdx[1] ] , oEdgeDir );
		
	// The linear part of the transformation should map ( edgeDir , edgePerpDir ) -> ( oEdgeDir , oEdgePerpDir )
	{
		SquareMatrix< Real , 2 > M , oM;
		M( 0 , 0 ) =     edgeDir[0] , M( 0 , 1 ) =     edgeDir[1];
		M( 1 , 0 ) = edgePerpDir[0] , M( 1 , 1 ) = edgePerpDir[1];
		oM( 0 , 0 ) =     oEdgeDir[0] , oM( 0 , 1 ) =     oEdgeDir[1];
		oM( 1 , 0 ) = oEdgePerpDir[0] , oM( 1 , 1 ) = oEdgePerpDir[1];
		edgeXForms[edge].xForm.linear = oM * M.inverse();
	}

#if 1
	// The transformation should also take ( v[0] + v[1] ) / 2 to ( ov[0] + ov[1] ) / 2
	edgeXForms[edge].xForm.constant = ( tVerts[ ov[0] ] + tVerts[ ov[1] ] - edgeXForms[edge].xForm.linear * ( tVerts[ v[0] ] + tVerts[ v[1] ] ) ) / (Real)2;
#else
	// The transformation should also take v[1] to ov[0]
	edgeXForms[edge].xForm.constant = tVerts[ ov[0] ] - edgeXForms[edge].xForm.linear * tVerts[ v[1] ];
#endif
}
template< class Real >
void FEM::RiemannianMesh< Real >::setEdgeXForms( Pointer( EdgeXForm< Real > ) edges ) const
{
	std::unordered_map< long long , int > edgeMap;
	for( int t=0 ; t<tCount ; t++ ) for( int v=0 ; v<3 ; v++ )
	{
		int idx = t*3 + ( (v+2)%3 );
		long long key = HalfEdgeKey( triangles[t][v] , triangles[t][(v+1)%3] );
		if( edgeMap.find(key)!=edgeMap.end() ) fprintf( stderr , "[ERROR] Edge is occupied\n" ) , exit( 0 );
		edgeMap[ key ] = idx;
	}
#pragma omp parallel for
	for( int t=0 ; t<tCount ; t++ ) for( int v=0 ; v<3 ; v++ )
	{
		int idx = t*3 + ( (v+2)%3 );
		std::unordered_map< long long , int >::const_iterator iter = edgeMap.find( HalfEdgeKey( triangles[t][(v+1)%3] , triangles[t][v] ) );
		if( iter==edgeMap.end() ) edges[idx].oppositeEdge = -1;
		else
		{
			edges[idx].oppositeEdge = iter->second;
			_setEdgeXForm( idx , edges );
		}
	}
}
template< class Real >
bool FEM::RiemannianMesh< Real >::edgeFlip( Pointer( EdgeXForm< Real > ) edges , int edge , Real eps )
{
	static const Point2D< Real > Corners[] = { Point2D< Real >(0,0) , Point2D< Real >(1,0) , Point2D< Real >(0,1) };
	int oEdge = edges[ edge ].oppositeEdge;
	int  t =  edge / 3 ,  v =  edge % 3;
	int ot = oEdge / 3 , ov = oEdge % 3;

	// First test that the edge is not on the boundary
	if( oEdge==-1 ) return false;

	// Get the coordinates of the old and new edges and test that if we can flip
	Point2D< Real > newEdge[] = { Corners[v] , edges[ oEdge ].xForm( Corners[ ov ] ) };
	Point2D< Real > oldEdge[] = { Corners[(v+1)%3] , Corners[(v+2)%3] };
	{

		// Solve for the point of intersection between the two corresponding line segments
		//  => Solve for s and t such that:
		//		newEdge[0] + s * ( newEdge[1]-newEdge[0]) = oldEdge[0] + t * ( oldEdge[1] - oldEdge[0] )
		// <=>	( newEdge[1]-newEdge[0] | -oldEdge[1]+oldEdge[0] ) * (s,t)^t = oldEdge[0] - newEdge[0]
		SquareMatrix< Real , 2 > M;
		for( int i=0 ; i<2 ; i++ ) M(0,i) = newEdge[1][i]-newEdge[0][i] , M(1,i) = -oldEdge[1][i]+oldEdge[0][i];
		Point2D< Real > st = M.inverse() * ( oldEdge[0] - newEdge[0] );

		// Test that the intersection point is in bounds
		if( st[0]<=eps || st[0]>=1-eps || st[1]<=eps || st[1]>=1-eps ) return false;
	}

	//     o              2,1
	//    / \             /|\
	//   /   \           / | \
	//  /     \         /  |  \
	// |-------|  =>  0| o | t |0
	//  \  e  /         \  |  /
	//   \   /           \ | /
	//    \ /             \|/
	//     v              1,2

	// The new triangles
	TriangleIndex tris[] = { TriangleIndex( triangles[t][(v+1)%3] , triangles[ot][ov] , triangles[t][v] ) , TriangleIndex( triangles[t][(v+2)%3] , triangles[t][v] , triangles[ot][ov] ) };

	// The new metrics
	SquareMatrix< Real , 2 > tensors[2];
	tensors[0](0,0) = Point2D< Real >::Dot( Corners[ov] - Corners[(ov+2)%3] , g[ot] * ( Corners[ov] - Corners[(ov+2)%3] ) );
	tensors[0](1,1) = Point2D< Real >::Dot( Corners[ v] - Corners[( v+1)%3] , g[ t] * ( Corners[ v] - Corners[( v+1)%3] ) );
	tensors[0](1,0) = tensors[0](0,1) = ( tensors[0](0,0) + tensors[0](1,1) - Point2D< Real >::Dot( newEdge[1]-newEdge[0] , g[t] * ( newEdge[1]-newEdge[0] ) ) ) / (Real)2.;

	tensors[1](0,0) = Point2D< Real >::Dot( Corners[ v] - Corners[( v+2)%3] , g[ t] * ( Corners[ v] - Corners[( v+2)%3] ) );
	tensors[1](1,1) = Point2D< Real >::Dot( Corners[ov] - Corners[(ov+1)%3] , g[ot] * ( Corners[ov] - Corners[(ov+1)%3] ) );
	tensors[1](1,0) = tensors[1](0,1) = ( tensors[1](0,0) + tensors[1](1,1) - Point2D< Real >::Dot( newEdge[1]-newEdge[0] , g[t] * ( newEdge[1]-newEdge[0] ) ) ) / (Real)2.;

	// Update the adjacencies of the neighboring triangles
	if( edges[ t*3+( v+1)%3].oppositeEdge!=-1 ) edges[ edges[ t*3+( v+1)%3 ].oppositeEdge ].oppositeEdge = 3*ot + 2;
	if( edges[ot*3+(ov+1)%3].oppositeEdge!=-1 ) edges[ edges[ot*3+(ov+1)%3 ].oppositeEdge ].oppositeEdge = 3* t + 2;
	if( edges[ t*3+( v+2)%3].oppositeEdge!=-1 ) edges[ edges[ t*3+( v+2)%3 ].oppositeEdge ].oppositeEdge = 3* t + 1;
	if( edges[ot*3+(ov+2)%3].oppositeEdge!=-1 ) edges[ edges[ot*3+(ov+2)%3 ].oppositeEdge ].oppositeEdge = 3*ot + 1;

	// Update the adjacencies of the current triangles
	int oldAdjacencies[][3] = { { edges[ t*3 ].oppositeEdge , edges[ t*3+1 ].oppositeEdge , edges[ t*3+2 ].oppositeEdge } , { edges[ot*3].oppositeEdge , edges[ot*3+1].oppositeEdge , edges[ot*3+2].oppositeEdge } };
	edges[3* t  ].oppositeEdge = 3*ot;
	edges[3* t+1].oppositeEdge = oldAdjacencies[0][( v+2)%3];
	edges[3* t+2].oppositeEdge = oldAdjacencies[1][(ov+1)%3];
	edges[3*ot  ].oppositeEdge = 3* t;
	edges[3*ot+1].oppositeEdge = oldAdjacencies[1][(ov+2)%3];
	edges[3*ot+2].oppositeEdge = oldAdjacencies[0][( v+1)%3];

	// Set the triangles and metrics
	triangles[t] = tris[0] , triangles[ot] = tris[1];
	g[t] = tensors[0] , g[ot] = tensors[1];

	// Update the transforms
	for( int j=0 ; j<3 ; j++ )
	{
		if( edges[3* t+j].oppositeEdge!=-1 ) _setEdgeXForm( 3* t+j , edges ) , _setEdgeXForm( edges[3* t+j].oppositeEdge , edges );
		if( edges[3*ot+j].oppositeEdge!=-1 ) _setEdgeXForm( 3*ot+j , edges ) , _setEdgeXForm( edges[3*ot+j].oppositeEdge , edges );
	}
	return true;
}
template< class Real >
void FEM::RiemannianMesh< Real >::sanityCheck( ConstPointer( EdgeXForm< Real > ) edges , Real eps ) const
{
	static const Point2D< Real > Corners[] = { Point2D< Real >(0,0) , Point2D< Real >(1,0) , Point2D< Real >(0,1) };
	for( int t=0 ; t<tCount ; t++ )
	{
		bool success = true;

		// Check that the metric tensor is symmetric
		if( fabs( g[t](1,0) - g[t](0,1) )>eps ) fprintf( stderr , "[ERROR] Metric tensor is not symetric: %g != %g\n" , g[t](1,0) , g[t](0,1) ) , success = false;

		// The characteristic polynomial of the metric tensor is:
		// P(x) = x^2 - Tr * x + det
		double a = 1. , b = -g[t].trace() , c =g[t].determinant();
		double disc = b*b - 4.*a*c;
		if( disc<0 ) fprintf( stderr , "[ERROR] Vanishing discriminant: %g\n" , disc ) , success = false;
		else
		{
			double x[] = { ( -b - sqrt( disc ) ) / (2.*a) , ( -b + sqrt( disc ) ) / (2.*a) };
			if( x[0]<=0 ) fprintf( stderr , "[ERROR] Metric tensor is not positive definite: %g %g\n" , x[0] , x[1] ) , success = false;
		}

		if( !success ) fprintf( stderr , "[ERROR] Failure on triangle: %d\n" , t ) , exit( 0 );
	}
	for( int e=0 ; e<tCount*3 ; e++ ) if( edges[e].oppositeEdge!=-1 )
	{
		bool success = true;
		int oe = edges[e].oppositeEdge;
		int t = e/3 , v = e%3 , ot = oe/3 , ov = oe%3;

		// Test the opposite of the opposite is itself
		if( e!=edges[oe].oppositeEdge ) fprintf( stderr , "[ERROR] edge is not the opposite of its opposite: %d != %d\n" , e , edges[oe].oppositeEdge ) , success = false;

		// Test that the shared vertices are the same (assumes oriented)
		if( triangles[t][(v+1)%3]!=triangles[ot][(ov+2)%3] || triangles[t][(v+2)%3]!=triangles[ot][(ov+1)%3] )
			fprintf( stderr , "[ERROR] Vertices don't match: %d %d != %d %d\n" , triangles[t][(v+1)%3] , triangles[t][(v+2)%3] , triangles[ot][(ov+2)%3] , triangles[ot][(ov+1)%3] ) , success = false;

		// Test that transforming across the edge and then back gives the identity
		{
			FEM::CoordinateXForm< Real > xForm = edges[oe].xForm * edges[e].xForm;
			Real cError = ( xForm.constant ).squareNorm() , lError = ( xForm.linear - SquareMatrix< Real , 2 >::Identity() ).squareNorm();
			if( cError > eps*eps || lError > eps*eps ) fprintf( stderr , "[ERROR] edge transformations are not inverses: %g %g\n" , cError , lError ) , success = false;
		}

		// Test that the lengths of the shared edge agree
		{
			Real l1 = (Real)sqrt( Point2D< Real >::Dot( Corners[( v+1)%3] - Corners[( v+2)%3] , g[ t] * ( Corners[( v+1)%3] - Corners[( v+2)%3] ) ) );
			Real l2 = (Real)sqrt( Point2D< Real >::Dot( Corners[(ov+1)%3] - Corners[(ov+2)%3] , g[ot] * ( Corners[(ov+1)%3] - Corners[(ov+2)%3] ) ) );
			if( fabs( l1-l2 )>eps ) fprintf( stderr , "[ERROR] edge lengths don't agree: %g != %g\n" , l1 , l2 ) , success = false;
		}

		// Test that the edges map into each other
		{
			Point2D< Real > e1 = Corners[(v+1)%3] - Corners[(v+2)%3];
			Point2D< Real > e2 = -edges[oe].xForm.linear * ( Corners[(ov+1)%3] - Corners[(ov+2)%3]  );
			if( Point2D< Real >::Dot( e1-e2 , g[t]*(e1-e2) )>eps*eps ) fprintf( stderr , "[ERROR] edges don't match: %g %g %g\n" , e1[0]-e2[0] , e1[1]-e2[1] , e1[2]-e2[2] ) , success = false;
		}

		// Test that the perpendiculars map into each other
		{
			Point2D< Real > p1 = Rotate90( g[t] , Corners[(v+1)%3] - Corners[(v+2)%3] );
			Point2D< Real > p2 = -edges[oe].xForm.linear * Rotate90( g[ot] , Corners[(ov+1)%3] - Corners[(ov+2)%3] );
			if( Point2D< Real >::Dot( p1-p2 , g[t]*(p1-p2) )>eps*eps ) fprintf( stderr , "[ERROR] edge perps don't match: %g %g %g\n" , p1[0]-p2[0] , p1[1]-p2[1] , p1[2]-p2[2] ) , success = false;
		}

		if( !success ) fprintf( stderr , "[ERROR] Failure on edge between triangles: %d %d\n" , t , ot ) , exit( 0 );
	}
}

template< class Real >
bool FEM::RiemannianMesh< Real >::isVoronoiEdge( ConstPointer( EdgeXForm< Real > ) edges , int e , Real eps ) const
{
	static const Point2D< Real > Corners[] = { Point2D< Real >(0,0) , Point2D< Real >(1,0) , Point2D< Real >(0,1) };
	int oe = edges[e].oppositeEdge;
	int  t =  e / 3 ,  v =  e % 3;
	int ot = oe / 3 , ov = oe % 3;
	if( oe==-1 ) return true;
	Point2D< Real > center = RightTriangle< Real >::Center( g[t] , RightTriangle< Real >::DUAL_CIRCUMCENTRIC ) , oVertex = edges[oe].xForm( Corners[ov] );
	return Point2D< Real >::Dot( center - oVertex , g[t] * ( center - oVertex ) )+eps > Point2D< Real >::Dot( center - Corners[0] , g[t] * ( center - Corners[0] ) );
}

template< class Real >
FEM::CoordinateXForm< Real > FEM::RiemannianMesh< Real >::getVertexXForm( ConstPointer( EdgeXForm< Real > ) edgeXForms , int t , int v ) const
{
	const int VertexToEdgeMap[] = { 1 , 2 , 0 };
	const int EdgeToVertexMap[] = { 1 , 2 , 0 };
	// Assume that the mesh is oriented
	FEM::CoordinateXForm< Real > xForm;
	int currentT = t , currentV = v;
	do
	{
		int edge = currentT*3 + VertexToEdgeMap[ currentV ] , oEdge = edgeXForms[edge].oppositeEdge;
		xForm = edgeXForms[ edge ].xForm * xForm;
		if( oEdge==-1 ) fprintf( stderr , "[ERROR] Boundary vertex\n" ) , exit( 0 );
		currentT = oEdge / 3;
		currentV = EdgeToVertexMap[ oEdge%3 ];
	}
	while( currentT!=t );
	return xForm;
}

template< class Real >
std::vector< int > FEM::RiemannianMesh< Real >::getVertexCorners( ConstPointer( EdgeXForm< Real > ) edgeXForms , int t , int v ) const
{
	// Circulate CCW
	const int VertexToEdgeMap[] = { 1 , 2 , 0 };
	const int EdgeToVertexMap[] = { 1 , 2 , 0 };
	// Assume that the mesh is oriented
	std::vector< int > neighbors;
	int currentT = t , currentV = v;
	do
	{
		int edge = currentT*3 + VertexToEdgeMap[ currentV ] , oEdge = edgeXForms[edge].oppositeEdge;
		neighbors.push_back( currentT*3 + currentV );
		if( oEdge==-1 ) fprintf( stderr , "[ERROR] Boundary vertex\n" ) , exit( 0 );
		currentT = oEdge / 3;
		currentV = EdgeToVertexMap[ oEdge%3 ];
	}
	while( currentT!=t );
	return neighbors;
}
template< class Real >
Real FEM::RiemannianMesh< Real >::getVertexConeAngle( ConstPointer( EdgeXForm< Real > ) edgeXForms , int t , int v ) const
{
	const int VertexToEdgeMap[] = { 1 , 2 , 0 };
	const int EdgeToVertexMap[] = { 1 , 2 , 0 };
	// Assume that the mesh is oriented
	Real angle = (Real)0;
	int currentT = t , currentV = v;
	do
	{
		int edge = currentT*3 + VertexToEdgeMap[ currentV ] , oEdge = edgeXForms[edge].oppositeEdge;
		angle += RightTriangle< Real >::Angle( g[currentT] , currentV );
		if( oEdge==-1 ) fprintf( stderr , "[ERROR] Boundary vertex\n" ) , exit( 0 );
		currentT = oEdge / 3;
		currentV = EdgeToVertexMap[ oEdge%3 ];
	}
	while( currentT!=t );
	return angle;
}

template< class Real >
FEM::CoordinateXForm< Real > FEM::RiemannianMesh< Real >::exp( ConstPointer( EdgeXForm< Real > ) edgeXForms , HermiteSamplePoint< Real >& p , Real eps ) const
{
	HermiteSamplePoint< Real > startP = p;
	CoordinateXForm< Real > xForm = CoordinateXForm< Real >::Identity();
	if( !Point2D< Real >::SquareNorm( p.v ) ) return xForm;
	const int MAX_ITERS = 10000;
	int count = 0;
	int inEdge = -1;
#if 1
	// If the starting point happens to be on an edge
	{
		int idx = -1;
		if     ( p.p[0]<=0 && p.v[0]<0 ) idx = 1;
		else if( p.p[1]<=0 && p.v[1]<0 ) idx = 2;
		else if( p.p[0]+p.p[1]>=1 && p.v[0]+p.v[1]>0 ) idx = 0;
		if( idx!=-1 )
		{
			const EdgeXForm< Real >& edge = edgeXForms[ p.tIdx*3 + idx ];
			p.tIdx = edge.oppositeEdge/3;
			p.p = edge.xForm( p.p ) ; p.v = edge.xForm.linear * p.v;
			inEdge = edge.oppositeEdge%3;
			xForm = edge.xForm * xForm;
		}
	}
#endif
	while( count<MAX_ITERS )
	{
		// Intersect the ray p + s*v with each of the three edges
		// Bottom edge:   p[1] + s * v[1] = 0                         => s = -p[1]/v[1]
		// Left edge:     p[0] + s * v[0] = 0                         => s = -p[0]/v[0]
		// Diagonal edge: p[1] + s * v[1] = 1 - ( p[0] + s * v[0] )   => s = ( 1 - p[0]  - p[1] ) / ( v[1] + v[0] )
		Real maxS = 0;
		int idx = -1;
		{
			Real s[] = { -p.p[1] / p.v[1]  , -p.p[0] / p.v[0] , ( Real(1.) - p.p[0]  - p.p[1] ) / ( p.v[1] + p.v[0] ) };
			if( inEdge!=2 && s[0]>0 ){ Real foo = p.p[0] + p.v[0] * s[0] ; if( foo>=-eps && foo<=1+eps ) if( s[0]>maxS ) idx = 2 , maxS = s[0]; }
			if( inEdge!=1 && s[1]>0 ){ Real foo = p.p[1] + p.v[1] * s[1] ; if( foo>=-eps && foo<=1+eps ) if( s[1]>maxS ) idx = 1 , maxS = s[1]; }
			if( inEdge!=0 && s[2]>0 ){ Real foo = p.p[0] + p.v[0] * s[2] ; if( foo>=-eps && foo<=1+eps ) if( s[2]>maxS ) idx = 0 , maxS = s[2]; }
		}
		if( idx==-1 )
		{
			fprintf( stderr , "[ERROR] FEM::Mesh::exp:\n" );
			fprintf( stderr , "        Ray does not intersect triangle[%d]: p=(%f %f) v=(%g %g) [%g/%g]\n" , count , p.p[0] , p.p[1] , p.v[0] , p.v[1] , Point2D< Real >::SquareNorm(p.v) , eps*eps );
			fprintf( stderr , "                             Started at[%d]: p=(%f %f) v=(%g %g)\n" , 0 , startP.p[0] , startP.p[1] , startP.v[0] , startP.v[1] );
			exit( 0 );
		}
		if( maxS>1 ) // The end-point is within the triangle
		{
			p.p += p.v , p.v -= p.v;
			return xForm;
		}
		else // The end-point is outside the triangle
		{
			const EdgeXForm< Real >& edge = edgeXForms[ p.tIdx*3 + idx ];

			p.p += p.v*maxS ; p.v -= p.v*maxS ; p.tIdx = edge.oppositeEdge/3;
			p.p = edge.xForm( p.p ) ; p.v = edge.xForm.linear * p.v;
			inEdge = edge.oppositeEdge%3;
			xForm = edge.xForm * xForm;
		}
		count++;
	}
	fprintf( stderr , "[WARNING] Failed to converge exp after %d iterations\n" , MAX_ITERS );
	return xForm;
}

template< class Real >
FEM::CoordinateXForm< Real > FEM::RiemannianMesh< Real >::flow( ConstPointer( EdgeXForm< Real > ) edgeXForms , ConstPointer( Point2D< Real > ) vf , Real flowTime , SamplePoint< Real >& p , Real minStepSize , Real eps , std::vector< SamplePoint< Real > >* path ) const
{
	CoordinateXForm< Real > xForm = CoordinateXForm< Real >::Identity();
	int MAX_ITERS = 1000000;
	int count = 0;
	int inEdge = -1;
	Real direction = (flowTime<0) ? (Real)-1. : (Real)1.;
	Real stepSizeLeft = minStepSize;
	Point2D< Real > v = vf[ p.tIdx ] * direction;
	flowTime *= direction;
	if( path ) path->push_back( p );
	while( count<MAX_ITERS )
	{
		if( !Point2D< Real >::SquareNorm( v ) ) return xForm;
		// Intersect the ray p + s * v with each of the three edges
		// Bottom edge:   p[1] + s * v[1] = 0                         => s = -p[1]/v[1]
		// Left edge:     p[0] + s * v[0] = 0                         => s = -p[0]/v[0]
		// Diagonal edge: p[1] + s * v[1] = 1 - ( p[0] + s * v[0] )   => s = ( 1 - p[0]  - p[1] ) / ( v[1] + v[0] )
		Real s = 0;
		int idx = -1;
		{
			Real _s[] = { -p.p[1] / v[1]  , -p.p[0] / v[0] , ( Real(1.) - p.p[0]  - p.p[1] ) / ( v[1] + v[0] ) };
			if( inEdge!=2 && _s[0]>0 ){ Real foo = p.p[0] + v[0] * _s[0] ; if( foo>=-eps && foo<=1+eps ) if( _s[0]>s ) idx = 2 , s = _s[0]; }
			if( inEdge!=1 && _s[1]>0 ){ Real foo = p.p[1] + v[1] * _s[1] ; if( foo>=-eps && foo<=1+eps ) if( _s[1]>s ) idx = 1 , s = _s[1]; }
			if( inEdge!=0 && _s[2]>0 ){ Real foo = p.p[0] + v[0] * _s[2] ; if( foo>=-eps && foo<=1+eps ) if( _s[2]>s ) idx = 0 , s = _s[2]; }
		}
#if 0
		if( idx==-1 )
		{
			fprintf( stderr , "[ERROR] Ray does not intersect triangle[%d]: (%f %f) (%g %g) [%g/%g]\n" , count , p.p[0] , p.p[1] , v[0] , v[1] , Point2D< Real >::SquareNorm(v) , eps*eps );
			Real s[] = { -p.p[1] / v[1]  , -p.p[0] / v[0] , ( Real(1.) - p.p[0]  - p.p[1] ) / ( v[1] + v[0] ) };
			if( inEdge!=2 ) { Real foo = p.p[0] + v[0] * s[0] ; printf( "\t0] %g -> %f\n" , s[0] , foo ); }
			if( inEdge!=1 ) { Real foo = p.p[1] + v[1] * s[1] ; printf( "\t1] %g -> %f\n" , s[1] , foo ); }
			if( inEdge!=0 ) { Real foo = p.p[0] + v[0] * s[2] ; printf( "\t2] %g -> %f\n" , s[2] , foo ); }
			exit( 0 );
		}
#else
		if( idx==-1 ) return xForm;
#endif
		Real squareStepSize = Point2D< Real >::Dot( v , g[p.tIdx] * v ) * s * s;
		bool updateVector = false;
		if( minStepSize>0 && squareStepSize>stepSizeLeft*stepSizeLeft )
		{
			s = stepSizeLeft / (Real)sqrt( Point2D< Real >::Dot( v , g[p.tIdx] * v ) );
			updateVector = true;
		}

		// If we can finish the flow
		if( flowTime<s )
		{
			p.p += v * flowTime;
			if( path ) path->push_back( p );
			return xForm;
		}
		// If we do not cross a boundary, change direction
		else if( updateVector )
		{
			p.p += v * s , flowTime -= s;

			// If the the vectors are oppositely oriented, terminate the flow
			if( Point2D< Real >::Dot( v , g[p.tIdx] * vf[p.tIdx] )*direction < 0 ) return xForm;

			v = vf[ p.tIdx ] * direction;
			stepSizeLeft = minStepSize;
			if( path ) path->push_back( p );
			inEdge = -1;
		}
		// If we cross the boundary, transport the direction of the flow
		else // The end-point is outside the triangle
		{
			// Advance along the flow until you hit the edge
			p.p += v*s , flowTime -= s;

			const EdgeXForm< Real >& edge = edgeXForms[ p.tIdx*3 + idx ];
			// Switch into the next triangle
			p.tIdx = edge.oppositeEdge/3;
			p.p = edge.xForm( p.p );
			v = edge.xForm.linear * v;

			// Mark the edge we came in on
			inEdge = edge.oppositeEdge%3;

			// Accumulate the transformations
			xForm = edge.xForm * xForm;

			stepSizeLeft -= (Real)sqrt( squareStepSize );
			if( path ) path->push_back( p );
		}
		count++;
	}
	fprintf( stderr , "[WARNING] Failed to converge flow after %d iterations\n" , MAX_ITERS );
	return xForm;
}



template< class Real >
FEM::CoordinateXForm< Real > FEM::RiemannianMesh< Real >::whitneyFlow(ConstPointer(EdgeXForm< Real >) edgeXForms, ConstPointer(Real) ce, Real flowTime, SamplePoint< Real >& p, Real minStepSize, Real eps, std::vector< SamplePoint< Real > >* path) const
{
	CoordinateXForm< Real > xForm = CoordinateXForm< Real >::Identity();
	int MAX_ITERS = 1000000;
	int count = 0;
	int inEdge = -1;
	Real direction = (flowTime<0) ? (Real)-1. : (Real)1.;
	Real stepSizeLeft = minStepSize;

	auto GetWhitneyVector = [&](SamplePoint< Real >& _p){
		//Real vertexValues[3] = { _p.p[1] * ce[3 * _p.tIdx + 1] - _p.p[0] * ce[3 * _p.tIdx + 2],
		//	(1 - _p.p[0] - _p.p[1])*ce[3 * _p.tIdx + 2] - _p.p[1] * ce[3 * _p.tIdx],
		//	_p.p[0] * ce[3 * _p.tIdx] - (1 - _p.p[0] - _p.p[1]) * ce[3 * _p.tIdx + 1] };
		//return gInverse[_p.tIdx] * Point2D< Real >(vertexValues[1] - vertexValues[0], vertexValues[2] - vertexValues[0]);
		return gInv[_p.tIdx] * Point2D< Real >(ce[3 * _p.tIdx + 2] * (1 - _p.p[1]) - _p.p[1] * (ce[3 * _p.tIdx + 1] + ce[3 * _p.tIdx]), _p.p[0] * (ce[3 * _p.tIdx] + ce[3 * _p.tIdx + 2]) - (1 - _p.p[0]) * ce[3 * _p.tIdx + 1]);
	};

	Point2D< Real > v = GetWhitneyVector(p) * direction;
	flowTime *= direction;
	if (path) path->push_back(p);
	while (count<MAX_ITERS)
	{
		if (!Point2D< Real >::SquareNorm(v)) return xForm;
		// Intersect the ray p + s * v with each of the three edges
		// Bottom edge:   p[1] + s * v[1] = 0                         => s = -p[1]/v[1]
		// Left edge:     p[0] + s * v[0] = 0                         => s = -p[0]/v[0]
		// Diagonal edge: p[1] + s * v[1] = 1 - ( p[0] + s * v[0] )   => s = ( 1 - p[0]  - p[1] ) / ( v[1] + v[0] )
		Real s = 0;
		int idx = -1;
		{
			Real _s[] = { -p.p[1] / v[1], -p.p[0] / v[0], (Real(1.) - p.p[0] - p.p[1]) / (v[1] + v[0]) };
			if (inEdge != 2 && _s[0]>0){ Real foo = p.p[0] + v[0] * _s[0]; if (foo >= -eps && foo <= 1 + eps) if (_s[0]>s) idx = 2, s = _s[0]; }
			if (inEdge != 1 && _s[1]>0){ Real foo = p.p[1] + v[1] * _s[1]; if (foo >= -eps && foo <= 1 + eps) if (_s[1]>s) idx = 1, s = _s[1]; }
			if (inEdge != 0 && _s[2]>0){ Real foo = p.p[0] + v[0] * _s[2]; if (foo >= -eps && foo <= 1 + eps) if (_s[2]>s) idx = 0, s = _s[2]; }
		}
#if 0
		if (idx == -1)
		{
			fprintf(stderr, "[ERROR] Ray does not intersect triangle[%d]: (%f %f) (%g %g) [%g/%g]\n", count, p.p[0], p.p[1], v[0], v[1], Point2D< Real >::SquareNorm(v), eps*eps);
			Real s[] = { -p.p[1] / v[1], -p.p[0] / v[0], (Real(1.) - p.p[0] - p.p[1]) / (v[1] + v[0]) };
			if (inEdge != 2) { Real foo = p.p[0] + v[0] * s[0]; printf("\t0] %g -> %f\n", s[0], foo); }
			if (inEdge != 1) { Real foo = p.p[1] + v[1] * s[1]; printf("\t1] %g -> %f\n", s[1], foo); }
			if (inEdge != 0) { Real foo = p.p[0] + v[0] * s[2]; printf("\t2] %g -> %f\n", s[2], foo); }
			exit(0);
		}
#else
		if (idx == -1) return xForm;
#endif
		Real squareStepSize = Point2D< Real >::Dot(v, g[p.tIdx] * v) * s * s;
		bool updateVector = false;
		if (minStepSize>0 && squareStepSize>stepSizeLeft*stepSizeLeft)
		{
			s = stepSizeLeft / (Real)sqrt(Point2D< Real >::Dot(v, g[p.tIdx] * v));
			updateVector = true;
		}

		// If we can finish the flow
		if (flowTime<s)
		{
			p.p += v * flowTime;
			if (path) path->push_back(p);
			return xForm;
		}
		// If we do not cross a boundary, change direction
		else if (updateVector)
		{
			p.p += v * s, flowTime -= s;

			// If the the vectors are oppositely oriented, terminate the flow
			if (Point2D< Real >::Dot(v, g[p.tIdx] * GetWhitneyVector(p))*direction < 0) return xForm;

			v = GetWhitneyVector(p) * direction;
			stepSizeLeft = minStepSize;
			if (path) path->push_back(p);
			inEdge = -1;
		}
		// If we cross the boundary, transport the direction of the flow
		else // The end-point is outside the triangle
		{
			// Advance along the flow until you hit the edge
			p.p += v*s, flowTime -= s;

			const EdgeXForm< Real >& edge = edgeXForms[p.tIdx * 3 + idx];
			// Switch into the next triangle
			p.tIdx = edge.oppositeEdge / 3;
			p.p = edge.xForm(p.p);
			v = edge.xForm.linear * v;

			// Mark the edge we came in on
			inEdge = edge.oppositeEdge % 3;

			// Accumulate the transformations
			xForm = edge.xForm * xForm;

			stepSizeLeft -= (Real)sqrt(squareStepSize);
			if (path) path->push_back(p);
		}
		count++;
	}
	fprintf(stderr, "[WARNING] Failed to converge flow after %d iterations\n", MAX_ITERS);
	return xForm;
}

template< class Real >
FEM::CoordinateXForm< Real > FEM::RiemannianMesh< Real >::gradientFlow(ConstPointer(EdgeXForm< Real >) edgeXForms, ConstPointer(Real) f, SamplePoint< Real >& p, Real minStepSize, const int targetVertexIndex, Real & totalTime, std::vector<int> & traversedTriangles, Real eps, std::vector< SamplePoint< Real > >* path) const
{
	CoordinateXForm< Real > xForm = CoordinateXForm< Real >::Identity();
	int MAX_ITERS = 1000000;
	int count = 0;
	int inEdge = -1;
	Real direction = (Real)-1.;
	Real stepSizeLeft = minStepSize;
	totalTime = 0;
	auto GetGradient = [&](int tIdx){
		return g[tIdx].inverse() * Point2D< Real >(f[triangles[tIdx][1]] - f[triangles[tIdx][0]], f[triangles[tIdx][2]] - f[triangles[tIdx][0]]);
	};
	traversedTriangles.push_back(p.tIdx);
	Point2D< Real > v = GetGradient(p.tIdx) * direction;
	//flowTime *= direction;
	if (path) path->push_back(p);
	while (count<MAX_ITERS)
	{
		if (!Point2D< Real >::SquareNorm(v)) return xForm;
		// Intersect the ray p + s * v with each of the three edges
		// Bottom edge:   p[1] + s * v[1] = 0                         => s = -p[1]/v[1]
		// Left edge:     p[0] + s * v[0] = 0                         => s = -p[0]/v[0]
		// Diagonal edge: p[1] + s * v[1] = 1 - ( p[0] + s * v[0] )   => s = ( 1 - p[0]  - p[1] ) / ( v[1] + v[0] )
		Real s = 0;
		int idx = -1;
		{
			Real _s[] = { -p.p[1] / v[1], -p.p[0] / v[0], (Real(1.) - p.p[0] - p.p[1]) / (v[1] + v[0]) };
			if (inEdge != 2 && _s[0]>0){ Real foo = p.p[0] + v[0] * _s[0]; if (foo >= -eps && foo <= 1 + eps) if (_s[0]>s) idx = 2, s = _s[0]; }
			if (inEdge != 1 && _s[1]>0){ Real foo = p.p[1] + v[1] * _s[1]; if (foo >= -eps && foo <= 1 + eps) if (_s[1]>s) idx = 1, s = _s[1]; }
			if (inEdge != 0 && _s[2]>0){ Real foo = p.p[0] + v[0] * _s[2]; if (foo >= -eps && foo <= 1 + eps) if (_s[2]>s) idx = 0, s = _s[2]; }
		}
#if 0
		if (idx == -1)
		{
			fprintf(stderr, "[ERROR] Ray does not intersect triangle[%d]: (%f %f) (%g %g) [%g/%g]\n", count, p.p[0], p.p[1], v[0], v[1], Point2D< Real >::SquareNorm(v), eps*eps);
			Real s[] = { -p.p[1] / v[1], -p.p[0] / v[0], (Real(1.) - p.p[0] - p.p[1]) / (v[1] + v[0]) };
			if (inEdge != 2) { Real foo = p.p[0] + v[0] * s[0]; printf("\t0] %g -> %f\n", s[0], foo); }
			if (inEdge != 1) { Real foo = p.p[1] + v[1] * s[1]; printf("\t1] %g -> %f\n", s[1], foo); }
			if (inEdge != 0) { Real foo = p.p[0] + v[0] * s[2]; printf("\t2] %g -> %f\n", s[2], foo); }
			exit(0);
		}
#else
		if (idx == -1) return xForm;
#endif
		Real squareStepSize = Point2D< Real >::Dot(v, g[p.tIdx] * v) * s * s;
		bool updateVector = false;
		if (minStepSize>0 && squareStepSize>stepSizeLeft*stepSizeLeft)
		{
			s = stepSizeLeft / (Real)sqrt(Point2D< Real >::Dot(v, g[p.tIdx] * v));
			updateVector = true;
		}

		//// If we can finish the flow
		//if (flowTime<s)
		//{
		//	p.p += v * flowTime;
		//	if (path) path->push_back(p);
		//	return xForm;
		//}
		// If we do not cross a boundary, change direction
		if (updateVector)
		{
			p.p += v * s, totalTime += s;

			// If the the vectors are oppositely oriented, terminate the flow
			if (Point2D< Real >::Dot(v, g[p.tIdx] * GetGradient(p.tIdx))*direction < 0) return xForm;

			v = GetGradient(p.tIdx) * direction;
			stepSizeLeft = minStepSize;
			if (path) path->push_back(p);
			inEdge = -1;
		}
		// If we cross the boundary, transport the direction of the flow
		else // The end-point is outside the triangle
		{
			// Advance along the flow until you hit the edge
			p.p += v*s, totalTime += s;

			const EdgeXForm< Real >& edge = edgeXForms[p.tIdx * 3 + idx];
			// Switch into the next triangle
			p.tIdx = edge.oppositeEdge / 3;
			traversedTriangles.push_back(p.tIdx);
			if (triangles[p.tIdx][0] == targetVertexIndex || triangles[p.tIdx][1] == targetVertexIndex || triangles[p.tIdx][2] == targetVertexIndex) return xForm;
			p.p = edge.xForm(p.p);
			v = edge.xForm.linear * v;

			// Mark the edge we came in on
			inEdge = edge.oppositeEdge % 3;

			// Accumulate the transformations
			xForm = edge.xForm * xForm;

			stepSizeLeft -= (Real)sqrt(squareStepSize);
			if (path) path->push_back(p);
		}
		count++;
	}
	fprintf(stderr, "[WARNING] Failed to converge flow after %d iterations\n", MAX_ITERS);
	return xForm;
}

template< class Real >
Real FEM::RiemannianMesh< Real >::flow( ConstPointer( EdgeXForm< Real > ) edgeXForms , ConstPointer( Point2D< Real > ) vf , Real flowTime , SamplePoint< Real >& p , FEM::CoordinateXForm< Real >& xForm , Real minFlowTime , Real eps ) const
{
	Real distance = (Real)0;
#define DEBUG_FLOW 0
	xForm = CoordinateXForm< Real >::Identity();
#if DEBUG_FLOW
	const int MAX_ITERS = 10;
#else // !DEBUG_FLOW
	const int MAX_ITERS = 1000000;
#endif // DEBUG_FLOW
	int count = 0;
	int inEdge = -1;
	Point2D< Real > v = vf[ p.tIdx ];
	Real direction = (flowTime<0) ? (Real)-1. : (Real)1.;
	flowTime *= direction;
	while( count<MAX_ITERS )
	{
		v *= direction;
		if( !Point2D< Real >::SquareNorm( v ) ) return distance;
		// Intersect the ray p + s*v with each of the three edges
		// Bottom edge:   p[1] + s * v[1] = 0                         => s = -p[1]/v[1]
		// Left edge:     p[0] + s * v[0] = 0                         => s = -p[0]/v[0]
		// Diagonal edge: p[1] + s * v[1] = 1 - ( p[0] + s * v[0] )   => s = ( 1 - p[0]  - p[1] ) / ( v[1] + v[0] )
		Real maxD = 0;
		int idx = -1;
		{
			Real s[] = { -p.p[1] / v[1]  , -p.p[0] / v[0] , ( Real(1.) - p.p[0]  - p.p[1] ) / ( v[1] + v[0] ) };
			if( inEdge!=2 && s[0]>0 ){ Real foo = p.p[0] + v[0] * s[0] ; if( foo>=-eps && foo<=1+eps ) if( s[0]>maxD ) idx = 2 , maxD = s[0]; }
			if( inEdge!=1 && s[1]>0 ){ Real foo = p.p[1] + v[1] * s[1] ; if( foo>=-eps && foo<=1+eps ) if( s[1]>maxD ) idx = 1 , maxD = s[1]; }
			if( inEdge!=0 && s[2]>0 ){ Real foo = p.p[0] + v[0] * s[2] ; if( foo>=-eps && foo<=1+eps ) if( s[2]>maxD ) idx = 0 , maxD = s[2]; }
		}
#if 0
		if( idx==-1 )
		{
			fprintf( stderr , "[ERROR] Ray does not intersect triangle[%d]: (%f %f) (%g %g) [%g/%g]\n" , count , p.p[0] , p.p[1] , v[0] , v[1] , Point2D< Real >::SquareNorm(v) , eps*eps );
			Real s[] = { -p.p[1] / v[1]  , -p.p[0] / v[0] , ( Real(1.) - p.p[0]  - p.p[1] ) / ( v[1] + v[0] ) };
			if( inEdge!=2 ) { Real foo = p.p[0] + v[0] * s[0] ; printf( "\t0]%g -> %f\n" , s[0] , foo ); }
			if( inEdge!=1 ) { Real foo = p.p[1] + v[1] * s[1] ; printf( "\t1]%g -> %f\n" , s[1] , foo ); }
			if( inEdge!=0 ) { Real foo = p.p[0] + v[0] * s[2] ; printf( "\t2]%g -> %f\n" , s[2] , foo ); }
			exit( 0 );
		}
#else
		Real vLength = (Real)sqrt( Point2D< Real >::Dot( v , g[p.tIdx] * v ) );
		if( idx==-1 ) return distance;
#endif
#if DEBUG_FLOW
		printf( "maxD[%d] %g\n" , count , maxD );
#endif // DEBUG_FLOW
		if( maxD>flowTime ) // The end-point is within the triangle
		{
			distance += vLength * flowTime;
			p.p += v*flowTime;
			return distance;
		}
		else // The end-point is outside the triangle
		{
			const EdgeXForm< Real >& edge = edgeXForms[ p.tIdx*3 + idx ];

			// If the the vectors on the two sides of the edge are oppositely oriented, terminate the flow
			if( Point2D< Real >::Dot( edge.xForm.linear * v , g[edge.oppositeEdge/3] * vf[edge.oppositeEdge/3] )*direction < 0 ) return distance;
			distance += vLength * maxD;
			p.p += v*maxD , p.tIdx = edge.oppositeEdge/3 , flowTime -= maxD;
			p.p = edge.xForm( p.p );
			v = vf[ p.tIdx ];
			inEdge = edge.oppositeEdge%3;

			xForm = edge.xForm * xForm;
		}
		count++;
	}
	fprintf( stderr , "[WARNING] Failed to converge flow after %d iterations\n" , MAX_ITERS );
	return distance;
#undef DEBUG_FLOW
}
/////////////////////////
// FEM::RiemannianMesh //
/////////////////////////
template< class Real >
inline void FEM::RiemannianMesh< Real >::makeUnitArea( void )
{
	double scale = 0;
#pragma omp parallel for reduction( + : scale )
	for( int i=0 ; i<tCount ; i++ ) scale += sqrt( g[i].determinant() );
	scale = 2. / scale;
#pragma omp parallel for
	for( int i=0 ; i<tCount ; i++ ) g[i] *= (Real)scale;
}
template< class Real >
inline Real FEM::RiemannianMesh< Real >::area( void ) const
{
	Real area = 0;
#pragma omp parallel for reduction( + : area )
	for( int i=0 ; i<tCount ; i++ ) area += (Real)sqrt( g[i].determinant() );
	return area / (Real)2.;
}
template< class Real >
inline Real FEM::RiemannianMesh< Real >::area( int idx ) const { return (Real)sqrt( g[idx].determinant() ) / (Real)2.; }

template< class Real >
template< class Vertex >
void FEM::RiemannianMesh< Real >::setMetricFromEmbedding( ConstPointer( Vertex ) vertices )
{
	DeletePointer( g );
	g = NewPointer< SquareMatrix< Real , 2 > >( tCount );

#pragma omp parallel for
	for( int i=0 ; i<tCount ; i++ )
	{
		Point3D< Real > e[] = { Point3D< Real >( vertices[ triangles[i][1] ] ) - Point3D< Real >( vertices[ triangles[i][0] ] ) , Point3D< Real >( vertices[ triangles[i][2] ] ) - Point3D< Real >( vertices[ triangles[i][0] ] ) };
		for( int j=0 ; j<2 ; j++ ) for( int k=0 ; k<2 ; k++ ) g[i](j,k) = Point3D< Real >::Dot( e[j] , e[k] );
		g[i](0,1) = g[i](1,0) = (Real)( ( g[i](0,1) + g[i](1,0) )/2 );

		if( !g[i].determinant() )
		{
			fprintf( stderr , "[WARNING] Vanishing metric tensor determinant\n" );
			printf( "%g %g %g\t%g %g %g\n" , e[0][0] , e[0][1] , e[0][2] , e[1][0] , e[1][1] , e[1][2] );
		}
	}
}
template< class Real >
void FEM::RiemannianMesh< Real >::setMetricFromEdgeLengths( ConstPointer( Real ) edgeLengths )
{
	DeletePointer( g );
	g = NewPointer< SquareMatrix< Real , 2 > >( tCount );
#pragma omp parallel for
	for( int i=0 ; i<tCount ; i++ )
	{
		// The conditions the matrix needs to satisfy are:
		// -- < g[i] * (1,0) , (1,0) > = edgeLengths[i*3+2]^2
		//		=> g[i](0,0) = edgeLengths[i*3+2]^2
		// -- < g[i] * (0,1) , (0,1) > = edgeLengths[i*3+1]^2
		//		=> g[i](1,1) = edgeLengths[i*3+1]^2
		// -- < g[i] * (-1,1) , (-1,1) > = edgeLengths[i*3+0]^2
		//		=> g[i](0,0) + g[i](1,1) - g[i](0,1) - g[i](1,0) = edgeLengths[i*3+0]^2
		//		=> - g[i](0,1) - g[i](1,0) = edgeLengths[i*3+0]^2 - edgeLengths[i*3+2]^2 - edgeLengths[i*3+1]^2
		//		=>  g[i](0,1) = g[i](1,0) = ( edgeLengths[i*3+2]^2 + edgeLengths[i*3+1]^2 - edgeLengths[i*3+0]^2 ) / 2

		g[i](0,0) = edgeLengths[i*3+2] * edgeLengths[i*3+2];
		g[i](1,1) = edgeLengths[i*3+1] * edgeLengths[i*3+1];
		g[i](0,1) = g[i](1,0) = ( g[i](0,0) + g[i](1,1) - edgeLengths[i*3] * edgeLengths[i*3] ) / (Real)2.;
	}
}
template< class Real >
void FEM::RiemannianMesh< Real >::setMetricFromSquareEdgeLengths( ConstPointer( Real ) squareEdgeLengths )
{
	DeletePointer( g );
	g = NewPointer< SquareMatrix< Real , 2 > >( tCount );
#pragma omp parallel for
	for( int i=0 ; i<tCount ; i++ )
	{
		g[i](0,0) = squareEdgeLengths[i*3+2];
		g[i](1,1) = squareEdgeLengths[i*3+1];
		g[i](0,1) = g[i](1,0) = ( g[i](0,0) + g[i](1,1) - squareEdgeLengths[i*3] ) / (Real)2.;
	}
}


template< class Real >
void FEM::RiemannianMesh< Real >::setInverseMetric()
{
	DeletePointer(gInv);
	gInv = NewPointer< SquareMatrix< Real, 2 > >(tCount);
#pragma omp parallel for
	for (int i = 0; i < tCount; i++)gInv[i] = g[i].inverse();
}

template< class Real >
SparseMatrix< Real , int > FEM::RiemannianMesh< Real >::gradientMatrix( int gradType ) const
{
	int _vCount = vCount();
	SparseMatrix< Real , int > grad;
	Point2D< Real > _grads[] = { Point2D< Real >( (Real)-1 , (Real)-1 ) , Point2D< Real >( (Real)1 , (Real)0 ) , Point2D< Real >( (Real)0 , (Real)1 ) };
	grad.resize( tCount*2 );
#pragma omp parallel for
	for( int i=0 ; i<tCount ; i++ )
	{
		SquareMatrix< Real , 2 > gInverse = g[i].inverse();
		if( gradType==HAT_GRADIENT_AND_ROTATED_GRADIENT ) grad.SetRowSize( 2*i , 6 ) , grad.SetRowSize( 2*i+1 , 6 );
		else                                              grad.SetRowSize( 2*i , 3 ) , grad.SetRowSize( 2*i+1 , 3 );
		for( int j=0 ; j<3 ; j++ )
		{
			Point2D< Real > _grad = gInverse * _grads[j];
			Point2D< Real > _gradPerp = Rotate90( g[i] , _grad );
			int inOffset = 0 , outOffset = 0;
			if( gradType&HAT_GRADIENT )
			{
				grad[2*i+0][j+inOffset] = MatrixEntry< Real , int >( triangles[i][j] + outOffset , _grad[0] );
				grad[2*i+1][j+inOffset] = MatrixEntry< Real , int >( triangles[i][j] + outOffset , _grad[1] );
				inOffset = 3 , outOffset = _vCount;
			}
			if( gradType&HAT_ROTATED_GRADIENT )
			{
				grad[2*i+0][j+inOffset] = MatrixEntry< Real , int >( triangles[i][j] + outOffset , _gradPerp[0] );
				grad[2*i+1][j+inOffset] = MatrixEntry< Real , int >( triangles[i][j] + outOffset , _gradPerp[1] );
			}
		}
	}
	return grad;
}
template< class Real >
SparseMatrix< Real , int > FEM::RiemannianMesh< Real >::gradientDualMatrix( int gradType ) const
{
	int _vCount = vCount();
	SparseMatrix< Real , int > grad;
	Point2D< Real > _grads[] = { Point2D< Real >( (Real)-1 , (Real)-1 ) , Point2D< Real >( (Real)1 , (Real)0 ) , Point2D< Real >( (Real)0 , (Real)1 ) };
	grad.resize( tCount*2 );
#pragma omp parallel for
	for( int i=0 ; i<tCount ; i++ )
	{
		Real a = area(i);
		SquareMatrix< Real , 2 > gInverse = g[i].inverse();
		if( gradType==HAT_GRADIENT_AND_ROTATED_GRADIENT ) grad.SetRowSize( 2*i , 6 ) , grad.SetRowSize( 2*i+1 , 6 );
		else                                              grad.SetRowSize( 2*i , 3 ) , grad.SetRowSize( 2*i+1 , 3 );
		for( int j=0 ; j<3 ; j++ )
		{
			Point2D< Real > _grad = gInverse * _grads[j];
			Point2D< Real > _gradPerp = Rotate90( g[i] , _grad );
			_grad = g[i] * _grad * a;
			_gradPerp = g[i] * _gradPerp * a;
			int inOffset = 0 , outOffset = 0;
			if( gradType&HAT_GRADIENT )
			{
				grad[2*i+0][j+inOffset] = MatrixEntry< Real , int >( triangles[i][j] + outOffset , _grad[0] );
				grad[2*i+1][j+inOffset] = MatrixEntry< Real , int >( triangles[i][j] + outOffset , _grad[1] );
				inOffset = 3 , outOffset = _vCount;
			}
			if( gradType&HAT_ROTATED_GRADIENT )
			{
				grad[2*i+0][j+inOffset] = MatrixEntry< Real , int >( triangles[i][j] + outOffset , _gradPerp[0] );
				grad[2*i+1][j+inOffset] = MatrixEntry< Real , int >( triangles[i][j] + outOffset , _gradPerp[1] );
			}
		}
	}
	return grad.transpose( gradType==HAT_GRADIENT_AND_ROTATED_GRADIENT ? 2*_vCount : _vCount );
}
template< class Real >
inline Pointer( Point2D< Real > ) FEM::RiemannianMesh< Real >::getGradient( ConstPointer( Real ) vertexValues , int gradType ) const
{
	Pointer( Point2D< Real > ) gradients = NewPointer< Point2D< Real > >( tCount );
	setGradient( vertexValues , gradType , gradients );
	return gradients;
}
template< class Real >
inline void FEM::RiemannianMesh< Real >::setGradient( ConstPointer( Real ) vertexValues , int gradType , Pointer( Point2D< Real > ) gradients , int systemFlag ) const
{
	int _vCount = vCount();
#pragma omp parallel for
	for( int i=0 ; i<tCount ; i++ )
	{
		Real values[3];
		int offset =  0;
		if( !(systemFlag & SYSTEM_ADD )  ) gradients[i] = Point2D< Real >();
		if( gradType & HAT_GRADIENT )
		{
			for( int j=0 ; j<3 ; j++ ) values[j] = ( systemFlag & SYSTEM_NEGATE ) ? -vertexValues[ triangles[i][j] + offset ] : vertexValues[ triangles[i][j] + offset ];
			gradients[i] += RightTriangle< Real >::Gradient( g[i] , values );
			offset = _vCount;
		}
		if( gradType & HAT_ROTATED_GRADIENT )
		{
			for( int j=0 ; j<3 ; j++ ) values[j] = ( systemFlag & SYSTEM_NEGATE ) ? -vertexValues[ triangles[i][j] + offset ] : vertexValues[ triangles[i][j] + offset ];
			gradients[i] += FEM::Rotate90( g[i] , RightTriangle< Real >::Gradient( g[i] , values ) );
		}
	}
}
template< class Real >
template< class Data >
inline Pointer( Data ) FEM::RiemannianMesh< Real >::getProlongation( ConstPointer( Data ) faceData ) const
{
	Pointer( Data ) vertexData = NewPointer< Data >( vCount() );
	setProlongation( faceData , vertexData );
	return vertexData;
}
template< class Real >
template< class Data >
inline void FEM::RiemannianMesh< Real >::setProlongation( ConstPointer( Data ) faceData , Pointer( Data ) vertexData , int systemFlag ) const
{
	int vCount = this->vCount();
	Pointer( double ) areas = NewPointer< double >( vCount );
	if( !( systemFlag & SYSTEM_ADD ) )
#pragma omp parallel for
		for( int i=0 ; i<vCount ; i++ ) areas[i] = 0 , vertexData[i] *= (Real)0;
	else
#pragma omp parallel for
		for( int i=0 ; i<vCount ; i++ ) areas[i] = 0;

	for( int i=0 ; i<tCount ; i++ )
	{
		double a = area(i) , _a = a;
		if( systemFlag & SYSTEM_NEGATE ) _a = -_a;
		for( int j=0 ; j<3 ; j++ )
		{
			areas[ triangles[i][j] ] += a;
			vertexData[ triangles[i][j] ] += faceData[i] * (Real)_a;
		}
	}
#pragma omp parallel for
	for( int i=0 ; i<vCount ; i++ ) vertexData[i] /= (Real)areas[i];
	DeletePointer( areas );
}

template< class Real >
SparseMatrix< Real , int >FEM::RiemannianMesh< Real >::_scalarMatrix( bool mass , bool lump ) const
{
	SparseMatrix< Real , int > M;
	M.resize( vCount() );
	std::vector< std::atomic< int > > rowSizes( M.rows );
	Pointer( SquareMatrix< Real , 3 > ) m = AllocPointer< SquareMatrix< Real , 3 > >( tCount );

#pragma omp parallel for
	for( int i=0 ; i<M.rows ; i++ ) rowSizes[i] = 1;

#pragma omp parallel for
	for( int t=0 ; t<tCount ; t++ )
	{
		m[t] = mass ? RightTriangle< Real >::GetScalarMassMatrix( g[t] , lump ) : RightTriangle< Real >::GetScalarStiffnessMatrix( g[t] );
		for( int j=0 ; j<3 ; j++ ) rowSizes[ triangles[t][j] ] +=2;
	}

#pragma omp parallel for
	for( int i=0 ; i<M.rows ; i++ )	M.SetRowSize( i , rowSizes[i] ) , M[i][0] = MatrixEntry< Real , int >( i , (Real)0 ) , rowSizes[i] = 1;

#pragma omp parallel for
	for( int t=0 ; t<tCount ; t++ ) for( int j=0 ; j<3 ; j++ )
	{
#pragma omp atomic
		M[ triangles[t][j] ][0].Value += m[t](j,j);
		M[ triangles[t][j] ][ rowSizes[ triangles[t][j] ]++ ] = MatrixEntry< Real , int >( triangles[t][(j+1)%3] , m[t]( j , (j+1)%3 ) );
		M[ triangles[t][j] ][ rowSizes[ triangles[t][j] ]++ ] = MatrixEntry< Real , int >( triangles[t][(j+2)%3] , m[t]( j , (j+2)%3 ) );
	}
	FreePointer( m );
#pragma omp parallel for
	for( int i=0 ; i<M.rows ; i++ )
	{
		std::sort( M[i]+1 , M[i] + M.rowSizes[i] , []( MatrixEntry< Real , int > e1 , MatrixEntry< Real , int > e2 ){ return e1.N<e2.N; } );
		int idx=0;
		for( int j=1 ; j<M.rowSizes[i] ; j++ )
			if( M[i][j].N==M[i][idx].N ) M[i][idx].Value += M[i][j].Value;
			else idx++ , M[i][idx] = M[i][j];
		M.ResetRowSize( i , idx+1 );
	}
	return M;
}
template< class Real > SparseMatrix< Real , int > FEM::RiemannianMesh< Real >::scalarMassMatrix( bool lump ) const { return _scalarMatrix( true , lump ); }
template< class Real > SparseMatrix< Real , int > FEM::RiemannianMesh< Real >::scalarStiffnessMatrix( void ) const { return _scalarMatrix( false , false ); }
template< class Real >
SparseMatrix< Real , int > FEM::RiemannianMesh< Real >::gradientMassMatrix( int gradType ) const
{
	SparseMatrix< Real , int > m = gradientMatrix( gradType );
	return m.transpose() * vectorFieldMassMatrix() * m;
}
template< class Real >
SparseMatrix< Real , int > FEM::RiemannianMesh< Real >::gradientStiffnessMatrix( int gradType ) const
{
	SparseMatrix< Real , int > stiffness , sMass , sMassInverse , sStiffness , biStiffness;
	sMass = scalarMassMatrix( false );
	sStiffness = scalarStiffnessMatrix();
	sMassInverse = SparseMatrix< Real , int >::Identity( sMass.rows );
	for( int i=0 ; i<sMass.rows ; i++ )
	{
		Real sum = (Real)0;
		for( int j=0 ; j<sMass.rowSizes[i] ; j++ ) sum += sMass[i][j].Value;
		sMassInverse[i][0].Value = (Real)1./sum;
	}
	biStiffness = sStiffness * sMassInverse * sStiffness;
	size_t dim = biStiffness.rows;
	if( gradType==HAT_GRADIENT_AND_ROTATED_GRADIENT ) stiffness.resize( dim*2 );
	else                                              stiffness.resize( dim );
#pragma omp parallel for
	for( int i=0 ; i<dim ; i++ )
	{
		stiffness.SetRowSize( i , biStiffness.rowSizes[i] );
		if( gradType==HAT_GRADIENT_AND_ROTATED_GRADIENT ) stiffness.SetRowSize( i+dim , biStiffness.rowSizes[i] );
		for( int j=0 ; j<biStiffness.rowSizes[i] ; j++ )
		{
			stiffness[i][j] = biStiffness[i][j];
			if( gradType==HAT_GRADIENT_AND_ROTATED_GRADIENT ) stiffness[i+dim][j] = MatrixEntry< Real , int >( biStiffness[i][j].N+(int)dim , biStiffness[i][j].Value );
		}
	}
	return stiffness;
}

template< class Real >
SparseMatrix< Real , int > FEM::RiemannianMesh< Real >::vectorFieldRotate90Matrix( void ) const
{
	SquareMatrix< Real , 2 > J;
	J(0,0) = J(1,1) = 0;
	J(0,1) = (Real)-1 , J(1,0) = (Real)1;

	SparseMatrix< Real , int > rotate90;
	rotate90.resize( 2*tCount );
#pragma omp parallel for
	for( int i=0 ; i<tCount ; i++ )
	{
		SquareMatrix< Real , 2 > gRoot = TensorRoot( g[i] );
		SquareMatrix< Real , 2 > _J = gRoot.inverse() * J * gRoot;
		for( int j=0 ; j<2 ; j++ )
		{
			rotate90.SetRowSize( 2*i+j , 2 );
			for( int k=0 ; k<2 ; k++ ) rotate90[2*i+j][k] = MatrixEntry< Real , int >( 2*i+k , _J(k,j) );
		}
	}
	return rotate90;
}
template< class Real >
SparseMatrix< Real , int > FEM::RiemannianMesh< Real >::vectorFieldMassMatrix( void ) const
{
	SparseMatrix< Real , int > mass;
	mass.resize( 2*tCount );
#pragma omp parallel for
	for( int i=0 ; i<tCount ; i++ )
	{
		Real a = area(i);
		mass.SetRowSize( 2*i , 2 ) , mass.SetRowSize( 2*i+1 , 2 );
		mass[2*i+0][0] = MatrixEntry< Real , int >( 2*i+0 , g[i](0,0) * a );
		mass[2*i+0][1] = MatrixEntry< Real , int >( 2*i+1 , g[i](0,1) * a );
		mass[2*i+1][0] = MatrixEntry< Real , int >( 2*i+0 , g[i](1,0) * a );
		mass[2*i+1][1] = MatrixEntry< Real , int >( 2*i+1 , g[i](1,1) * a );
	}
	return mass;
}
template< class Real >
SquareMatrix< Real , 2 > FEM::RiemannianMesh< Real >::vectorFieldDotMass( int t , Point2D< Real > v ) const
{
	SquareMatrix< Real , 2 > m;
	Real a = area( t );
	for( int j=0 ; j<2 ; j++ ) for( int k=0 ; k<2 ; k++ ) m( j , k ) = v[j] * v[k] * a;
	m( 1 , 0 ) = m( 0 , 1 ) = (Real)( ( m(1,0)+m(0,1) )/2 );
	return g[t] * m * g[t];
}
template< class Real >
SparseMatrix< Real , int > FEM::RiemannianMesh< Real >::vectorFieldDotMassMatrix( ConstPointer( Point2D< Real > ) vectorField ) const
{
	SparseMatrix< Real , int > mass;
	mass.resize( 2*tCount );
#pragma omp parallel for
	for( int i=0 ; i<tCount ; i++ )
	{
		mass.SetRowSize( 2*i , 2 ) , mass.SetRowSize( 2*i+1 , 2 );
		Real a = area(i);
		SquareMatrix< Real , 2 > m;
		for( int j=0 ; j<2 ; j++ ) for( int k=0 ; k<2 ; k++ ) m( j , k ) = vectorField[i][j]*vectorField[i][k]*a;
		m = g[i] * m * g[i];
		for( int j=0 ; j<2 ; j++ ) for( int k=0 ; k<2 ; k++ ) mass[2*i+j][k] = MatrixEntry< Real , int >( 2*i+k , m(j,k) );
	}
	return mass;
}

template< class Real >
SparseMatrix< Real , int > FEM::RiemannianMesh< Real >::vectorFieldStiffnessMatrix( int dualType , int quadratureType ) const
{
	std::vector< EdgeXForm< Real > > edges( tCount*3 );
	setEdgeXForms( GetPointer( edges ) );
	return vectorFieldStiffnessMatrix( edges , dualType , quadratureType );
}
template< class Real >
SparseMatrix< Real , int > FEM::RiemannianMesh< Real >::vectorFieldDivergenceMatrix( void ) const
{
	std::vector< FEM::EdgeXForm< Real > > edges( tCount*3 );
	setEdgeXForms( GetPointer( edges ) );
	return vectorFieldDivergenceMatrix( edges );
}
template< class Real >
SparseMatrix< Real , int > FEM::RiemannianMesh< Real >::vectorFieldCovariantDerivativeTraceMatrix( int dualType ) const
{
	std::vector< FEM::EdgeXForm< Real > > edges( tCount*3 );
	setEdgeXForms( GetPointer( edges ) );
	return vectorFieldCovariantDerivativeTraceMatrix( edges , dualType );
}
template< class Real >
SparseMatrix< Real , int > FEM::RiemannianMesh< Real >::vectorFieldCovariantDerivativeTraceMatrix2( int dualType ) const
{
	std::vector< FEM::EdgeXForm< Real > > edges( tCount*3 );
	setEdgeXForms( GetPointer( edges ) );
	return vectorFieldCovariantDerivativeTraceMatrix2( edges , dualType );
}

template< class Real >
SparseMatrix< Real , int > FEM::RiemannianMesh< Real >::vectorFieldStiffnessMatrix( ConstPointer( EdgeXForm< Real > ) edges , ConstPointer( Point2D< Real > ) centers ) const
{
	SparseMatrix< Real , int > stiffness;
	stiffness.resize( 2*tCount );
	Pointer( Real ) edgeWeights = AllocPointer< Real >( tCount*3 );
#pragma omp parallel for
	for( int i=0 ; i<tCount ; i++ )
	{
		Point2D< Real > dirs[3];
		for( int j=0 ; j<3 ; j++ )
		{
			int e = i*3+j , oe = edges[e].oppositeEdge;
			if( oe!=-1 ) dirs[j] = edges[oe].xForm( centers[oe/3] ) - centers[i];
		}
#if 1
		Real a = ( area(i) / 3 * 2 );
		for( int j=0 ; j<3 ; j++ ) edgeWeights[i*3+j] = a / Point2D< Real >::Dot( dirs[j] , g[i] * dirs[j] );
for( int j=0 ; j<3 ; j++ ) if( Point2D< Real >::Dot( dirs[j] , g[i] * dirs[j] )==0 ) printf( "uh oh\n" );
#else
		Real a = area(i);
		Point3D< Real > weights = TraceWeights( g[i] , dirs );
if( weights[0]<0 || weights[1]<0 || weights[2]<0 ) printf( "Weights: %g %g %g\n" , weights[0] , weights[1] , weights[2] );
		for( int j=0 ; j<3 ; j++ ) edgeWeights[3*i+j] = weights[j] * a;
#endif
	}

	for( int i=0 ; i<tCount ; i++ )
	{
		int count = 1;
		for( int j=0 ; j<3 ; j++ ) if( edges[i*3+j].oppositeEdge!=-1 ) count++;
		stiffness.SetRowSize( 2*i , 2*count ) , stiffness.SetRowSize( 2*i+1 , 2*count );
		stiffness[2*i+0][0] = MatrixEntry< Real , int >( 2*i+0 , (Real)0 ) , stiffness[2*i+1][0] = MatrixEntry< Real , int >( 2*i+0 , (Real)0 );
		stiffness[2*i+0][1] = MatrixEntry< Real , int >( 2*i+1 , (Real)0 ) , stiffness[2*i+1][1] = MatrixEntry< Real , int >( 2*i+1 , (Real)0 );

		count = 1;
		for( int j=0 ; j<3 ; j++ ) if( edges[i*3+j].oppositeEdge!=-1 )
		{
			int edge = i*3+j , oppositeEdge = edges[edge].oppositeEdge , ii = oppositeEdge / 3 , jj = oppositeEdge % 3;

			Real s = edgeWeights[edge] + edgeWeights[oppositeEdge];
			stiffness[2*i+0][0].Value += s*g[i](0,0) , stiffness[2*i+1][0].Value += s*g[i](0,1);
			stiffness[2*i+0][1].Value += s*g[i](1,0) , stiffness[2*i+1][1].Value += s*g[i](1,1);

			SquareMatrix< Real , 2 > xPort = g[i] * edges[ oppositeEdge ].xForm.linear;
			stiffness[2*i+0][2*count+0] = MatrixEntry< Real , int >( 2*ii+0 , -xPort(0,0)*s ) , stiffness[2*i+1][2*count+0] = MatrixEntry< Real , int >( 2*ii+0 , -xPort(0,1)*s );
			stiffness[2*i+0][2*count+1] = MatrixEntry< Real , int >( 2*ii+1 , -xPort(1,0)*s ) , stiffness[2*i+1][2*count+1] = MatrixEntry< Real , int >( 2*ii+1 , -xPort(1,1)*s );
			count++;
		}
	}
	FreePointer( edgeWeights );
	return stiffness;
}
template< class Real >
SparseMatrix< Real , int > FEM::RiemannianMesh< Real >::vectorFieldStiffnessMatrix( ConstPointer( EdgeXForm< Real > ) edges , int dualType , int quadratureType ) const
{
	SparseMatrix< Real , int > stiffness;
	stiffness.resize( 2*tCount );
	Pointer( Real ) edgeWeights = AllocPointer< Real >( tCount*3 );
#pragma omp parallel for
	for( int i=0 ; i<tCount ; i++ )
	{
		Real a = area(i);
		Point2D< Real > dirs[3];
		setTriangleDerivativeDirections( i , edges , dualType , dirs );
		Point3D< Real > weights;
		CircularQuadratureWeights( g[i] , dirs , 3 , &weights[0] , quadratureType );
		weights /= (Real)( M_PI );
		for( int j=0 ; j<3 ; j++ ) edgeWeights[i*3+j] = a / Point2D< Real >::Dot( dirs[j] , g[i] * dirs[j] ) * weights[j];
	}

#pragma omp parallel for
	for( int i=0 ; i<tCount ; i++ )
	{
		int count = 1;
		for( int j=0 ; j<3 ; j++ ) if( edges[i*3+j].oppositeEdge!=-1 ) count++;
		stiffness.SetRowSize( 2*i , 2*count ) , stiffness.SetRowSize( 2*i+1 , 2*count );

		count = 1;
		for( int k=0 ; k<2 ; k++ ) for( int l=0 ; l<2 ; l++ ) stiffness[2*i+k][l] = MatrixEntry< Real , int >( 2*i+l , (Real)0 );
		for( int j=0 ; j<3 ; j++ ) if( edges[i*3+j].oppositeEdge!=-1 )
		{
			int ii = edges[i*3+j].oppositeEdge / 3;
			for( int k=0 ; k<2 ; k++ ) for( int l=0 ; l<2 ; l++ ) stiffness[2*i+k][2*count+l] = MatrixEntry< Real , int >( 2*ii+l , (Real)0 );
			count++;
		}

		count = 1;
		for( int j=0 ; j<3 ; j++ ) if( edges[i*3+j].oppositeEdge!=-1 )
		{
			int e = i*3+j , oe = edges[e].oppositeEdge;

			Real s = edgeWeights[e] + edgeWeights[oe];

			SquareMatrix< Real , 2 > xPort = g[i] * edges[oe].xForm.linear;
			for( int k=0 ; k<2 ; k++ ) for( int l=0 ; l<2 ; l++ ) stiffness[2*i+l][k].Value += s*g[i](k,l) , stiffness[2*i+l][2*count+k].Value -= xPort(k,l)*s;
			count++;
		}
	}
	FreePointer( edgeWeights );
	return stiffness;
}


template< class Real >
SparseMatrix< Real, int > FEM::RiemannianMesh< Real >::vectorFieldStiffnessMatrix(ConstPointer(EdgeXForm< Real >) edges, const std::vector<int> & triangleIndices, int dualType, int quadratureType) const
{

	std::unordered_map<int, int> triangleMap;
	for (int i = 0; i < triangleIndices.size(); i++) triangleMap[triangleIndices[i]] = i;

	SparseMatrix< Real, int > stiffness;
	stiffness.resize(2 * triangleIndices.size());
	Pointer(Real) edgeWeights = AllocPointer< Real >(triangleIndices.size() * 3);
#pragma omp parallel for
	for (int it = 0; it<triangleIndices.size(); it++)
	{
		int triangleIndex = triangleIndices[it];
		Real a = area(triangleIndex);
		Point2D< Real > dirs[3];
		setTriangleDerivativeDirections(triangleIndex, edges, dualType, dirs);
		Point3D< Real > weights;
		CircularQuadratureWeights(g[triangleIndex], dirs, 3, &weights[0], quadratureType);
		weights /= (Real)(M_PI);
		for (int j = 0; j<3; j++) edgeWeights[it * 3 + j] = a / Point2D< Real >::Dot(dirs[j], g[triangleIndex] * dirs[j]) * weights[j];
	}

#pragma omp parallel for
	for (int it = 0; it<triangleIndices.size(); it++)
	{
		int triangleIndex = triangleIndices[it];
		int count = 1;
		for (int j = 0; j<3; j++) if (edges[triangleIndex * 3 + j].oppositeEdge != -1 && triangleMap.find(edges[triangleIndex * 3 + j].oppositeEdge / 3) != triangleMap.end()) count++;
		stiffness.SetRowSize(2 * it, 2 * count), stiffness.SetRowSize(2 * it + 1, 2 * count);

		count = 1;
		for (int k = 0; k<2; k++) for (int l = 0; l<2; l++) stiffness[2 * it + k][l] = MatrixEntry< Real, int >(2 * it + l, (Real)0);
		for (int j = 0; j<3; j++) if (edges[triangleIndex * 3 + j].oppositeEdge != -1 && triangleMap.find(edges[triangleIndex * 3 + j].oppositeEdge /3) != triangleMap.end())
		{
			int ii = edges[triangleIndex * 3 + j].oppositeEdge / 3;
			for (int k = 0; k<2; k++) for (int l = 0; l<2; l++) stiffness[2 * it + k][2 * count + l] = MatrixEntry< Real, int >(2 * triangleMap[ii] + l, (Real)0);
			count++;
		}

		count = 1;
		for (int j = 0; j<3; j++) if (edges[triangleIndex * 3 + j].oppositeEdge != -1 && triangleMap.find(edges[triangleIndex * 3 + j].oppositeEdge /3) != triangleMap.end())
		{
			//int e = triangleIndex * 3 + j, oe = edges[triangleIndex * 3 + j].oppositeEdge;
			int oe = edges[triangleIndex * 3 + j].oppositeEdge;
			int ii = edges[triangleIndex * 3 + j].oppositeEdge / 3;
			int jj = edges[triangleIndex * 3 + j].oppositeEdge % 3;
			Real s = edgeWeights[it * 3 + j] + edgeWeights[triangleMap[ii] * 3 + jj];

			SquareMatrix< Real, 2 > xPort = g[triangleIndex] * edges[oe].xForm.linear;
			for (int k = 0; k<2; k++) for (int l = 0; l<2; l++) stiffness[2 * it + l][k].Value += s*g[triangleIndex](k, l), stiffness[2 * it + l][2 * count + k].Value -= xPort(k, l)*s;
			count++;
		}
	}
	FreePointer(edgeWeights);
	return stiffness;
}


template< class Real >
SparseMatrix< Real , int > FEM::RiemannianMesh< Real >::vectorFieldStiffnessMatrix_( ConstPointer( EdgeXForm< Real > ) edges , int dualType , int quadratureType , bool linearFit ) const
{
	struct Entry
	{
		int i , j;
		SquareMatrix< Real , 2 > v;
		Entry( int i=-1 , int j=-1 , SquareMatrix< Real , 2 > v = SquareMatrix< Real , 2 >() ){ this->i = i , this->j = j , this->v = v; }
	};
	std::vector< Entry > entries( tCount*16 );

	auto setEntries = []( const RiemannianMesh< Real >* mesh , ConstPointer( EdgeXForm< Real > ) edges , int t , int dualType , Pointer( Entry ) entries , int quadratureType , bool linearFit )
	{
		entries += 16*t;

		Point2D< Real > dirs[3];
		mesh->setTriangleDerivativeDirections( t , edges , dualType , dirs );

		int tIndices[] = { t , -1 , -1 , -1 };
		Matrix< Real , 8 , 6 > finiteDifference;
		SquareMatrix< Real , 2 > identity = SquareMatrix< Real , 2 >::Identity();
		for( int v=0 ; v<3 ; v++ )
		{
			int e = t*3+v , oe = edges[e].oppositeEdge , ot = oe / 3;
			if( oe!=-1 )
			{
				tIndices[v+1] = ot;
				for( int i=0 ; i<2 ; i++ ) for( int j=0 ; j<2 ; j++ ) finiteDifference( i , 2*v+j ) = identity(i,j) , finiteDifference( 2*(v+1)+i , 2*v+j ) = -edges[oe].xForm.linear(i,j);
			}
		}
		SquareMatrix< Real , 8 > form;
		if( linearFit )
		{
			SquareMatrix< Real , 6 > tForm = TraceForm( mesh->g[t] , dirs );
			SquareMatrix< Real , 6 > res = LinearFitResidual( dirs );
			SquareMatrix< Real , 6 > d = res.transpose() * MCTraceForm( mesh->g[t] , dirs , quadratureType ) * res;
			form = finiteDifference.transpose() * ( tForm + d ) * finiteDifference * mesh->area(t);
		}
		else form = finiteDifference.transpose() * MCTraceForm( mesh->g[t] , dirs , quadratureType ) * finiteDifference * mesh->area(t);

		for( int i=0 ; i<4 ; i++ ) for( int j=0 ; j<4 ; j++ )
		{
			entries[4*i+j].i = tIndices[i] , entries[4*i+j].j = tIndices[j];
			// [NOTE] The indexing is reversed because the subsequent setting of the matrix coefficients is done by having "i" index the row, not the column
			for( int ii=0 ; ii<2 ; ii++ ) for( int jj=0 ; jj<2 ; jj++ ) entries[4*i+j].v(ii,jj) = form(2*i+jj,2*j+ii);
		}
	};

#pragma omp parallel for
	for( int t=0 ; t<tCount ; t++ ) setEntries( this , edges , t , dualType , GetPointer( entries ) , quadratureType , linearFit );

	SparseMatrix< Real , int > stiffness;
	stiffness.resize( 2*tCount );
#pragma omp parallel for
	for( int i=0 ; i<entries.size() ; i++ ) if( entries[i].i!=-1 && entries[i].j!=-1 )
	{
#pragma omp atomic
		stiffness.rowSizes[ 2*entries[i].i+0 ] += 2;
#pragma omp atomic
		stiffness.rowSizes[ 2*entries[i].i+1 ] += 2;
	}
#pragma omp parallel for
	for( int i=0 ; i<stiffness.rows ; i++ )
	{
		int temp = stiffness.rowSizes[i];
		stiffness.rowSizes[i] = 0;
		stiffness.SetRowSize( i , temp );
		stiffness.rowSizes[i] = 0;
	}

	for( int i=0 ; i<entries.size() ; i++ ) if( entries[i].i!=-1 && entries[i].j!=-1 )
	{
		int ii = entries[i].i , jj = entries[i].j;
		const SquareMatrix< Real , 2 >& temp = entries[i].v;
		stiffness[ 2*ii+0 ][ stiffness.rowSizes[2*ii+0]++ ] = MatrixEntry< Real , int >( 2*jj+0 , temp(0,0) );
		stiffness[ 2*ii+0 ][ stiffness.rowSizes[2*ii+0]++ ] = MatrixEntry< Real , int >( 2*jj+1 , temp(1,0) );
		stiffness[ 2*ii+1 ][ stiffness.rowSizes[2*ii+1]++ ] = MatrixEntry< Real , int >( 2*jj+0 , temp(0,1) );
		stiffness[ 2*ii+1 ][ stiffness.rowSizes[2*ii+1]++ ] = MatrixEntry< Real , int >( 2*jj+1 , temp(1,1) );
	}

	return stiffness;
}
template< class Real >
SparseMatrix< Real , int > FEM::RiemannianMesh< Real >::vectorFieldDivergenceMatrix( ConstPointer( EdgeXForm< Real > ) edges ) const
{
	Point2D< Real > corners[] = { Point2D< Real >( (Real)0 , (Real)0 ) , Point2D< Real >( (Real)1 , (Real)0 ) , Point2D< Real >( (Real)0 , (Real)1 ) };
	SparseMatrix< Real , int > divergence;
	divergence.resize( tCount );

	for( int i=0 ; i<tCount ; i++ )
	{
		Real a = area(i);
		int count = 0;
		for( int j=0 ; j<3 ; j++ ) if( edges[i*3+j].oppositeEdge!=-1 ) count++;
		divergence.SetRowSize( i , 2*count );

		count = 0;
		for( int j=0 ; j<3 ; j++ ) if( edges[i*3+j].oppositeEdge!=-1 )
		{
			int edge = i*3+j , oppositeEdge = edges[edge].oppositeEdge , ii = oppositeEdge / 3 , jj = oppositeEdge % 3;
			Point2D< Real > e = Rotate90( g[i] , corners[ (j+2)%3 ] - corners[ (j+1)%3 ] );
			// The contribution across edge e is:
			//		e^t * g[i] * edges[ oppositeEdge ].xForm.linear
			e = ( ( SquareMatrix< Real , 2 > )edges[oppositeEdge].xForm.linear.transpose() ) * ( g[i] * e );
			e /= a * 2;
			divergence[i][2*count+0] = MatrixEntry< Real , int >( 2*ii + 0 , e[0] );
			divergence[i][2*count+1] = MatrixEntry< Real , int >( 2*ii + 1 , e[1] );
			count++;
		}
	}
	return divergence;
}
template< class Real >
SparseMatrix< Real , int > FEM::RiemannianMesh< Real >::vectorFieldCovariantDerivativeTraceMatrix( ConstPointer( EdgeXForm< Real > ) edges , int dualType ) const
{
	SparseMatrix< Real , int > covariantDerivativeTrace;
	covariantDerivativeTrace.resize( tCount );

	Pointer( Real ) triangleAreas = AllocPointer< Real >( tCount );
#pragma omp parallel for
	for( int i=0 ; i<tCount ; i++ ) triangleAreas[i] = area( i );

#pragma omp parallel for
	for( int i=0 ; i<tCount ; i++ )
	{
		int count = 1;
		for( int j=0 ; j<3 ; j++ ) if( edges[i*3+j].oppositeEdge!=-1 ) count++;
		covariantDerivativeTrace.SetRowSize( i , 2*count );
		for( int k=0 ; k<2 ; k++ ) covariantDerivativeTrace[i][k] = MatrixEntry< Real , int >( 2*i+k , (Real)0 );

		Point2D< Real > dirs[3];
		setTriangleDerivativeDirections( i , edges , dualType , dirs );
		Point3D< Real > traceWeights = TraceWeights( g[i] , dirs );

		count = 1;
		for( int j=0 ; j<3 ; j++ ) if( edges[i*3+j].oppositeEdge!=-1 )
		{
			int edge = i*3+j , oppositeEdge = edges[edge].oppositeEdge , ii = oppositeEdge / 3 , jj = oppositeEdge % 3;

			// Given triangles T and T', the covariant derivative across the shared edge will be:
			//		( edges[ oppositeEdge ].xForm.linear * V[T'] - V[T] ) / l
			// And the contribution to the trace will be:
			//		< ( edges[ oppositeEdge ].xForm.linear * V[T'] - V[T] ) / l , dirs[j] >_g * traceWeights[j]
			//		( < V[T'] , edges[ oppositeEdge ].xForm.linear^t * g * dirs[j] > - < V[T] , g * dirs[j] > ) * traceWeights[j] / l
			Point2D< Real > gDir = g[i] * dirs[j] * traceWeights[j];

			for( int k=0 ; k<2 ; k++ ) covariantDerivativeTrace[i][k].Value -= gDir[k];

			gDir = ( ( SquareMatrix< Real , 2 > )edges[ oppositeEdge ].xForm.linear.transpose() ) * gDir;
			for( int k=0 ; k<2 ; k++ ) covariantDerivativeTrace[i][2*count+k] = MatrixEntry< Real , int >( 2*ii+k , gDir[k] );

			count++;
		}
	}
	FreePointer( triangleAreas );
	return covariantDerivativeTrace;
}
template< class Real >
SparseMatrix< Real , int > FEM::RiemannianMesh< Real >::vectorFieldCovariantDerivativeTraceMatrix2( ConstPointer( EdgeXForm< Real > ) edges , int dualType ) const
{
	SparseMatrix< Real , int > covariantDerivativeTrace;
	covariantDerivativeTrace.resize( tCount );

#pragma omp parallel for
	for( int i=0 ; i<tCount ; i++ )
	{
		// Get the number of triangle neighbors
		int count = 1;
		for( int j=0 ; j<3 ; j++ ) if( edges[i*3+j].oppositeEdge!=-1 ) count++;
		covariantDerivativeTrace.SetRowSize( i , 2*count );

		// Initialize the matrix entries
		count = 1;
		for( int k=0 ; k<2 ; k++ ) covariantDerivativeTrace[i][k] = MatrixEntry< Real , int >( 2*i+k , (Real)0 );
		for( int j=0 ; j<3 ; j++ ) if( edges[i*3+j].oppositeEdge!=-1 )
		{
			int ii = edges[i*3+j].oppositeEdge / 3;
			for( int k=0 ; k<2 ; k++ ) covariantDerivativeTrace[i][2*count+k] = MatrixEntry< Real , int >( 2*ii+k , (Real)0 );
			count++;
		}

		Point2D< Real > dirs[3];
		setTriangleDerivativeDirections( i , edges , dualType , dirs );
		Matrix< Real , 6 , 4 > linearFit = LinearFit( dirs );

		count = 1;
		for( int j=0 ; j<3 ; j++ ) if( edges[i*3+j].oppositeEdge!=-1 )
		{
			// Given triangles T and T', the change in derivative across the shared edge will be:
			//		edges[ oppositeEdge ].xForm.linear * V[T'] - V[T]
			// And the contribution to the covariance matrix will be:
			//		linearFits[j] * ( edges[ oppositeEdge ].xForm.linear * V[T'] - V[T] )
			Matrix< Real , 2 , 4 > lFit , _lFit;
			for( int k=0 ; k<2 ; k++ ) for( int l=0 ; l<4 ; l++ ) lFit(k,l) = linearFit(j*2+k,l);
			_lFit = lFit * edges[ edges[i*3+j].oppositeEdge ].xForm.linear;

			for( int k=0 ; k<2 ; k++ ) covariantDerivativeTrace[i][k].Value -= lFit(k,0) + lFit(k,3) , covariantDerivativeTrace[i][2*count+k].Value += _lFit(k,0) + _lFit(k,3);

			count++;
		}
	}
	return covariantDerivativeTrace;
}


template< class Real >
void FEM::RiemannianMesh< Real >::setVectorFieldDot( ConstPointer( Point2D< Real > ) vf1 , ConstPointer( Point2D< Real > ) vf2 , Pointer( Real ) tValues , int systemFlag ) const
{
#pragma omp parallel for
	for( int i=0 ; i<tCount ; i++ )
	{
		Real dot = Point2D< Real >::Dot( g[i] * vf1[i] , vf2[i] );
		if( systemFlag & SYSTEM_NEGATE ) dot = -dot;
		if( systemFlag & SYSTEM_ADD ) tValues[i] += dot;
		else                          tValues[i]  = dot;
	}
}
template< class Real >
void FEM::RiemannianMesh< Real >::setVectorFieldDotDual( ConstPointer( Point2D< Real > ) vf1 , ConstPointer( Real ) tValues , Pointer( Point2D< Real > ) vf2 , int systemFlag ) const
{
	if( !( systemFlag & SYSTEM_ADD ) )
#pragma omp parallel for
		for( int i=0  ; i<tCount ; i++ ) vf2[i] *= (Real)0;
#pragma omp parallel for
	for( int i=0 ; i<tCount ; i++ )
	{
		Real a = area(i) * tValues[i];
		if( systemFlag & SYSTEM_NEGATE ) a = -a;
		Point2D< Real > dot = g[i] * vf1[i] * a;
#pragma omp atomic
		vf2[i][0] += dot[0];
#pragma omp atomic
		vf2[i][1] += dot[1];
	}
}
template< class Real >
inline Real FEM::RiemannianMesh< Real >::getIntegral( ConstPointer( Real ) coefficients ) const
{
	Real integral = (Real)0;
#pragma omp parallel for
	for( int i=0 ; i<tCount ; i++ )
	{
		SquareMatrix< Real , 3 > mass;
		FEM::RightTriangle< Real >::SetScalarMassMatrix( g[i] , mass , false );
		for( int j=0 ; j<3 ; j++ )
		{
			Real sum = (Real)0;
			for( int k=0 ; k<3 ; k++ ) sum += mass(j,k);
#pragma omp atomic
			integral += coefficients[ triangles[i][j] ] * sum;
		}
	}
	return integral;
}
template< class Real >
inline Real FEM::RiemannianMesh< Real >::getDotProduct( ConstPointer( Real ) coefficients1 , ConstPointer( Real ) coefficients2 , bool lump ) const
{
	Real dotProduct = (Real)0;
#pragma omp parallel for reduction( + : dotProduct )
	for( int i=0 ; i<tCount ; i++ )
	{
		SquareMatrix< Real , 3 > mass;
		FEM::RightTriangle< Real >::SetScalarMassMatrix( g[i] , mass , lump );
		for( int j=0 ; j<3 ; j++ ) for( int k=0 ; k<3 ; k++ ) dotProduct += mass(j,k) * coefficients1[ triangles[i][j] ] * coefficients2[ triangles[i][k] ];
	}
	return dotProduct;
}
template< class Real >
inline void FEM::RiemannianMesh< Real >::rotate90( Pointer( Point2D< Real > ) vf ) const
{
#pragma omp parallel for
	for( int i=0 ; i<tCount ; i++ ) vf[i] = FEM::Rotate90( g[i] , vf[i] );
}
template< class Real >
inline void FEM::RiemannianMesh< Real >::setTriangleDerivativeDirections( int t , ConstPointer( EdgeXForm< Real > ) edges , int dualType , Point2D< Real > dirs[3] ) const
{
	static const double SQRT_THREE_QUARTERS = sqrt( 3./4 );
	for( int j=0 ; j<3 ; j++ )
	{
		int o = edges[t*3+j].oppositeEdge , tt = o / 3;
		if( o!=-1 ) dirs[j] = edges[o].xForm( RightTriangle< Real >::Center( g[tt] , dualType ) ) - RightTriangle< Real >::Center( g[t] , dualType );
		else        dirs[j] = RightTriangle< Real >::EdgeReflect( g[t] , j , RightTriangle< Real >::Center( g[t] , dualType ) ) - RightTriangle< Real >::Center( g[t] , dualType );
		if( dualType==RightTriangle< Real >::DUAL_CIRCUMCENTER_PROJECTED_BARYCENTRIC )
		{
			Point2D< Real > dir = Rotate90( g[t] , RightTriangle< Real >::Edges[j] );
			dirs[j] = dir * Point2D< Real >::Dot( dirs[j] , g[t] * dir ) / Point2D< Real >::Dot( dir , g[t] * dir );
		}
		else if( dualType==RightTriangle< Real >::DUAL_ISOGON_PROJECTED_BARYCENTRIC )
		{
			Point2D< Real > dir = RightTriangle< Real >::EdgeMidpoints[j] - Rotate90( g[t] ,  RightTriangle< Real >::Edges[j] ) * (Real)SQRT_THREE_QUARTERS - RightTriangle< Real >::Center( g[t] , RightTriangle< Real >::DUAL_ISOGONIC );
			dirs[j] = dir * Point2D< Real >::Dot( dirs[j] , g[t] * dir ) / Point2D< Real >::Dot( dir , g[t] * dir );
		}
	}
}

#ifdef SUPPORT_LINEAR_PROGAM
#undef SUPPORT_LINEAR_PROGAM
#endif // SUPPORT_LINEAR_PROGAM