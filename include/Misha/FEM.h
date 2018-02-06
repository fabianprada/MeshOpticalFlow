#ifndef FEM_INCLUDED
#define FEM_INCLUDED

#include <string.h>
#include "SparseMatrix.h"
#include "Geometry.h"
#include "Array.h"
#include <unordered_set>
namespace FEM
{
	enum
	{
		SYSTEM_SET    = 0 ,
		SYSTEM_ADD    = 1 ,
		SYSTEM_NEGATE = 2
	};
	enum
	{
		QUADRATURE_ANGULAR=1 ,
		QUADRATURE_SQUARE_LENGTH=2 ,
	};

	template< class Real > Real Area( const SquareMatrix< Real , 2 >& tensor , const Point2D< Real > tri[3] );
	template< class Real > Point3D< Real > TraceWeights( const SquareMatrix< Real , 2 >& tensor , const Point2D< Real > directions[3] );
	template< class Real > void TraceWeights( const Point2D< Real >* directions , int dirCount , Real* weights );
	template< class Real > void CircularQuadratureWeights( const SquareMatrix< Real , 2 >& tensor , const Point2D< Real >* dirs , int dirCount , Real* weights , int quadratureType );

	// The matrix that takes the values at the three prescribed directions and returns the 2x2 linear transform that best matches
	template< class Real > Matrix< Real , 6 , 4 > LinearFit( const Point2D< Real > directions[3] );
	// The matrix that evaluates the best-fit matrix at the three prescribed directions
	template< class Real > SquareMatrix< Real , 6 > LinearFitEvaluation( const Point2D< Real > directions[3] );
	// The matrix that evaluates the difference between the best-fit predicted values and the actual values
	template< class Real > SquareMatrix< Real , 6 > LinearFitResidual( const Point2D< Real > directions[3] );

	template< class Real > SquareMatrix< Real , 6 > MCTraceForm( const SquareMatrix< Real , 2 >& tensor , const Point2D< Real > directions[3] , int quadratureType=0 );
	template< class Real > SquareMatrix< Real , 6 > TraceForm( const SquareMatrix< Real , 2 >& tensor , const Point2D< Real > directions[3] );

	template< class Real > Point2D< Real > Rotate90( const SquareMatrix< Real , 2 >& tensor , Point2D< Real > v );
	template< class Real > SquareMatrix< Real , 2 > MakeConformal( const SquareMatrix< Real , 2 >& sourceTensor , const SquareMatrix< Real , 2 >& targetTensor );
	template< class Real > SquareMatrix< Real , 2 > MakeAuthalic ( const SquareMatrix< Real , 2 >& sourceTensor , const SquareMatrix< Real , 2 >& targetGensor );
	template< class Real > SquareMatrix< Real , 2 > TensorRoot( const SquareMatrix< Real , 2 >& tensor );

	template< class Real >
#if 0
	struct RightTriangle : public SquareMatrix< Real , 2 >
#else
	struct RightTriangle
#endif
	{
		enum
		{
			DUAL_BARYCENTRIC ,
			DUAL_CIRCUMCENTRIC ,
			DUAL_CIRCUMCENTER_PROJECTED_BARYCENTRIC ,
			DUAL_INCENTRIC ,
			DUAL_ISOGONIC ,
			DUAL_ISOGON_PROJECTED_BARYCENTRIC ,
			DUAL_COUNT
		};
		static const char* DualNames[];// = { "barycentric" , "circumcentric" , "circumcenter projected barycentric" , "incentric" , "isogonic" , "isogon projected barycentric" };
		static const Point2D< Real > Corners[];
		static const Point2D< Real > EdgeMidpoints[];
		static const Point2D< Real > Edges[];

#if 0
		RightTriangle( const SquareMatrix< Real , 2 >& m = SquareMatrix< Real , 2 >::Identity() ){ (*this)=m; }
		SquareMatrix< Real , 3 > massMatrix( bool lump=false ) const;
		SquareMatrix< Real , 3 > stiffnessMatrix( void ) const;
		Real edgeDot( int e1 , int e2 ) const;
		Real squareEdgeLength( int e ) const;
		Real edgeLength( int e ) const;
		Real dot( Point2D< Real > v1 , Point2D< Real > v2 ) const;
		Real squareLength( Point2D< Real > v ) const;
		Real length( Point2D< Real > v ) const;
		Real angle( int v ) const;
		Point2D< Real > gradient( const Real values[3] ) const;
		Point2D< Real > center( int dualType ) const;
		Point3D< Real > centerAreas( int dualType ) const;
		Point3D< Real > subTriangleAreas( Point2D< Real > center ) const;
		Point2D< Real > edgeReflect( int e , Point2D< Real > p ) const;
#else
		static SquareMatrix< Real , 3 > GetScalarMassMatrix( const SquareMatrix< Real , 2 >& tensor , bool lump=false );
		static SquareMatrix< Real , 3 > GetScalarStiffnessMatrix( const SquareMatrix< Real , 2 >& tensor );
		static void SetScalarMassMatrix( const SquareMatrix< Real , 2 >& tensor , SquareMatrix< Real , 3 >& massMatrix , bool lump=false );
		static void SetScalarStiffnessMatrix( const SquareMatrix< Real , 2 >& tensor , SquareMatrix< Real , 3 >& stiffnessMatrix );
		static Real EdgeDot( const SquareMatrix< Real , 2 >& tensor , int e1 , int e2 );
		static Real SquareEdgeLength( const SquareMatrix< Real , 2 >& tensor , int e );
		static Real EdgeLength( const SquareMatrix< Real , 2 >& tensor , int e );
		static Real Dot( const SquareMatrix< Real , 2 >& tensor , Point2D< Real > v1 , Point2D< Real > v2 );
		static Real SquareLength( const SquareMatrix< Real , 2 >& tensor , Point2D< Real > v );
		static Real Length( const SquareMatrix< Real , 2 >& tensor , Point2D< Real > v );
		static Real Angle( const SquareMatrix< Real , 2 >& tensor , int v );
		static Point2D< Real > Gradient( const SquareMatrix< Real , 2 >& tensor , const Real values[3] );
		static Point2D< Real > Center( const SquareMatrix< Real , 2 >& tensor , int dualType );
		static Point3D< Real > CenterAreas( const SquareMatrix< Real , 2 >& tensor , int dualType );
		static Point3D< Real > SubTriangleAreas( const SquareMatrix< Real , 2 >& tensor , Point2D< Real > center );
		static Point2D< Real > EdgeReflect( const SquareMatrix< Real , 2 >& tensor , int e , Point2D< Real > p );
#endif
	};

	// Structures for calculating geodesics
	template< class Real >
	struct SamplePoint
	{
		int tIdx;
		Point2D< Real > p;
		SamplePoint( void ){ ; }
		SamplePoint( int tIdx , const Point2D< Real > p ){ this->tIdx = tIdx , this->p = p; }
	};
	template< class Real >
	struct HermiteSamplePoint : public SamplePoint< Real >
	{
		using SamplePoint<Real>::tIdx;
		using SamplePoint<Real>::p;
		Point2D< Real > v;
		HermiteSamplePoint( void ){ ; }
		HermiteSamplePoint( int tIdx , const Point2D< Real >& p , const Point2D< Real > v=Point2D< Real >() ){ this->tIdx = tIdx , this->p = p , this->v = v; }
		HermiteSamplePoint( const SamplePoint<Real>& p , const Point2D< Real > v=Point2D< Real >() ){ tIdx = p.tIdx , this->p = p.p , this->v = v; }
	};
	template< class Real >
	struct CoordinateXForm : public Group< CoordinateXForm< Real > >
	{
		SquareMatrix< Real , 2 > linear;
		Point2D< Real > constant;

		CoordinateXForm( void ){ constant = Point2D< Real >() , linear = SquareMatrix< Real , 2 >::Identity(); }
		// (A,s) * p = A*p + s
		Point2D< Real > operator() ( const Point2D< Real >& p ) const { return linear*p + constant; }
		Point2D< Real > operator * ( const Point2D< Real >& p ) const { return linear*p + constant; }

		// (A,s) * (B,t) * p = (A,s) * (B*p + t) = A*B*p + (A*t + s)
		void SetIdentity( void ){ constant = Point2D< Real >() , linear = SquareMatrix< Real , 2 >::Identity(); }
		void Multiply( const CoordinateXForm& xForm ){ constant += linear * xForm.constant , linear *= xForm.linear; }
		void Invert( void ){ linear = linear.inverse() , constant = - linear * constant; }
	};
	template< class Real >
	struct EdgeXForm
	{
		int oppositeEdge;
		CoordinateXForm< Real > xForm;
		EdgeXForm( void ) { oppositeEdge = -1; }
	};

	// This structure represents a Riemmanian mesh, with the triangles giving the connectivity and the square (symmetric) matrices giving the metric
	template< class Real >
	struct RiemannianMesh
	{
		Pointer( SquareMatrix< Real , 2 > ) g;
		Pointer(SquareMatrix< Real, 2 >) gInv;
		Pointer( TriangleIndex ) triangles;
		size_t tCount;

		RiemannianMesh( void ){ triangles = NullPointer< TriangleIndex >() , tCount = 0 , g = NullPointer< SquareMatrix< Real , 2 > >(); }
		inline int vCount( void ) const;

		template< class Vertex > void setMetricFromEmbedding( ConstPointer( Vertex ) vertices );
		void setMetricFromEdgeLengths( ConstPointer( Real ) edgeLengths );
		void setMetricFromSquareEdgeLengths( ConstPointer( Real ) squareEdgeLengths );
		void makeUnitArea( void );
		void setInverseMetric(void);
		Real area( void ) const;
		Real area( int idx ) const;

		// Perform 1-to-4 subdivision
		RiemannianMesh subdivide( void ) const;

		// Estimate the second fundamental form using the normal variation
		template< class Vertex > Pointer( SquareMatrix< Real , 2 > ) getSecondFundamentalForm( ConstPointer( Vertex ) vertices ,                                          int nSmooth=0 ) const;
		template< class Vertex > void                                setSecondFundamentalForm( ConstPointer( Vertex ) vertices , Pointer( SquareMatrix< Real , 2 > ) II , int nSmooth=0 ) const;
		template< class Vertex > Pointer( SquareMatrix< Real , 2 > ) getSecondFundamentalForm( ConstPointer( Vertex ) vertices , ConstPointer( Point3D< Real > ) vNormals                                          ) const;
		template< class Vertex > void                                setSecondFundamentalForm( ConstPointer( Vertex ) vertices , ConstPointer( Point3D< Real > ) vNormals , Pointer( SquareMatrix< Real , 2 > ) II ) const;


		///////////////////////////
		// Topological Operators //
		///////////////////////////

		// Average per-triangle values into the vertices
		template< class Data > Pointer( Data ) getProlongation( ConstPointer( Data ) faceData                                                          ) const;
		template< class Data > void            setProlongation( ConstPointer( Data ) faceData , Pointer( Data ) vertexData , int systemFlag=SYSTEM_SET ) const;


		/////////////////////////
		// Geometric Operators //
		/////////////////////////

		// Mass and stiffness matrices for scalar functions expressed in terms of the basis functions
		SparseMatrix< Real , int > scalarMassMatrix( bool lump ) const;
		SparseMatrix< Real , int > scalarStiffnessMatrix( void ) const;

		// Vector field basis defined in terms of the gradients of the hat basis
		// functions and the (in-plane) rotation of the gradients by 90 degress.
		// Note: This basis cannot represent the harmonic vector fields.
		enum
		{
			HAT_GRADIENT = 1 ,
			HAT_ROTATED_GRADIENT = 2 ,
			HAT_GRADIENT_AND_ROTATED_GRADIENT = HAT_GRADIENT | HAT_ROTATED_GRADIENT
		};
		// Operators for transforming a representation of a vector field in terms of the coefficients of the hat basis functions
		// into a representation in terms of per-triangle tangent vector
		SparseMatrix< Real , int > gradientMatrix( int gradType ) const;
		SparseMatrix< Real , int > gradientDualMatrix( int gradType ) const;
		Pointer( Point2D< Real > ) getGradient( ConstPointer( Real ) vertexValues , int gradType                                                                    ) const;
		void                       setGradient( ConstPointer( Real ) vertexValues , int gradType , Pointer( Point2D< Real > ) gradients , int systemFlag=SYSTEM_SET ) const;
		// Mass and stiffness matrices for vector fields expressed in terms of the (rotated) gradients of the hat basis functions
		SparseMatrix< Real , int > gradientMassMatrix( int gradType ) const;
		SparseMatrix< Real , int > gradientStiffnessMatrix( int gradType ) const;

		// Compute the vector-field system matrix, expressed in terms of triangle coefficients
		SquareMatrix< Real , 2 > vectorFieldDotMass( int t , Point2D< Real > v ) const;
		SparseMatrix< Real , int > vectorFieldRotate90Matrix( void ) const;
		SparseMatrix< Real , int > vectorFieldMassMatrix( void ) const;
		SparseMatrix< Real , int > vectorFieldDotMassMatrix( ConstPointer( Point2D< Real > ) vectorField ) const;
		SparseMatrix< Real , int > vectorFieldStiffnessMatrix( int dualType , int quadratureType=0 ) const;
		// Construct the connection
		// [NOTE] If linear fit is disabled, vectorFieldStiffnessMatrix_ gives the same matrix as vectorFieldStiffnessMatrix (though with additional zero-valued entries)
		SparseMatrix< Real , int > vectorFieldStiffnessMatrix ( ConstPointer( EdgeXForm< Real > ) edges , int dualType , int quadratureType=0 ) const;
		SparseMatrix< Real , int > vectorFieldStiffnessMatrix_( ConstPointer( EdgeXForm< Real > ) edges , int dualType , int quadratureType=0 , bool linearFit=true ) const;
		SparseMatrix< Real, int > vectorFieldStiffnessMatrix(ConstPointer(EdgeXForm< Real >) edges, const std::vector<int> & triangleIndices, int dualType, int quadratureType = 0) const;
		

		SparseMatrix< Real , int > vectorFieldStiffnessMatrix( ConstPointer( EdgeXForm< Real > ) edges , ConstPointer( Point2D< Real > ) centers ) const;
		SparseMatrix< Real , int > vectorFieldCovariantDerivativeTraceMatrix( int dualType ) const;
		SparseMatrix< Real , int > vectorFieldCovariantDerivativeTraceMatrix( ConstPointer( EdgeXForm< Real > ) edges , int dualType ) const;
		SparseMatrix< Real , int > vectorFieldCovariantDerivativeTraceMatrix2( int dualType ) const;
		SparseMatrix< Real , int > vectorFieldCovariantDerivativeTraceMatrix2( ConstPointer( EdgeXForm< Real > ) edges , int dualType ) const;
		SparseMatrix< Real , int > vectorFieldDivergenceMatrix( void ) const;
		SparseMatrix< Real , int > vectorFieldDivergenceMatrix( ConstPointer( EdgeXForm< Real > ) edges ) const;

		void setVectorFieldDot    ( ConstPointer( Point2D< Real > ) vf1 , ConstPointer( Point2D< Real > ) vf2 , Pointer( Real ) tValues , int systemFlag=SYSTEM_SET ) const;
		void setVectorFieldDotDual( ConstPointer( Point2D< Real > ) vf1 , ConstPointer( Real ) tValues , Pointer( Point2D< Real > ) vf2 , int systemFlag=SYSTEM_SET ) const;

		// Integrate the piecewise linear function over the mesh
		Real getIntegral( ConstPointer( Real ) coefficients ) const;
		Real getDotProduct( ConstPointer( Real ) c1 , ConstPointer( Real ) c2 , bool lump ) const;

		void rotate90( Pointer( Point2D< Real > ) vf ) const;

		void setTriangleDerivativeDirections( int t , ConstPointer( EdgeXForm< Real > ) edges , int dualType , Point2D< Real > dirs[3] ) const;

		CoordinateXForm< Real > exp ( ConstPointer( EdgeXForm< Real > ) edgeXForms , HermiteSamplePoint< Real >& p , Real eps=(Real)0 ) const;
		CoordinateXForm< Real > flow( ConstPointer( EdgeXForm< Real > ) edgeXForms , ConstPointer( Point2D< Real > ) vf , Real flowTime , SamplePoint< Real >& p , Real minStepSize , Real eps=(Real)0 , std::vector< SamplePoint< Real > >* path=NULL ) const;
		CoordinateXForm< Real > gradientFlow(ConstPointer(EdgeXForm< Real >) edgeXForms, ConstPointer(Real) f, SamplePoint< Real >& p, Real minStepSize, const int targetVertexIndex, Real & totalTime, std::vector<int> & traversedTriangles, Real eps = (Real)0, std::vector< SamplePoint< Real > >* path = NULL) const;
		CoordinateXForm< Real > whitneyFlow(ConstPointer(EdgeXForm< Real >) edgeXForms, ConstPointer(Real) ce, Real flowTime, SamplePoint< Real >& p, Real minStepSize, Real eps = (Real)0, std::vector< SamplePoint< Real > >* path = NULL) const;


		Real flow( ConstPointer( EdgeXForm< Real > ) edgeXForms , ConstPointer( Point2D< Real > ) vf , Real flowTime , SamplePoint< Real >& p , CoordinateXForm< Real >& xForm , Real minStepSize , Real eps=(Real)0 ) const;
		
		Pointer( EdgeXForm< Real > ) getEdgeXForms( void ) const;
		void                         setEdgeXForms( Pointer( EdgeXForm< Real > ) edgeXForms ) const;
		std::vector< int > getVertexCorners( ConstPointer( EdgeXForm< Real > ) edgeXForms , int t , int v ) const;
		Real getVertexConeAngle( ConstPointer( EdgeXForm< Real > ) edgeXForms , int t , int v ) const;
		CoordinateXForm< Real > getVertexXForm( ConstPointer( EdgeXForm< Real > ) edgeXForms , int t , int v ) const;
		bool edgeFlip( Pointer( EdgeXForm< Real > ) , int e , Real eps=0 );
		bool isVoronoiEdge( ConstPointer( EdgeXForm< Real > ) edgeXForms , int e , Real eps=0 ) const;

		void sanityCheck( ConstPointer( EdgeXForm< Real > ) edges , Real eps=0 ) const;
	protected:
		void _setEdgeXForm( int edge , Pointer( EdgeXForm< Real > ) edgeXForms ) const;
		SparseMatrix< Real , int > _scalarMatrix( bool mass , bool lump ) const;
	};
}
template< class Real > const char* FEM::RightTriangle< Real >::DualNames[] = { "barycentric" , "circumcentric" , "circumcenter projected barycentric" , "incentric" , "isogonic" , "isogon projected barycentric" };

template< class Real > const Point2D< Real > FEM::RightTriangle< Real >::Corners[] = { Point2D< Real >(0,0) , Point2D< Real >(1,0) , Point2D< Real >(0,1) };
template< class Real > const Point2D< Real > FEM::RightTriangle< Real >::EdgeMidpoints[] = { Point2D< Real >((Real)0.5,(Real)0.5) , Point2D< Real >(0,(Real)0.5) , Point2D< Real >((Real)0.5,0) };
template< class Real > const Point2D< Real > FEM::RightTriangle< Real >::Edges[] = { Point2D< Real >(-1,1) , Point2D< Real >(0,-1) , Point2D< Real >(1,0) };

#include "FEM.inl"

#endif // FEM_INCLUDED