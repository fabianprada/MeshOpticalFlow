enum {
	PROJECTED_BARICENTRIC_WEIGHTS,
	BARICENTRIC_WEIGHTS,
	INVERSE_COTANGENT_WEIGHTS
};

template<class Real>
class ConnectionVectorField : public VectorField<Real> {
public:
	ConnectionVectorField(const int mode = PROJECTED_BARICENTRIC_WEIGHTS){
		connectionMode = mode;
	}
	void InitializeCoefficients(const FEM::RiemannianMesh< Real > & mesh);
	void InitializeSmoothOperator(const FEM::RiemannianMesh< Real > & mesh);
	void InitializeProlonagtionOperator(const FEM::RiemannianMesh< Real > & mesh);
	Real MeasureGradientFieldSmoothness(const FEM::RiemannianMesh< Real > & mesh, std::vector<Real> & potential);
	int connectionMode;
};


template<class Real>
void ConnectionVectorField<Real>::InitializeCoefficients(const FEM::RiemannianMesh< Real > & mesh) {
	int tCount = mesh.tCount;
	coeffs.resize(2 * tCount,0);
}

template<class Real>
void ConnectionVectorField<Real>::InitializeSmoothOperator(const FEM::RiemannianMesh< Real > & mesh) {
	
	int tCount = mesh.tCount;
	smoothOperator.resize(2 * tCount);

	Pointer(FEM::EdgeXForm< Real >) edges = mesh.getEdgeXForms();

	Point2D< Real > e[3] = { Point2D< Real >(-1.0, 1.0), Point2D< Real >(0.0, -1.0), Point2D< Real >(1.0, 0.0) };

	Point2D< Real > center((Real)1. / 3, (Real)1. / 3);
	for (int i = 0; i<tCount; i++)
	{
		int tNeighbours = 0;
		for (int j = 0; j < 3; j++) if (edges[i * 3 + j].oppositeEdge != -1) tNeighbours++;
		smoothOperator.SetRowSize(2 * i, 2 * (tNeighbours + 1));
		smoothOperator.SetRowSize(2 * i + 1, 2 * (tNeighbours + 1));
		
		smoothOperator[2 * i + 0][0] = MatrixEntry< Real, int >(2 * i + 0, 0.0);
		smoothOperator[2 * i + 0][1] = MatrixEntry< Real, int >(2 * i + 1, 0.0);

		smoothOperator[2 * i + 1][0] = MatrixEntry< Real, int >(2 * i + 0, 0.0);
		smoothOperator[2 * i + 1][1] = MatrixEntry< Real, int >(2 * i + 1, 0.0);

		tNeighbours = 1;

		for (int j = 0; j<3; j++) if (edges[i * 3 + j].oppositeEdge != -1)
		{
			int oppositeEdge = edges[i * 3 + j].oppositeEdge, ii = oppositeEdge / 3;
			Real l;
			if (connectionMode == PROJECTED_BARICENTRIC_WEIGHTS){//Baricentric Areas / Projected Baricentric Distances = Edge Lenght Squared /Baricentric Areas
				int jj = oppositeEdge % 3;
				l = Point2D< Real >::Dot(e[j], mesh.g[i] * e[j]) / (4.0 * (mesh.area(i) + mesh.area(ii))/3.0);
			}
			else if (connectionMode == BARICENTRIC_WEIGHTS) {//Baricentric Areas / Baricentric Distances
				Point2D< Real > d = center - edges[oppositeEdge].xForm(center);
				l = ((mesh.area(i) + mesh.area(ii))/3.0) / Point2D< Real >::Dot(d, mesh.g[i] * d);
			}
			else if (connectionMode == INVERSE_COTANGENT_WEIGHTS) {//Inverse Cotangent
				int jj = oppositeEdge % 3;
				l = 1.0 / (Point2D< Real >::Dot(-e[(j + 1) % 3], mesh.g[i] * e[(j + 2) % 3]) / (2.0 * mesh.area(i)) + Point2D< Real >::Dot(-e[(jj + 1) % 3], mesh.g[ii] * e[(jj + 2) % 3]) / (2.0*mesh.area(ii)));
			}
			else {
				printf("Undefined Connection Mode \n");
			}

			//stiffnessTriplets.push_back(E_Triplet(2 * i + 0, 2 * i + 0, l*mesh.g[i](0, 0)));
			//stiffnessTriplets.push_back(E_Triplet(2 * i + 1, 2 * i + 0, l*mesh.g[i](0, 1)));
			//stiffnessTriplets.push_back(E_Triplet(2 * i + 0, 2 * i + 1, l*mesh.g[i](1, 0)));
			//stiffnessTriplets.push_back(E_Triplet(2 * i + 1, 2 * i + 1, l*mesh.g[i](1, 1)));

			smoothOperator[2 * i + 0][0].Value += l*mesh.g[i](0, 0);
			smoothOperator[2 * i + 0][1].Value += l*mesh.g[i](1, 0);
			smoothOperator[2 * i + 1][0].Value += l*mesh.g[i](0, 1);
			smoothOperator[2 * i + 1][1].Value += l*mesh.g[i](1, 1);

			SquareMatrix< Real, 2 > xPort = mesh.g[i] * edges[oppositeEdge].xForm.linear;
			//stiffnessTriplets.push_back(E_Triplet(2 * i + 0, 2 * ii + 0, -xPort(0, 0)*l));
			//stiffnessTriplets.push_back(E_Triplet(2 * i + 1, 2 * ii + 0, -xPort(0, 1)*l));
			//stiffnessTriplets.push_back(E_Triplet(2 * i + 0, 2 * ii + 1, -xPort(1, 0)*l));
			//stiffnessTriplets.push_back(E_Triplet(2 * i + 1, 2 * ii + 1, -xPort(1, 1)*l));

			smoothOperator[2 * i + 0][2 * tNeighbours + 0] = MatrixEntry< Real, int >(2 * ii + 0, -xPort(0, 0)*l);
			smoothOperator[2 * i + 0][2 * tNeighbours + 1] = MatrixEntry< Real, int >(2 * ii + 1, -xPort(1, 0)*l);
			smoothOperator[2 * i + 1][2 * tNeighbours + 0] = MatrixEntry< Real, int >(2 * ii + 0, -xPort(0, 1)*l);
			smoothOperator[2 * i + 1][2 * tNeighbours + 1] = MatrixEntry< Real, int >(2 * ii + 1, -xPort(1, 1)*l);

			tNeighbours++;
		}
	}
}

template<class Real>
void ConnectionVectorField<Real>::InitializeProlonagtionOperator(const FEM::RiemannianMesh< Real > & mesh) {
	int tCount = mesh.tCount;
	prolongationOperator.resize(2 * tCount);
	for (int i = 0; i < 2 * tCount; i++){
		prolongationOperator.SetRowSize(i, 1);
		prolongationOperator[i][0] = MatrixEntry< Real, int >(i, 1.0);
	}
	restrictionOperator = prolongationOperator.transpose();
}

template<class Real>
Real ConnectionVectorField<Real>::MeasureGradientFieldSmoothness(const FEM::RiemannianMesh< Real > & mesh, std::vector<Real> & potential) {
	Pointer(TriangleIndex) triangles = mesh.triangles;
	int tCount = mesh.tCount;
	
	std::vector<Real> gradient(2 * tCount);

	for (int i = 0; i < tCount; i++) {
		Point2D<Real> diff(potential[triangles[i][1]] - potential[triangles[i][0]], potential[triangles[i][2]] - potential[triangles[i][0]]);
		SquareMatrix< Real, 2 > gInverse = mesh.g[i].inverse();
		Point2D<Real> paramGrad = gInverse * diff;
		gradient[2 * i + 0] = paramGrad[0];
		gradient[2 * i + 1] = paramGrad[1];
	}

	std::vector<Real> smoothOperatorDotGradient(2 * tCount);
	smoothOperator.Multiply(GetPointer(gradient), GetPointer(smoothOperatorDotGradient));
	Real gradientDotSmoothDotGraddient = Dot(GetPointer(gradient), GetPointer(smoothOperatorDotGradient), 2 * tCount);
	return gradientDotSmoothDotGraddient;
}