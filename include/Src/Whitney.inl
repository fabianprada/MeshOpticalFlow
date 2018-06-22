template<class Real>
class WhitneyVectorField : public VectorField<Real> {
public:
	void InitializeCoefficients(const FEM::RiemannianMesh< Real > & mesh);
	void InitializeSmoothOperator(const FEM::RiemannianMesh< Real > & mesh);
	void InitializeProlonagtionOperator(const FEM::RiemannianMesh< Real > & mesh);
	Real MeasureGradientFieldSmoothness(const FEM::RiemannianMesh< Real > & mesh, std::vector<Real> & potential);
	//Topology
	std::vector<int> reducedEdgeIndex;
	std::vector<int> expandedEdgeIndex;
	std::vector<bool> positiveOrientedEdge;
	SparseMatrix<Real, int> signedEdge;

	//DEC
	SparseMatrix<Real, int> d0;
	SparseMatrix<Real, int> d1;
	SparseMatrix<Real, int> m0;
	SparseMatrix<Real, int> m1;
	SparseMatrix<Real, int> m2;
	SparseMatrix<Real, int> m0_inv;
	SparseMatrix<Real, int> m1_inv;
	SparseMatrix<Real, int> m2_inv;
	SparseMatrix<Real, int> rotationalEnergy;
	SparseMatrix<Real, int> divergenceEnergy;
};

template<class Real>
void WhitneyVectorField<Real>::InitializeCoefficients(const FEM::RiemannianMesh< Real > & mesh) {
	Pointer(TriangleIndex) triangles = mesh.triangles;
	int tCount = mesh.tCount;


	reducedEdgeIndex.resize(3 * tCount, -1);
	expandedEdgeIndex.reserve(3 * tCount);
	positiveOrientedEdge.resize(3 * tCount, true);

	Pointer(FEM::EdgeXForm< Real >) edges = mesh.getEdgeXForms();

	int currentIndex = 0;
	for (int i = 0; i < tCount; i++)for (int j = 0; j < 3; j++) {
		if (reducedEdgeIndex[3 * i + j] == -1) {
			expandedEdgeIndex.push_back(3 * i + j);
			reducedEdgeIndex[3 * i + j] = currentIndex;
			int oppositeEdge = edges[3 * i + j].oppositeEdge;
			if (oppositeEdge != -1) {
				reducedEdgeIndex[oppositeEdge] = currentIndex;
				positiveOrientedEdge[oppositeEdge] = false;
			}
			currentIndex++;
		}
	}
	int eCount = expandedEdgeIndex.size();
	coeffs.resize(eCount, 0);

	signedEdge.resize(3 * tCount);
	for (int t = 0; t < tCount; t++) {
		for (int j = 0; j < 3; j++) signedEdge.SetRowSize(3 * t + j, 1);
		for (int j = 0; j < 3; j++) {
			signedEdge[3 * t + j][0] = MatrixEntry<Real, int>(reducedEdgeIndex[3 * t + j], positiveOrientedEdge[3 * t + j] ? 1 : -1);
		}
	}
}

template<class Real>
void WhitneyVectorField<Real>::InitializeProlonagtionOperator(const FEM::RiemannianMesh< Real > & mesh) {
	Pointer(TriangleIndex) triangles = mesh.triangles;
	int tCount = mesh.tCount;

	Point2D< Real > grad[3] = { Point2D< Real >(-1.0, -1.0), Point2D< Real >(1.0, 0.0), Point2D< Real >(0.0, 1.0) };

	prolongationOperator.resize(2 * tCount);
	for (int t = 0; t < tCount; t++) {
		prolongationOperator.SetRowSize(2 * t, 3);
		prolongationOperator.SetRowSize(2 * t + 1, 3);
		if (!mesh.g[t].determinant()) {
			fprintf(stderr, "[WARNING] Vanishing metric tensor determinant\n");
		}
		SquareMatrix< Real, 2 > iTensor = mesh.g[t].inverse();
		for (int k = 0; k < 3; k++) {
			Point2D<Real> gradDiff = iTensor*((grad[(k + 2) % 3] - grad[(k + 1) % 3]) / 3.0);
			if (!positiveOrientedEdge[3 * t + k]) gradDiff *= -1;
			prolongationOperator[2 * t][k] = MatrixEntry<Real, int>(reducedEdgeIndex[3 * t + k], gradDiff[0]);
			prolongationOperator[2 * t + 1][k] = MatrixEntry<Real, int>(reducedEdgeIndex[3 * t + k], gradDiff[1]);
		}
	}

	restrictionOperator = prolongationOperator.transpose();
}


template<class Real>
void WhitneyVectorField<Real>::InitializeSmoothOperator(const FEM::RiemannianMesh< Real > & mesh) {

	Pointer(TriangleIndex) triangles = mesh.triangles;
	int tCount = mesh.tCount;

	//d0
	int eCount = expandedEdgeIndex.size();
	d0.resize(eCount);
	for (int i = 0; i < eCount; i++) {
		d0.SetRowSize(i, 2);
		int t = expandedEdgeIndex[i] / 3;
		int v = expandedEdgeIndex[i] % 3;
		d0[i][0] = MatrixEntry< Real, int >(triangles[t][(v + 1) % 3], Real(-1));
		d0[i][1] = MatrixEntry< Real, int >(triangles[t][(v + 2) % 3], Real(1));
	}


	//d1
	d1.resize(tCount);
	for (int i = 0; i < tCount; i++) {
		d1.SetRowSize(i, 3);
		for (int j = 0; j < 3; j++) {
			d1[i][j] = MatrixEntry< Real, int >(reducedEdgeIndex[3 * i + j], positiveOrientedEdge[3 * i + j] ? Real(1) : Real(-1));
		}
	}

	//m0
	//Barycentric areas
	int vCount = mesh.vCount();

	std::vector<Real> baricentryArea(vCount, Real(0));
	for (int t = 0; t < tCount; t++)for (int v = 0; v < 3; v++) {
		Real area = mesh.area(t) / Real(3);
		baricentryArea[triangles[t][v]] += area;
	}
	m0.resize(vCount);
	m0_inv.resize(vCount);
	for (int i = 0; i < vCount; i++) {
		m0.SetRowSize(i, 1);
		m0[i][0] = MatrixEntry< Real, int >(i, baricentryArea[i]);

		m0_inv.SetRowSize(i, 1);
		m0_inv[i][0] = MatrixEntry< Real, int >(i, 1.0 / baricentryArea[i]);
	}

	//m1 
	Pointer(FEM::EdgeXForm< Real >) edges = mesh.getEdgeXForms();
	m1.resize(eCount);
	m1_inv.resize(eCount);
	Point2D< Real > grad[3] = { Point2D< Real >(-1.0, -1.0), Point2D< Real >(1.0, 0.0), Point2D< Real >(0.0, 1.0) };
	for (int i = 0; i < eCount; i++) {
		int t = expandedEdgeIndex[i] / 3;
		int v = expandedEdgeIndex[i] % 3;

		Real r = -mesh.area(t) * Point2D< Real >::Dot(grad[(v + 1) % 3], mesh.g[t].inverse() * grad[(v + 2) % 3]);

		int oppositeEdge = edges[3 * t + v].oppositeEdge;
		if (oppositeEdge != -1) {
			int tt = oppositeEdge / 3;
			int vv = oppositeEdge % 3;
			r += -mesh.area(tt) * Point2D< Real >::Dot(grad[(vv + 1) % 3], mesh.g[tt].inverse() * grad[(vv + 2) % 3]);
		}

		m1.SetRowSize(i, 1);
		m1[i][0] = MatrixEntry< Real, int >(i, r);

		m1_inv.SetRowSize(i, 1);
		m1_inv[i][0] = MatrixEntry< Real, int >(i, r ? 1 / r : 0);
	}


	//m2 Inverse Triangle areas

	m2.resize(tCount);
	m2_inv.resize(tCount);
	for (int i = 0; i < tCount; i++) {
		m2.SetRowSize(i, 1);
		m2[i][0] = MatrixEntry< Real, int >(i, 1.0 / mesh.area(i));

		m2_inv.SetRowSize(i, 1);
		m2_inv[i][0] = MatrixEntry< Real, int >(i, mesh.area(i));
	}

	//Set differential opertors
	rotationalEnergy = d1.transpose() * m2 * d1;
	divergenceEnergy = m1 * d0 * m0_inv * d0.transpose() * m1;

	smoothOperator = (rotationalEnergy + divergenceEnergy)* 0.5;
}


template<class Real>
Real WhitneyVectorField<Real>::MeasureGradientFieldSmoothness(const FEM::RiemannianMesh< Real > & mesh, std::vector<Real> & potential) {
	int eCount = expandedEdgeIndex.size();
	std::vector<Real> gradient(eCount);
	d0.Multiply(GetPointer(potential), GetPointer(gradient));

	std::vector<Real> smoothOperatorDotGradient(eCount);
	smoothOperator.Multiply(GetPointer(gradient), GetPointer(smoothOperatorDotGradient));
	Real gradientDotSmoothDotGraddient = Dot(GetPointer(gradient), GetPointer(smoothOperatorDotGradient), eCount);
	return gradientDotSmoothDotGraddient;
}