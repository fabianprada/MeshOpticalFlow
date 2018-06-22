template<class Real>
class ConformalVectorField : public VectorField<Real> {
public:
	void InitializeCoefficients(const FEM::RiemannianMesh< Real > & mesh);
	void InitializeSmoothOperator(const FEM::RiemannianMesh< Real > & mesh);
	void InitializeProlonagtionOperator(const FEM::RiemannianMesh< Real > & mesh);
	Real MeasureGradientFieldSmoothness(const FEM::RiemannianMesh< Real > & mesh, std::vector<Real> & potential);
	int vCount;
};

template<class Real>
void ConformalVectorField<Real>::InitializeCoefficients(const FEM::RiemannianMesh< Real > & mesh){
	vCount = mesh.vCount();
	coeffs.resize(2 * vCount, 0);
}

template<class Real>
void ConformalVectorField<Real>::InitializeSmoothOperator(const FEM::RiemannianMesh< Real > & mesh){
	
	SparseMatrix<Real, int> _mass = mesh.scalarMassMatrix(true);
	SparseMatrix<Real, int> _stiffness = mesh.scalarStiffnessMatrix();
	

	SparseMatrix<Real, int> inverseMass;
	inverseMass.resize(2 * vCount);
	for (int i = 0; i < vCount; i++) {
		Real inverse = 1.0 / _mass[i][0].Value;
		for (int offset = 0; offset < 2; offset++) {
			inverseMass.SetRowSize(i + vCount*offset, 1);
			inverseMass[i + vCount*offset][0] = MatrixEntry< Real, int >(i + vCount*offset, inverse);
		}
	}


	SparseMatrix<Real, int> stiffness;
	stiffness.resize(2 * vCount);
	for (int i = 0; i < vCount; i++) {
		for (int offset = 0; offset < 2; offset++){
			stiffness.SetRowSize(i + vCount*offset, _stiffness.RowSize(i));
			for (int j = 0; j < _stiffness.RowSize(i); j++) {
				stiffness[i + vCount*offset][j]  = MatrixEntry< Real, int >(_stiffness[i][j].N + vCount*offset, _stiffness[i][j].Value);
			}
		}
	}
	smoothOperator = (stiffness * inverseMass * stiffness)* 0.5;
}

template<class Real>
void ConformalVectorField<Real>::InitializeProlonagtionOperator(const FEM::RiemannianMesh< Real > & mesh) {
	Pointer(TriangleIndex) triangles = mesh.triangles;
	int tCount = mesh.tCount;

	Point2D< Real > grad[3] = { Point2D< Real >(-1.0, -1.0), Point2D< Real >(1.0, 0.0), Point2D< Real >(0.0, 1.0) };
	Point2D< Real > rotGrad[3] = { Point2D< Real >(1.0, -1.0), Point2D< Real >(0.0, 1.0), Point2D< Real >(-1.0, 0.0) };
	prolongationOperator.resize(2 * tCount);

	for (int t = 0; t < tCount; t++){
		prolongationOperator.SetRowSize(2 * t, 6);
		prolongationOperator.SetRowSize(2 * t + 1, 6);
		if (!mesh.g[t].determinant()) {
			fprintf(stderr, "[WARNING] Vanishing metric tensor determinant\n");
		}
		SquareMatrix< Real, 2 > gInverse = mesh.g[t].inverse();
		Real invSqrtDet = 1.0 / sqrt(mesh.g[t].determinant());
		for (int k = 0; k < 3; k++) {
			Point2D<Real> paramGrad = gInverse*grad[k];
			prolongationOperator[2 * t][k] = MatrixEntry<Real, int>(triangles[t][k], paramGrad[0]);
			prolongationOperator[2 * t + 1][k] = MatrixEntry<Real, int>(triangles[t][k], paramGrad[1]);
		}
		for (int k = 0; k < 3; k++) {
			Point2D<Real> rotParamGrad = rotGrad[k] * invSqrtDet;
			prolongationOperator[2 * t][k + 3] = MatrixEntry<Real, int>(triangles[t][k] + vCount, rotParamGrad[0]);
			prolongationOperator[2 * t + 1][k + 3] = MatrixEntry<Real, int>(triangles[t][k] + vCount, rotParamGrad[1]);
		}
	}
	restrictionOperator = prolongationOperator.transpose();
}


template<class Real>
Real ConformalVectorField<Real>::MeasureGradientFieldSmoothness(const FEM::RiemannianMesh< Real > & mesh, std::vector<Real> & potential) {
	std::vector<Real> gradient(2 * vCount,0);
	for (int i = 0; i < potential.size();i++) gradient[i] = potential[i];

	std::vector<Real> smoothOperatorDotGradient(2 * vCount);
	smoothOperator.Multiply(GetPointer(gradient), GetPointer(smoothOperatorDotGradient));
	Real gradientDotSmoothDotGraddient = Dot(GetPointer(gradient), GetPointer(smoothOperatorDotGradient), 2 * vCount);
	return gradientDotSmoothDotGraddient;

}