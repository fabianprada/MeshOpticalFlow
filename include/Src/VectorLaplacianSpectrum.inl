#include <Src/SparseMatrixParser.h>
#include <Src/EigenvalueSolver.h>

template<class Real> 
void ComputeSpectrum(VectorField<Real> * vectorField, const FEM::RiemannianMesh< Real > & mesh, const int numEigenvalues, std::vector<std::vector<Point2D<Real>>> & laplaceEigenVectors, const double shift = 1e-8){
	Pointer(TriangleIndex) triangles = mesh.triangles;
	int tCount = mesh.tCount;

	SparseMatrix<Real, int> vfMass;
	vfMass.resize(2 * tCount);
	for (int t = 0; t < tCount;t++){
		Real tMass = mesh.area(t);
		for (int k = 0; k < 2; k++){
			vfMass.SetRowSize(2 * t + k, 2);
			for (int l = 0; l < 2; l++) vfMass[2 * t + k][l] = MatrixEntry<Real, int>(2 * t + l, mesh.g[t](k, l)*tMass);
		}
	}

	SparseMatrix<Real, int> massOperator = vectorField->restrictionOperator * vfMass * vectorField->prolongationOperator;
	
	Eigen::SparseMatrix<double> _smoothOperator;
	Eigen::SparseMatrix<double> _massOperator;
	SparseMatrixParser(vectorField->smoothOperator, _smoothOperator);
	SparseMatrixParser(massOperator, _massOperator);

	SparseEigenProblem eigenProblem(_massOperator.rows(), _smoothOperator, _massOperator);
	if (!eigenProblem.ComputePartialSpectrum_ShiftedMode(numEigenvalues, "LM", shift)) {
		printf("Unable to Compute Laplacian Spectrum \n");
	}

	laplaceEigenVectors.resize(numEigenvalues);
	int numCoeffs = vectorField->coeffs.size();
	std::vector<Real> eigenVector(numCoeffs);
	std::vector<Real> prolongedEigenVector(2 * tCount);
	for (int i = 0; i < numEigenvalues; i++){
		for (int j = 0; j < numCoeffs; j++) eigenVector[j] = (Real)eigenProblem.eigenVectors[i][j];
		vectorField->prolongationOperator.Multiply(GetPointer(eigenVector), GetPointer(prolongedEigenVector));
		laplaceEigenVectors[i].resize(tCount);
		for (int j = 0; j < tCount; j++) laplaceEigenVectors[i][j] = Point2D<Real>(prolongedEigenVector[2 * j], prolongedEigenVector[2 * j + 1]);
	}
}