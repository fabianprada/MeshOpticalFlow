#include <Misha/LinearSolvers.h>

enum{
	WHITNEY_VECTOR_FIELD,
	CONFORMAL_VECTOR_FIELD,
	CONNECTION_VECTOR_FIELD
};

template<class Real>
class VectorField{
public:
	void Init(const FEM::RiemannianMesh< Real > & mesh);
	virtual void InitializeCoefficients(const FEM::RiemannianMesh< Real > & mesh) = 0;
	virtual void InitializeSmoothOperator(const FEM::RiemannianMesh< Real > & mesh) = 0;
	virtual void InitializeProlonagtionOperator(const FEM::RiemannianMesh< Real > & mesh) = 0;
	bool initializedSolver;
	SparseMatrix<Real, int> prolongationOperator;
	SparseMatrix<Real, int> restrictionOperator;
	SparseMatrix<Real, int> smoothOperator;
	std::vector<Real> coeffs;
	EigenCholeskySolverLDLt * opticalFlowSolver;

	void UpdateOpticalFlow(const SparseMatrix<Real, int> & dataTerm, const std::vector<Real> & rhs, Real vectorSmoothWeight, std::vector<Point2D<Real>> & tField);
	void GetTriangleVectorField(std::vector<Point2D<Real>> & tField) const;
	Real GetVectorFieldSmoothness() const;
	virtual Real MeasureGradientFieldSmoothness(const FEM::RiemannianMesh< Real > & mesh, std::vector<Real> & potential) = 0;
};

template<class Real>
void VectorField<Real>::Init(const FEM::RiemannianMesh< Real > & mesh){
	InitializeCoefficients(mesh);
	InitializeSmoothOperator(mesh);
	InitializeProlonagtionOperator(mesh);
	initializedSolver = false;
}


template<class Real>
Real VectorField<Real>::GetVectorFieldSmoothness() const {
	std::vector<Real> smoothCoeffs(coeffs.size());
	smoothOperator.Multiply(GetPointer(coeffs), GetPointer(smoothCoeffs));
	return Dot(GetPointer(coeffs), GetPointer(smoothCoeffs), coeffs.size());
}

template<class Real>
void VectorField<Real>::UpdateOpticalFlow(const SparseMatrix<Real, int> & _dataTerm, const std::vector<Real> & _rhs, Real vectorSmoothWeight, std::vector<Point2D<Real>> & tField){
	int numCoeffs = coeffs.size();
	

	Timer t; 
	SparseMatrix<Real, int> dataTerm = restrictionOperator * _dataTerm * prolongationOperator;
	std::vector<Real> rhs(coeffs.size());
	restrictionOperator.Multiply(GetPointer(_rhs), GetPointer(rhs));
	if (Verbose.set) printf("\t System prolongation and restriction: %.4f(s)\n", t.elapsed());

	{
		Real scale = (Real)1. / sqrt(dataTerm.SquareNorm());
		if(0) printf("Scale %g \n", scale);
		dataTerm *= scale;
		for (int i = 0; i < rhs.size(); i++) rhs[i] *= scale;
	}

	if (0) printf("Num rows %d \n", smoothOperator.Rows());
	if (0) printf("Data sparsity %f \n", double(dataTerm.Entries()) / double(dataTerm.Rows()));
	if (0) printf("Smootness sparsity %f \n", double(smoothOperator.Entries()) / double(smoothOperator.Rows()));

	SparseMatrix<Real, int> opticalFlowMatrix = dataTerm + smoothOperator * vectorSmoothWeight;

	if (0) printf("Full system sparsity %f \n", double(opticalFlowMatrix.Entries()) / double(opticalFlowMatrix.Rows()));
	if (0) printf("Full system size %d \n",opticalFlowMatrix.Entries());
	std::vector<Real>solution(numCoeffs, Real(0));

	if (0) printf("\t Data Matrix  [%d x %d: %f]\n", dataTerm.rows, dataTerm.rows, (Real)dataTerm.Entries() / dataTerm.rows);
	if (0) printf("\t Smoothness Matrix  [%d x %d: %f]\n", smoothOperator.rows, smoothOperator.rows, (Real)smoothOperator.Entries() / smoothOperator.rows);

	// Solve for the regularized flow field
	if (!initializedSolver) {
		opticalFlowSolver = new EigenCholeskySolverLDLt(opticalFlowMatrix, true);
		initializedSolver = true;
	}
	t.reset();
	opticalFlowSolver->update(opticalFlowMatrix);
	if (Verbose.set) printf("\t Numerical factorization: %.4f(s)\n", t.elapsed());
	t.reset();
	opticalFlowSolver->solve(GetPointer(rhs), GetPointer(solution));
	if (Verbose.set) printf("\t Back substitution: %.4f(s)\n", t.elapsed());

	//Solve the flow field optimal scale
	t.reset();

	std::vector<Real> dataTermTimesSolution(numCoeffs);
	dataTerm.Multiply(GetPointer(solution), GetPointer(dataTermTimesSolution));
	Real denom = Dot(GetPointer(solution), GetPointer(dataTermTimesSolution),numCoeffs);
	Real num = Dot(GetPointer(solution), GetPointer(rhs),numCoeffs);

	Real step;
	if (denom) step = num / denom;
	else step = 0.0;
	if (step) for (int i = 0; i < numCoeffs; i++) coeffs[i] += solution[i] * step;

	if (Verbose.set) printf("\t GetScale: %.4f(s)\n", t.elapsed());

	GetTriangleVectorField(tField);
}

template<class Real>
void VectorField<Real>::GetTriangleVectorField(std::vector< Point2D<Real> > & tField) const{
	if (!tField.size()) printf("ERROR: Triangle field buffer not initialized! \n");
	std::vector<Real> _tField(tField.size() * 2);
	prolongationOperator.Multiply(GetPointer(coeffs), GetPointer(_tField));
	for (int i = 0; i < tField.size(); i++)tField[i] = Point2D<Real>(_tField[2 * i], _tField[2 * i + 1]);
}

#include "Whitney.inl"
#include "Conformal.inl"
#include "Connection.inl"