#ifndef EIGENVALUE_SOLVER_INCLUDE
#define EIGENVALUE_SOLVER_INCLUDE

#define USE_PARDISO_FOR_SPECTRUM 0

#include "arpack/arrgsym.h"
#if USE_PARDISO_FOR_SPECTRUM
#include <Eigen/PardisoSupport>
#else
#include <Eigen/Sparse>
#endif
#include "Eigen/QR"
#include "Eigen/LU"
#include "Eigen/SVD"

typedef Eigen::SparseMatrix<double> E_SparseMatrix; 
typedef Eigen::Triplet<double> E_Triplet;
typedef Eigen::VectorXd E_Vector;
#if USE_PARDISO_FOR_SPECTRUM
typedef Eigen::PardisoLDLT<E_SparseMatrix > E_Cholesky;
#else
typedef Eigen::SimplicialLDLT<E_SparseMatrix> E_Cholesky;
#endif
typedef Eigen::MatrixXd E_Matrix;

int ReadVector(E_Vector & vec, const char * fileName){

	FILE * file;
	file = fopen(fileName, "rb");
	if (!file) { printf("Unable to read %s \n", fileName); return 0; }
	int vecSize;
	fread(&vecSize, sizeof(int), 1, file);
	vec.resize(vecSize);
	fread(vec.data(), sizeof(double), vecSize, file);
	fclose(file);
	return 1;
}

void WriteVector(const E_Vector & vec, const char * fileName){

	FILE * file;
	file = fopen(fileName, "wb");
	int vecSize = (int)vec.size();
	fwrite(&vecSize, sizeof(int), 1, file);
	fwrite(vec.data(), sizeof(double), vecSize, file);
	fclose(file);
}

int ReadMatrix(E_Matrix & mat, const char * fileName){

	FILE * file;
	file = fopen(fileName, "rb");
	if (!file) { printf("Unable to read %s \n", fileName); return 0; }
	int nrows;
	int ncols;
	fread(&nrows, sizeof(int), 1, file);
	fread(&ncols, sizeof(int), 1, file);
	//printf("%d x %d \n", nrows, ncols);
	mat.resize(nrows,ncols);
	fread(mat.data(), sizeof(double), nrows*ncols, file);
	fclose(file);
	return 1;
}

void WriteMatrix(const E_Matrix & mat, const char * fileName){

	FILE * file;
	file = fopen(fileName, "wb");
	int nrows = mat.rows();
	int ncols = mat.cols();
	//printf("%d x %d \n", nrows, ncols);
	fwrite(&nrows, sizeof(int), 1, file);
	fwrite(&ncols, sizeof(int), 1, file);
	fwrite(mat.data(), sizeof(double), nrows*ncols, file);
	fclose(file);
}

//Solve Ax = lambda*Bx. A,B must be symmetric, A positive-semidefinite and B positive-definite
class SparseEigenProblem{
public:
	SparseEigenProblem(const int p_dimension, const E_SparseMatrix & p_A, const E_SparseMatrix & p_B){
		dimension = p_dimension;
		A = p_A;
		B = p_B;
	}

	SparseEigenProblem(const int p_dimension, const std::vector<E_Triplet> & triplets_A, const std::vector<E_Triplet> & triplets_B){
		dimension = p_dimension;
		A = E_SparseMatrix(dimension, dimension);
		B = E_SparseMatrix(dimension, dimension);
		A.setFromTriplets(triplets_A.begin(), triplets_A.end());
		B.setFromTriplets(triplets_B.begin(), triplets_B.end());
	}
	int dimension;
	std::vector<double> eigenValues;
	std::vector<E_Vector> eigenVectors;
	E_SparseMatrix A;
	E_SparseMatrix B;
	/*
	mode:
	"LM" = Largest Magnitude (default)
	"SM" = Smallest Magnitude
	"LA" = Largest  Algebraic Value
	"SA" = Smallest Algebraic Value
	"BE" = Both End
	*/
	bool ComputePartialSpectrum_RegularMode(const int numEigenvectors, char * mode = "LM");
	bool ComputePartialSpectrum_ShiftedMode(const int numEigenvectors, char * mode = "LM", double spectral_shift = 0.0);
	bool ComputeEigenvectors(ARrcSymGenEig<double> & prob, const int numEigenvectors);
};

bool SparseEigenProblem::ComputeEigenvectors(ARrcSymGenEig<double> & prob, const int numEigenvectors){
	// Finding eigenvalues and eigenvectors.
	prob.FindEigenvectors();
	// Printing solution.
	int nconv;
	nconv = prob.ConvergedEigenvalues();
	std::cout << "Real symmetric eigenvalue problem: A*x - B*x*lambda" << std::endl;
	std::cout << "Dimension of the system            : " << dimension << std::endl;
	std::cout << "Number of 'requested' eigenvalues  : " << numEigenvectors << std::endl;
	std::cout << "Number of 'converged' eigenvalues  : " << nconv << std::endl;
	std::cout << "Number of Arnoldi vectors generated: " << prob.GetNcv() << std::endl;
	std::cout << "Number of iterations taken         : " << prob.GetIter() << std::endl;
	std::cout << std::endl;

	if (!prob.EigenvaluesFound()) return false;
	if (nconv < numEigenvectors) return false;

	eigenValues.resize(numEigenvectors);
	eigenVectors.resize(numEigenvectors);
	std::cout << "Eigenvalues:" << std::endl;
	for (int j = 0; j< numEigenvectors; j++) {
		eigenValues[j] = prob.Eigenvalue(j);
		eigenVectors[j] = E_Vector(dimension);
		for (int i = 0; i < dimension; i++)eigenVectors[j][i] = prob.Eigenvector(j, i);
	}
	for (int j = 0; j < numEigenvectors; j++) printf("%0.8f \n", eigenValues[j]);
	return true;
}

bool SparseEigenProblem::ComputePartialSpectrum_RegularMode(const int numEigenvectors, char * mode){
	printf("Computing Cholesky(B) \n");
	E_Cholesky B_Cholesky(B);
	printf("Solving Eigenvalue Problem\n");
	ARrcSymGenEig<double> prob(dimension, numEigenvectors, mode);
	E_Vector E_getVector(dimension);
	E_Vector E_putVector(dimension);
	double* A_getVector;
	double* A_putVector;
	// Finding an Arnoldi basis.
	int iter = 0;
	while (!prob.ArnoldiBasisFound()){
		iter++;
		printf("Arnoldi Iter %07d \r", iter);
		prob.TakeStep();
		int probIdo = prob.GetIdo();
		if ((probIdo == 1) || (probIdo == -1) || (probIdo == 2)){
			A_getVector = prob.GetVector();
			A_putVector = prob.PutVector();
			for (int i = 0; i < dimension; i++) E_getVector[i] = A_getVector[i];
			if ((probIdo == 1) || (probIdo == -1)) {
				//w <-inv(B)*A*v, v<-A*v
				E_getVector = A*E_getVector;
				E_putVector = B_Cholesky.solve(E_getVector);
				for (int i = 0; i < dimension; i++) A_getVector[i] = E_getVector[i];
			}
			else if (probIdo == 2) {
				//w <- B*v.
				E_putVector = B*E_getVector;
			}
			for (int i = 0; i < dimension; i++) A_putVector[i] = E_putVector[i];
		}
	}
	return ComputeEigenvectors(prob, numEigenvectors);
}
bool SparseEigenProblem::ComputePartialSpectrum_ShiftedMode(const int numEigenvectors, char * mode, double spectral_shift){
	printf("Computing Cholesky(A - shift*B) \n");
	E_Cholesky A_minus_shift_B_Cholesky(A - spectral_shift*B);
	printf("Solving Eigenvalue Problem\n");
	ARrcSymGenEig<double> prob('S', dimension, numEigenvectors, spectral_shift, mode);
	
	E_Vector E_getVector(dimension);
	E_Vector E_putVector(dimension);

	double* A_getVector;
	double* A_putVector;
	// Finding an Arnoldi basis.
	int iter = 0;
	while (!prob.ArnoldiBasisFound()){
		iter++;
		printf("Arnoldi Iter %07d \r", iter);
		prob.TakeStep();
		int probIdo = prob.GetIdo();
		if ((probIdo == -1) || (probIdo == 1) || (probIdo == 2)){
			A_putVector = prob.PutVector();
			if (probIdo == -1){
					//w <-inv(A-shift*B)*B*v
					A_getVector = prob.GetVector();
					for (int i = 0; i < dimension; i++) E_getVector[i] = A_getVector[i];
					E_getVector = B*E_getVector;
					E_putVector = A_minus_shift_B_Cholesky.solve(E_getVector);
			}
			else if (probIdo == 1){
					//w <-inv(A-shift*B)*u
					A_getVector = prob.GetProd();
					for (int i = 0; i < dimension; i++) E_getVector[i] = A_getVector[i];
					E_putVector = A_minus_shift_B_Cholesky.solve(E_getVector);
			}
			else{//(probIdo == 2)
				//w <- B*v.
				A_getVector = prob.GetVector();
				for (int i = 0; i < dimension; i++) E_getVector[i] = A_getVector[i];
				E_putVector = B*E_getVector;
			}
			for (int i = 0; i < dimension; i++) A_putVector[i] = E_putVector[i];
		}
	}
	return ComputeEigenvectors(prob, numEigenvectors);
}
#endif// EIGENVALUE_SOLVER_INCLUDE