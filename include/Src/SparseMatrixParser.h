#pragma once

class matrixRowEntrie {
public:
	matrixRowEntrie(int p_index, double p_value) {
		index = p_index;
		value = p_value;
	}
	int index;
	double value;
};

struct matrixRowEntrieCompare {
	bool operator() (const matrixRowEntrie& lhs, const matrixRowEntrie& rhs) const
	{
		return lhs.index <rhs.index;
	}
};

template< class RealIn, class RealOut>
void SparseMatrixParser(const SparseMatrix< RealIn, int > & _M, Eigen::SparseMatrix<RealOut> & M) {
	int rows = _M.rows;
	int cols = 0;
	std::vector<Eigen::Triplet<RealOut>> triplets;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < _M.rowSizes[i]; j++) {
			int col = _M[i][j].N;
			double value = _M[i][j].Value;
			triplets.push_back(Eigen::Triplet<RealOut>(i, col, value));
			cols = std::max<int>(col, cols);
		}
	}
	M.resize(rows, cols + 1);
	M.setFromTriplets(triplets.begin(), triplets.end());
}

template< class RealIn, class RealOut>
void SparseMatrixParser(const Eigen::SparseMatrix<RealIn> & M, SparseMatrix< RealOut, int > & _M) {

	int nrows = M.rows();

	std::vector<std::set<matrixRowEntrie, matrixRowEntrieCompare>> rowValues(nrows);
	for (int k = 0; k < M.outerSize(); ++k) {
		for (Eigen::SparseMatrix<RealIn>::InnerIterator it(M, k); it; ++it) {
			RealIn value = it.value();
			int row = it.row();
			int col = it.col();
			matrixRowEntrie newEntrie(col, value);
			auto setIter = rowValues[row].find(newEntrie);
			if (setIter == rowValues[row].end()) {
				rowValues[row].insert(newEntrie);
			}
			else {
				matrixRowEntrie oldEntrie = *setIter;
				oldEntrie.value += value;
				rowValues[row].erase(setIter);
				rowValues[row].insert(oldEntrie);
			}
		}
	}

	_M.resize(nrows);
	for (int i = 0; i < rowValues.size(); i++) {
		std::set<matrixRowEntrie, matrixRowEntrieCompare> currentRowValues = rowValues[i];
		int numEntries = currentRowValues.size();
		_M.SetRowSize(i, numEntries);
		int offset = 0;
		for (auto iter = currentRowValues.begin(); iter != currentRowValues.end(); iter++) {
			matrixRowEntrie entrie = *iter;
			_M[i][offset] = MatrixEntry< RealOut, int >(entrie.index, entrie.value);
			offset++;
		}
	}
}

template< class RealIn, class RealOut>
void SparseMatrixParser(const Eigen::SparseMatrix<RealIn> & M, SparseMatrix< RealOut, int > & _M, bool useSymmetry) {
	int nrows = M.rows();
	_M.resize(nrows);

	auto ParseAndCompressMatrix = [&](const Eigen::SparseMatrix<RealIn> & Mt) {
#pragma omp parallel for
		for (int k = 0; k < Mt.outerSize(); ++k) {
			//Compact Row
			std::map<int, double> rowValues;
			for (Eigen::SparseMatrix<RealIn>::InnerIterator it(Mt, k); it; ++it) {
				RealIn value = it.value();
				int row = it.row();
				auto mapIter = rowValues.find(row);
				if (mapIter == rowValues.end()) {
					rowValues[row] = value;
				}
				else {
					mapIter->second += value;
				}
			}
			//Write Row
			int numEntries = rowValues.size();
			_M.SetRowSize(k, numEntries);
			int offset = 0;
			for (auto iter = rowValues.begin(); iter != rowValues.end(); iter++) {
				_M[k][offset] = MatrixEntry< RealOut, int >(iter->first, iter->second);
				offset++;
			}
		}
	};

	if (useSymmetry) {
		const Eigen::SparseMatrix<RealIn> & Mt = M;
		ParseAndCompressMatrix(Mt);
	}
	else {
		const Eigen::SparseMatrix<RealIn> & Mt = M.transpose();
		ParseAndCompressMatrix(Mt);
	}
}
