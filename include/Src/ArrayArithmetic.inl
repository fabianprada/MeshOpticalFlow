template< class Real >
Real SquareDifference(ConstPointer(Real) values1, ConstPointer(Real) values2, size_t size)
{
	Real diff2 = (Real)0;
	for (int i = 0; i<size; i++) diff2 += (values1[i] - values2[i]) * (values1[i] - values2[i]);
	return diff2;
}
template< class Real >
Real SquareNorm(ConstPointer(Real) values, size_t size)
{
	Real norm2 = (Real)0;
	for (int i = 0; i<size; i++) norm2 += values[i] * values[i];
	return norm2;
}
template< class Real >
Real Dot(ConstPointer(Real) values1, ConstPointer(Real) values2, size_t size)
{
	Real dot = (Real)0;
	for (int i = 0; i<size; i++) dot += values1[i] * values2[i];
	return dot;
}