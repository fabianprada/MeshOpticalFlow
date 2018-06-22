#pragma once

template< class Real >
struct PlyMetricFace
{
	unsigned int nr_vertices, nr_square_lengths;
	int *vertices;
	Real *square_lengths;
	PlyMetricFace(void) { vertices = NULL, square_lengths = NULL, nr_vertices = nr_square_lengths = 0; }
	~PlyMetricFace(void) { resize(0); }
	PlyMetricFace(const PlyMetricFace& face)
	{
		vertices = NULL, square_lengths = NULL;
		(*this) = face;
	}
	PlyMetricFace& operator = (const PlyMetricFace& face)
	{
		if (vertices) free(vertices), vertices = NULL;
		if (square_lengths) free(square_lengths), square_lengths = NULL;
		nr_vertices = face.nr_vertices, nr_square_lengths = face.nr_square_lengths;
		if (nr_vertices) vertices = (int*)malloc(sizeof(int)*nr_vertices);
		else              vertices = NULL;
		if (nr_square_lengths) square_lengths = (Real*)malloc(sizeof(Real)*nr_square_lengths);
		else                    square_lengths = NULL;
		memcpy(vertices, face.vertices, sizeof(int)*nr_vertices);
		memcpy(square_lengths, face.square_lengths, sizeof(Real)*nr_square_lengths);
		return *this;
	}
	void resize(unsigned int count)
	{
		if (vertices) free(vertices), vertices = NULL;
		if (square_lengths) free(square_lengths), square_lengths = NULL;
		nr_vertices = nr_square_lengths = 0;
		if (count)
		{
			vertices = (int*)malloc(sizeof(int)*count), nr_vertices = count;
			square_lengths = (Real*)malloc(sizeof(Real)*count), nr_square_lengths = count;
		}
	}
	int& operator[] (int idx) { return vertices[idx]; }
	const int& operator[] (int idx) const { return vertices[idx]; }
	Real  square_length(int idx) const { return square_lengths[idx]; }
	Real& square_length(int idx) { return square_lengths[idx]; }
	int size(void) const { return nr_vertices; }

	const static int Components = 2;
	static PlyProperty Properties[];
};
template<>
PlyProperty PlyMetricFace< float >::Properties[] =
{
	{ "vertex_indices", PLY_INT, PLY_INT, offsetof(PlyMetricFace, vertices), 1, PLY_INT, PLY_INT, (int)offsetof(PlyMetricFace, nr_vertices) },
	{ "square_lengths", PLY_FLOAT, PLY_FLOAT, (int)offsetof(PlyMetricFace, square_lengths), 1, PLY_INT, PLY_INT, (int)offsetof(PlyMetricFace, nr_square_lengths) },
};
template<>
PlyProperty PlyMetricFace< double >::Properties[] =
{
	{ "vertex_indices", PLY_INT, PLY_INT, offsetof(PlyMetricFace, vertices), 1, PLY_INT, PLY_INT, (int)offsetof(PlyMetricFace, nr_vertices) },
	{ "square_lengths", PLY_DOUBLE, PLY_DOUBLE, (int)offsetof(PlyMetricFace, square_lengths), 1, PLY_INT, PLY_INT, (int)offsetof(PlyMetricFace, nr_square_lengths) },
};