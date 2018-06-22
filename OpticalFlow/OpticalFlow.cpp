/*
Copyright (c) 2018, Michael Kazhdan and Fabian Prada
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution.

Neither the name of the Johns Hopkins University nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
*/

//#define EIGEN_USE_MKL_ALL
#define SMOOTH_FIRST 1
#undef ARRAY_DEBUG

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <algorithm>
#include <vector>
#include <Misha/CmdLineParser.h>
#include <Misha/Algebra.h>
#include <Misha/Ply.h>
#include <Misha/LinearSolvers.h>
#include <Misha/Timer.h>
#include <Misha/PNG.h>
#include <Misha/FEM.h>
#include <Src/MeshFlow.inl>
#include <Eigen/Sparse>
#include <Eigen/QR>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <Src/ArrayArithmetic.inl>
#include <Src/VectorField.h>
#include <Src/VectorIO.h>
#include <Src/MetricFace.h>


cmdLineParameter< char* > Mesh("mesh"), Out("out");
cmdLineParameterArray< char*, 2 > In("in");
cmdLineParameter< int > VectorFieldMode("vfMode", WHITNEY_VECTOR_FIELD), ConnectionMode("cMode", PROJECTED_BARICENTRIC_WEIGHTS), Levels("iterations", 10), Threads("threads", omp_get_max_threads()), PadRadius("pad", 2);
cmdLineParameter< float > ScalarSmoothWeight("sSmooth", 3e-3f), VectorFieldSmoothWeight("vfSmooth"), VectorFieldSmoothWeightThreshold("vfSThreshold", 1e-8f), SubdivideEdgeLength("eLength", 0.006f), DoGWeight("dogWeight", 1.f), DoGSmooth("dogSmooth", (float)1e-4), GSSearch("search", 1.f);
cmdLineParameter< float > ScalarWeightMultiplier("sMultiply", 0.25f), VectorFieldWeightMultiplier("vMultiply",1.0f);
cmdLineReadable DivergenceFree("divFree");
cmdLineReadable Verbose("verbose"), ShowError("error"), Nearest("nearest"), Debug("debug"), LogSpace("log");
cmdLineReadable* params[] = { &Mesh, &In, &Out, &VectorFieldMode, &ConnectionMode, &Levels, &Threads, &ScalarSmoothWeight, &VectorFieldSmoothWeight, &Verbose, &ShowError, &ScalarWeightMultiplier, &VectorFieldWeightMultiplier, &VectorFieldSmoothWeightThreshold, &DoGWeight, &DoGSmooth, &SubdivideEdgeLength, &Nearest, &PadRadius, &Debug, &LogSpace, &GSSearch, &DivergenceFree, NULL };

void ShowUsage(const char* ex)
{
	printf("Usage %s:\n", ex);
	printf("I/O Parameters: \n");
	printf("\t[--%s <input textures (.ply or .png)>\n", In.name);
	printf("\t[--%s <input geometry (.ply)>]\n", Mesh.name);
	printf("\t[--%s <output file (.ply or .png)>]\n", Out.name);

	printf("Processing Parameters: \n");
	printf("\t[--%s <subdivide edges up to this diagonal fraction> = %0.3f]\n", SubdivideEdgeLength.name, SubdivideEdgeLength.value);
	printf("\t[--%s <alignment iterations>=%d]\n", Levels.name, Levels.value);


	printf("Scalar Field Parameters: \n");

	printf("\t[--%s <scalar smoothing weight>=%f]\n", ScalarSmoothWeight.name, ScalarSmoothWeight.value);
	printf("\t[--%s <scalar weight multiplication factor>=%g]\n", ScalarWeightMultiplier.name, ScalarWeightMultiplier.value);
	printf("\t[--%s <difference of Gaussians blending weight>=%g]\n", DoGWeight.name, DoGWeight.value);
	printf("\t[--%s <difference of Gaussians smoothing weight>=%g]\n", DoGSmooth.name, DoGSmooth.value);

	printf("Vector Field Parameters: \n");
	printf("\t[--%s <vector field mode >=%d]\n", VectorFieldMode.name, VectorFieldMode.value);
	printf("\t \t [%d] Whitney \n", WHITNEY_VECTOR_FIELD);
	printf("\t \t [%d] Conformal \n", CONFORMAL_VECTOR_FIELD);
	printf("\t \t [%d] Connection \n", CONNECTION_VECTOR_FIELD);
	printf("\t[--%s <connection mode >=%d]\n", ConnectionMode.name, ConnectionMode.value);
	printf("\t \t [%d] Projected baricentric \n", PROJECTED_BARICENTRIC_WEIGHTS);
	printf("\t \t [%d] Baricentric dual \n", BARICENTRIC_WEIGHTS);
	printf("\t \t [%d] Inverse cotangents \n", INVERSE_COTANGENT_WEIGHTS);
	printf("\t[--%s <vector field smoothing weight>= Whitney -> %g,Conformal -> %g,Connection -> %g]\n", VectorFieldSmoothWeight.name,3e-6, 5e-7, 1e4);
	printf("\t[--%s <vector field weight multiplication factor>=%g]\n", VectorFieldWeightMultiplier.name, VectorFieldWeightMultiplier.value);
	printf("\t[--%s <vector field weight threshold>=%g]\n", VectorFieldSmoothWeightThreshold.name, VectorFieldSmoothWeightThreshold.value);

	printf("Auxiliar Parameters: \n");
	printf("\t[--%s <parallelization threads>=%d]\n", Threads.name, Threads.value);
	printf("\t[--%s <padding radius>=%d]\n", PadRadius.name, PadRadius.value);
	printf("\t[--%s <golden secition search range multiplier>=%g]\n", GSSearch.name, GSSearch.value);
	printf("\t[--%s]\n", DivergenceFree.name);
	printf("\t[--%s]\n", LogSpace.name);
	printf("\t[--%s]\n", Nearest.name);
	printf("\t[--%s]\n", ShowError.name);
	printf("\t[--%s]\n", Verbose.name);
	printf("\t[--%s]\n", Debug.name);

}

template< class Real >
void OutputImage(const char* fileName, const std::vector<Point3D< Real >> & pixels, int width, int height, bool flipY)
{
	unsigned char* _pixels = new unsigned char[width * height * 3];

	for (int i = 0; i<width; i++) for (int j = 0; j<height; j++) for (int c = 0; c<3; c++)
		if (flipY) _pixels[3 * ((height - 1 - j)*width + i) + c] = static_cast<unsigned char>(std::max< int >(0, std::min< int >(255, (int)pixels[j*width + i][c])));
		else        _pixels[3 * ((j)*width + i) + c] = static_cast<unsigned char>(std::max< int >(0, std::min< int >(255, (int)pixels[j*width + i][c])));
		char* ext = GetFileExtension(fileName);
		if (!strcasecmp(ext, "png")) PNGWriteColor(fileName, _pixels, width, height);
		else fprintf(stderr, "[ERROR] Unrecognized image extension: %s\n", ext), exit(0);
		delete[] _pixels;
}

template< class Real >
void OutputImage(const char* fileName, const Point3D< Real >* pixels, int width, int height, bool flipY)
{
	unsigned char* _pixels = new unsigned char[width * height * 3];

	for (int i = 0; i<width; i++) for (int j = 0; j<height; j++) for (int c = 0; c<3; c++)
	if (flipY) _pixels[3 * ((height - 1 - j)*width + i) + c] = (unsigned char)std::max< int >(0, std::min< int >(255, (int)pixels[j*width + i][c]));
	else        _pixels[3 * ((j)*width + i) + c] = (unsigned char)std::max< int >(0, std::min< int >(255, (int)pixels[j*width + i][c]));
	char* ext = GetFileExtension(fileName);
	if (!strcasecmp(ext, "png")) PNGWriteColor(fileName, _pixels, width, height);
	else fprintf(stderr, "[ERROR] Unrecognized image extension: %s\n", ext), exit(0);
	delete[] _pixels;
}
template< class Real >
void OutputMesh(const char* fileName, const std::vector< Point3D< Real > >& vertices, const std::vector< Point3D< Real > >& colors, const std::vector< TriangleIndex >& triangles, int file_type)
{
	std::vector< PlyColorVertex< float > > _vertices(vertices.size());
	for (int i = 0; i<vertices.size(); i++)
	{
		_vertices[i].point = Point3D< float >(vertices[i]), _vertices[i].color = Point3D< float >(colors[i]);
		for (int j = 0; j<3; j++) _vertices[i].color[j] = std::min< float >(255.f, std::max< float >(0.f, _vertices[i].color[j]));
	}
	PlyWriteTriangles(fileName, _vertices, triangles, PlyColorVertex< float >::WriteProperties, PlyColorVertex< float >::WriteComponents, file_type);
}
template< class Real, int Channels >
void OutputMesh(const char* fileName, const std::vector< Point3D< Real > >& vertices, const std::vector< Point< Real, Channels > >& colors, const std::vector< TriangleIndex >& triangles, int file_type)
{
	if (Channels != 3 && Channels != 6){ fprintf(stderr, "[WARNING] Can only output mesh for 3 and 6 channel signals\n"); return; }
	std::vector< PlyColorVertex< float > > _vertices(vertices.size());
	for (int i = 0; i<vertices.size(); i++)
	{
		Point< float, Channels > color;
		_vertices[i].point = Point3D< float >(vertices[i]), color = Point< float, Channels >(colors[i]);
		if (Channels == 3) for (int j = 0; j<3; j++) _vertices[i].color[j] = std::min< float >(255.f, std::max< float >(0.f, color[j]));
		else if (Channels == 6) for (int j = 0; j<3; j++) _vertices[i].color[j] = std::min< float >(255.f, std::max< float >(0.f, color[j] + color[j + 3]));
	}
	PlyWriteTriangles(fileName, _vertices, triangles, PlyColorVertex< float >::WriteProperties, PlyColorVertex< float >::WriteComponents, file_type);
}
template< class Real >
void OutputMesh(const char* fileName, const std::vector< Point3D< Real > >& vertices, const std::vector< Point2D< Real > >& flowField, const std::vector< TriangleIndex >& triangles, int file_type)
{
	std::vector< PlyVertex< float > > _vertices(vertices.size());
	std::vector< PlyVFFace< float > > _triangles(triangles.size());
	for (int i = 0; i<triangles.size(); i++)
	{
		_triangles[i].resize(3);
		for (int j = 0; j<3; j++) _triangles[i][j] = triangles[i][j];
		_triangles[i].v =
			(vertices[triangles[i][1]] - vertices[triangles[i][0]]) * (float)flowField[i][0] +
			(vertices[triangles[i][2]] - vertices[triangles[i][0]]) * (float)flowField[i][1];
	}
	for (int i = 0; i<vertices.size(); i++) _vertices[i].point = Point3D< float >(vertices[i]);
	PlyWritePolygons(fileName, _vertices, _triangles, PlyVertex< float >::WriteProperties, PlyVertex< float >::WriteComponents, PlyVFFace< float >::WriteProperties, PlyVFFace< float >::WriteComponents, PLY_BINARY_NATIVE);
}
template< class Real, class V >
V Sample(const std::vector< V >& values, const std::vector< TriangleIndex >& triangles, FEM::SamplePoint< Real > p)
{
	return
		values[triangles[p.tIdx][0]] * (Real)(1. - p.p[0] - p.p[1]) +
		values[triangles[p.tIdx][1]] * (Real)(p.p[0]) +
		values[triangles[p.tIdx][2]] * (Real)(p.p[1]);
}
template< class Real, class V >
V Sample(ConstPointer(V) values, ConstPointer(TriangleIndex) triangles, FEM::SamplePoint< Real > p)
{
	return
		values[triangles[p.tIdx][0]] * (Real)(1. - p.p[0] - p.p[1]) +
		values[triangles[p.tIdx][1]] * (Real)(p.p[0]) +
		values[triangles[p.tIdx][2]] * (Real)(p.p[1]);
}


template< class Real, class V >
void ResampleSignal(const FEM::RiemannianMesh< Real >& mesh, ConstPointer(Point2D< Real >) flowField, ConstPointer(FEM::EdgeXForm< Real >) edges, const std::vector< V >& in, std::vector< V >& out, Real length, int threads = 1)
{
	std::vector< int > counts(in.size(), 0);
	out.resize(in.size());

#pragma omp parallel for num_threads( threads )
	for (int i = 0; i<in.size(); i++) out[i] *= (Real)0;

	for (int i = 0; i<mesh.tCount; i++)
	{
		FEM::SamplePoint< Real > p(i, Point2D< Real >((Real)1. / 3, (Real)1. / 3));
		mesh.flow(edges, flowField, length, p, (Real)1e-2);
		V c = Sample< Real, V >(GetPointer(in), mesh.triangles, p);
		for (int j = 0; j<3; j++) out[mesh.triangles[i][j]] += c, counts[mesh.triangles[i][j]]++;
	}

#pragma omp parallel for num_threads( threads )
	for (int i = 0; i<in.size(); i++) out[i] /= (Real)counts[i];
}

template< class Real, class V >
void ResampleSignalWhitney(const FEM::RiemannianMesh< Real >& mesh, ConstPointer(Real) flowField, ConstPointer(FEM::EdgeXForm< Real >) edges, const std::vector< V >& in, std::vector< V >& out, Real length, int threads = 1)
{
	std::vector< int > counts(in.size(), 0);
	out.resize(in.size());

#pragma omp parallel for num_threads( threads )
	for (int i = 0; i<in.size(); i++) out[i] *= (Real)0;

	for (int i = 0; i<mesh.tCount; i++)
	{
		FEM::SamplePoint< Real > p(i, Point2D< Real >((Real)1. / 3, (Real)1. / 3));
		mesh.whitneyFlow(edges, flowField, length, p, (Real)1e-2);
		V c = Sample< Real, V >(GetPointer(in), mesh.triangles, p);
		for (int j = 0; j<3; j++) out[mesh.triangles[i][j]] += c, counts[mesh.triangles[i][j]]++;
	}

#pragma omp parallel for num_threads( threads )
	for (int i = 0; i<in.size(); i++) out[i] /= (Real)counts[i];
}

template< class Real, class V >
void ResampleSignalWhitneyComposedFlow(const FEM::RiemannianMesh< Real >& mesh, const std::vector<std::vector<Real>> & flowField, ConstPointer(FEM::EdgeXForm< Real >) edges, const std::vector< V >& in, std::vector< V >& out, Real length, int threads = 1)
{
	std::vector< int > counts(in.size(), 0);
	out.resize(in.size());

#pragma omp parallel for num_threads( threads )
	for (int i = 0; i<in.size(); i++) out[i] *= (Real)0;

	for (int i = 0; i<mesh.tCount; i++)
	{
		FEM::SamplePoint< Real > p(i, Point2D< Real >((Real)1. / 3, (Real)1. / 3));
		for (int f = flowField.size() - 1; f >= 0; f--){
			mesh.whitneyFlow(edges, (ConstPointer(Real))GetPointer(flowField[f]), length, p, (Real)1e-2);
		}
		V c = Sample< Real, V >(GetPointer(in), mesh.triangles, p);
		for (int j = 0; j<3; j++) out[mesh.triangles[i][j]] += c, counts[mesh.triangles[i][j]]++;
	}

#pragma omp parallel for num_threads( threads )
	for (int i = 0; i<in.size(); i++) out[i] /= (Real)counts[i];
}


template< class Real, int Channels >
struct FlowData
{
	int gradientType;
	int flowType;

	FEM::RiemannianMesh< Real > mesh;
	std::vector< Point3D< Real > > vertices;
	std::vector< TriangleIndex > triangles;
	std::vector< Real > triangleArea;
	std::vector< Point< Real, Channels > > signals[2];
	EigenCholeskySolverLLt *sSolver;
	SparseMatrix< Real, int > sM, vfM;
	SparseMatrix< Real, int > sMass, sStiffness, vfMass, vfStiffness, triangleMass, grad;
	SparseMatrix< Real, int > biStiffness;

	Pointer(FEM::EdgeXForm< Real >) edges;
	std::vector< Point2D< Real > > tFlowField;


	Real getError(void)
	{
		Real error = (Real)0;
#pragma omp parallel for num_threads( Threads.value ) reduction ( + : error )
		for (int i = 0; i<triangles.size(); i++)
		{
			FEM::SamplePoint< Real > p[] = { FEM::SamplePoint< Real >(i, Point2D< Real >((Real)1. / 3, (Real)1. / 3)), FEM::SamplePoint< Real >(i, Point2D< Real >((Real)1. / 3, (Real)1. / 3)) };
			FEM::SamplePoint< Real > q[] = { p[0], p[1] };
			for (int s = 0; s<2; s++)

			mesh.flow(edges, (ConstPointer(Point2D< Real >))GetPointer(tFlowField), (Real)(s == 0 ? -1. : 1.), p[s], (Real)1e-2);
			error += Point< Real, Channels >::SquareNorm(Sample< Real, Point< Real, Channels > >(signals[0], triangles, q[0]) - Sample< Real, Point< Real, Channels > >(signals[1], triangles, p[1])) * triangleMass[i][0].Value;
			error += Point< Real, Channels >::SquareNorm(Sample< Real, Point< Real, Channels > >(signals[0], triangles, p[0]) - Sample< Real, Point< Real, Channels > >(signals[1], triangles, q[1])) * triangleMass[i][0].Value;
		}
		return error / (Real)2.;
	}
	Real getWhitneyError(void)
	{
		Real error = (Real)0;
#pragma omp parallel for num_threads( Threads.value ) reduction ( + : error )
		for (int i = 0; i<triangles.size(); i++)
		{
			FEM::SamplePoint< Real > p[] = { FEM::SamplePoint< Real >(i, Point2D< Real >((Real)1. / 3, (Real)1. / 3)), FEM::SamplePoint< Real >(i, Point2D< Real >((Real)1. / 3, (Real)1. / 3)) };
			FEM::SamplePoint< Real > q[] = { p[0], p[1] };
			for (int s = 0; s < 2; s++){
				mesh.whitneyFlow(edges, (ConstPointer(Real))GetPointer(eFlowField), (Real)(s == 0 ? -1. : 1.), p[s], (Real)1e-2);
			}
			error += Point< Real, Channels >::SquareNorm(Sample< Real, Point< Real, Channels > >(signals[0], triangles, q[0]) - Sample< Real, Point< Real, Channels > >(signals[1], triangles, p[1])) * triangleMass[i][0].Value;
			error += Point< Real, Channels >::SquareNorm(Sample< Real, Point< Real, Channels > >(signals[0], triangles, p[0]) - Sample< Real, Point< Real, Channels > >(signals[1], triangles, q[1])) * triangleMass[i][0].Value;
		}
		return error / (Real)2.;
	}
	Real getWhitneySymmetricError(void)
	{
		Real error = (Real)0;
#pragma omp parallel for num_threads( Threads.value ) reduction ( + : error )
		for (int i = 0; i<triangles.size(); i++)
		{
			FEM::SamplePoint< Real > p[] = { FEM::SamplePoint< Real >(i, Point2D< Real >((Real)1. / 3, (Real)1. / 3)), FEM::SamplePoint< Real >(i, Point2D< Real >((Real)1. / 3, (Real)1. / 3)) };
			for (int s = 0; s < 2; s++){
				mesh.whitneyFlow(edges, (ConstPointer(Real))GetPointer(eFlowField), (Real)(s == 0 ? -0.5 : 0.5), p[s], (Real)1e-2);
			}
			error += Point< Real, Channels >::SquareNorm(Sample< Real, Point< Real, Channels > >(signals[0], triangles, p[0]) - Sample< Real, Point< Real, Channels > >(signals[1], triangles, p[1])) * triangleMass[i][0].Value;
		}
		return error;
	}

	void smoothSignal(std::vector< Point< Real, Channels > > smoothed[2], Real smoothWeight)
	{
		for (int s = 0; s<2; s++) smoothed[s].resize(vertices.size());
#pragma omp parallel for num_threads( Threads.value )
		for (int i = 0; i<sM.rows; i++) for (int j = 0; j<sM.rowSizes[i]; j++) sM[i][j].Value = sMass[i][j].Value + sStiffness[i][j].Value * smoothWeight;
		sSolver->update(sM);
		Pointer(Real) x = AllocPointer< Real >(vertices.size());
		Pointer(Real) b = AllocPointer< Real >(vertices.size());
		for (int s = 0; s<2; s++) for (int c = 0; c<Channels; c++)
		{
#pragma omp parallel for num_threads( Threads.value )
			for (int i = 0; i<vertices.size(); i++) x[i] = signals[s][i][c];
			sMass.Multiply(x, b);
			sSolver->solve((ConstPointer(Real))b, x);
#pragma omp parallel for num_threads( Threads.value )
			for (int i = 0; i<vertices.size(); i++) smoothed[s][i][c] = x[i];
		}
		FreePointer(x);
		FreePointer(b);
	}

	void smoothSignal(const std::vector< Point< Real, Channels > > in, std::vector< Point< Real, Channels > > & out, Real smoothWeight)
	{
		out.resize(vertices.size());
#pragma omp parallel for num_threads( Threads.value )
		for (int i = 0; i<sM.rows; i++) for (int j = 0; j<sM.rowSizes[i]; j++) sM[i][j].Value = sMass[i][j].Value + sStiffness[i][j].Value * smoothWeight;
		sSolver->update(sM);
		Pointer(Real) x = AllocPointer< Real >(vertices.size());
		Pointer(Real) b = AllocPointer< Real >(vertices.size());
		for (int c = 0; c<Channels; c++)
		{
#pragma omp parallel for num_threads( Threads.value )
			for (int i = 0; i<vertices.size(); i++) x[i] = in[i][c];
			sMass.Multiply(x, b);
			sSolver->solve((ConstPointer(Real))b, x);
#pragma omp parallel for num_threads( Threads.value )
			for (int i = 0; i<vertices.size(); i++) out[i][c] = x[i];
		}
		FreePointer(x);
		FreePointer(b);
	}

	void smoothSignal(const std::vector< Point3D<Real> > & in, std::vector< Point3D<Real> > & out, Real smoothWeight)
	{
		out.resize(vertices.size());
#pragma omp parallel for num_threads( Threads.value )
		for (int i = 0; i<sM.rows; i++) for (int j = 0; j<sM.rowSizes[i]; j++) sM[i][j].Value = sMass[i][j].Value + sStiffness[i][j].Value * smoothWeight;
		sSolver->update(sM);
		Pointer(Real) x = AllocPointer< Real >(vertices.size());
		Pointer(Real) b = AllocPointer< Real >(vertices.size());
		for (int c = 0; c<3; c++)
		{
#pragma omp parallel for num_threads( Threads.value )
			for (int i = 0; i<vertices.size(); i++) x[i] = in[i][c];
			sMass.Multiply(x, b);
			sSolver->solve((ConstPointer(Real))b, x);
#pragma omp parallel for num_threads( Threads.value )
			for (int i = 0; i<vertices.size(); i++) out[i][c] = x[i];
		}
		FreePointer(x);
		FreePointer(b);
	}
};

template< class Real, int Channels >
void SetDataTerm(const std::vector<TriangleIndex> & triangles, const std::vector<Real> & triangleAreas, std::vector< Point< Real, Channels > > values[2], SparseMatrix<Real, int> & dataTerm, std::vector<Real> & rhs){
	int tCount = triangles.size();
	dataTerm.resize(2 * tCount);
	rhs.resize(2 * tCount, 0);

	for (int t = 0; t < tCount; t++){
		for (int j = 0; j < 2; j++) dataTerm.SetRowSize(2 * t + j, 2);
		for (int k = 0; k < 2; k++)for (int l = 0; l < 2; l++) dataTerm[2 * t + k][l] = MatrixEntry<Real, int>(2 * t + l, 0);

		Real tArea = triangleAreas[t];

		Point<Real, Channels> _value[2][3];
		for (int s = 0; s < 2; s++) for (int j = 0; j < 3; j++) _value[s][j] = values[s][triangles[t][j]];
		Point<Real, Channels> _difference[3];
		for (int j = 0; j < 3; j++) _difference[j] = _value[0][j] - _value[1][j];

		for (int c = 0; c < Channels; c++){

			Point3D<Real> f((_value[0][0][c] + _value[1][0][c]) / 2.0, (_value[0][1][c] + _value[1][1][c]) / 2.0, (_value[0][2][c] + _value[1][2][c]) / 2.0);
			Real meanDifference = (_difference[0][c] + _difference[1][c] + _difference[2][c]) / 3;
			Point2D<Real> gamma = Point2D< Real >(f[1] - f[0], f[2] - f[0]);
			
			for (int k = 0; k < 2; k++)for (int l = 0; l < 2; l++) dataTerm[2 * t + k][l].Value += gamma[k] * gamma[l] * tArea;
			for (int k = 0; k < 3; k++)rhs[2 * t + k] += gamma[k] * meanDifference* tArea;
		}
	}
}

template< class Real, int Channels >
void UpdateFlow (FlowData< Real, Channels >& flowData, VectorField<Real> * vf, Real scalarSmoothWeight, Real vectorSmoothWeight){
	std::vector< Point< Real, Channels > > smoothed[2];
	std::vector< Point< Real, Channels > > resampled[2];

	const std::vector< Point3D< Real > >& vertices = flowData.vertices;
	const std::vector< TriangleIndex >& triangles = flowData.triangles;
	ConstPointer(Point3D< Real >) _vertices = (ConstPointer(Point3D< Real >))GetPointer(vertices);

	Timer t;

#if SMOOTH_FIRST 
	if (scalarSmoothWeight) for (int s = 0; s < 2; s++) flowData.smoothSignal(flowData.signals[s], smoothed[s], scalarSmoothWeight);
	if (Verbose.set) printf("\t Signal Smoothing: %.4f(s)\n", t.elapsed());

	t.reset();
	for (int s = 0; s<2; s++) ResampleSignal(flowData.mesh, (ConstPointer(Point2D< Real >))GetPointer(flowData.tFlowField), (ConstPointer(FEM::EdgeXForm< Real >))flowData.edges, smoothed[s], resampled[s], (Real)(s == 0 ? -0.5 : 0.5), Threads.value);
	if (Verbose.set) printf("\t Signal advection : %.4f(s)\n", t.elapsed());
#else
	for (int s = 0; s<2; s++) ResampleSignal(flowData.mesh, (ConstPointer(Point2D< Real >))GetPointer(flowData.tFlowField), (ConstPointer(FEM::EdgeXForm< Real >))flowData.edges, flowData.signals[s], resampled[s], (Real)(s == 0 ? -0.5 : 0.5), Threads.value);
	if (Verbose.set) printf("\t Signal advection : %.4f(s)\n", t.elapsed());

	t.reset();
	if (scalarSmoothWeight) for (int s = 0; s < 2; s++) flowData.smoothSignal(resampled[s], smoothed[s], scalarSmoothWeight);
	if (Verbose.set) printf("\t Signal Smoothing: %.4f(s)\n", t.elapsed());
	for (int s = 0; s < 2; s++) resampled[s] = smoothed[s];
#endif

	if (0) {
		std::vector<Real> potential(resampled[0].size());
		for (int i = 0; i < resampled[0].size(); i++)potential[i] = resampled[0][i][0];
		Real gradientFieldSmoothness = vf->MeasureGradientFieldSmoothness(flowData.mesh, potential);
		printf("Gradient field smoothness %g \n", gradientFieldSmoothness);
	}

	if (Debug.set)
	{
		static int count = 0;
		char fileName[512];
		sprintf(fileName, "resampled.S.%d.ply", count), OutputMesh(fileName, vertices, resampled[0], triangles, PLY_BINARY_NATIVE);
		sprintf(fileName, "resampled.T.%d.ply", count), OutputMesh(fileName, vertices, resampled[1], triangles, PLY_BINARY_NATIVE);
		count++;
	}

	SparseMatrix< Real, int > _dataTerm;
	std::vector< Real > _rhs;
	t.reset();
	SetDataTerm(triangles, flowData.triangleArea, resampled, _dataTerm, _rhs);
	if (Verbose.set) printf("\t Set Data Term: %.4f(s)\n", t.elapsed());
	if(0) printf("Data term %g \n",_dataTerm.SquareNorm());
	vf->UpdateOpticalFlow(_dataTerm, _rhs, vectorSmoothWeight, flowData.tFlowField);
}

template< class Real >
struct InputGeometryData
{
	std::vector< Point3D< Real > > vertices[2];
	std::vector< Point3D< Real > > colors[2];

	template< int Channels > void flow(const FlowData< Real, Channels >& flowData, Real alpha, std::vector< Point3D< Real > > outputColors[2], int threads = 1)
	{
		for (int s = 0; s<2; s++)
		{
			Real length = (Real)(s == 0 ? -alpha : 1. - alpha);
			ResampleSignal(flowData.mesh, (ConstPointer(Point2D< Real >))GetPointer(flowData.tFlowField), (ConstPointer(FEM::EdgeXForm< Real >))flowData.edges, colors[s], outputColors[s], length, threads);
		}
	}
};
template< class Real >
struct InputTextureData
{
	int tWidth, tHeight;
	unsigned char* textures[2];
	FEM::SamplePoint< Real >* textureSource;
	std::vector< Point2D< Real > > triangleTextures;
	std::vector< Point3D< Real > > vertices;

	InputTextureData(void){ tWidth = tHeight = 0, textures[0] = textures[1] = NULL; }
	template< int Channels > void flow(const FlowData< Real, Channels >& flowData, Real alpha, Point3D< Real >* outputTextures[2], int threads)
	{
		for (int s = 0; s<2; s++)
		{
			Real length = (Real)(s == 0 ? -alpha : 1. - alpha);
#pragma omp parallel for num_threads( threads )
			for (int i = 0; i<tWidth*tHeight; i++) if (textureSource[i].tIdx != -1)
			{
				FEM::SamplePoint< Real > p = textureSource[i];
				flowData.mesh.flow(flowData.edges, GetPointer(flowData.tFlowField), length, p, (Real)1e-2);
				Point2D< Real > q = triangleTextures[p.tIdx * 3 + 0] * ((Real)1. - p.p[0] - p.p[1]) + triangleTextures[p.tIdx * 3 + 1] * p.p[0] + triangleTextures[p.tIdx * 3 + 2] * p.p[1];
				outputTextures[s][i] = Sample(textures[s], tWidth, tHeight, q, !Nearest.set);
			}
		}
	}

	template< int Channels > void flow(const FlowData< Real, Channels >& flowData, int frames, Point3D< Real >** outputTextures[2], int threads)
	{
		Pointer(FEM::SamplePoint< Real >) _textureSource = AllocPointer< FEM::SamplePoint< Real > >(tWidth * tHeight);
		for (int s = 0; s<2; s++)
		{
			memcpy(_textureSource, textureSource, sizeof(FEM::SamplePoint< Real >) * tWidth * tHeight);
			Real alpha = (Real)1. / (frames - 1);
			Real length = (Real)(s == 0 ? -alpha : alpha);
			for (int f = 0; f < frames; f++) for (int j = 0; j < tHeight; j++) for (int i = 0; i < tWidth; i++) outputTextures[s][f][j*tWidth + i] = Point3D< Real >((Real)(textures[s][3 * ((tHeight - j - 1)*tWidth + i)]), (Real)(textures[s][3 * ((tHeight - j - 1)*tWidth + i) + 1]), (Real)(textures[s][3 * ((tHeight - j - 1)*tWidth + i) + 2]));
			for (int f = 1; f < frames; f++)
			{
#pragma omp parallel for num_threads( threads )
				for (int i = 0; i < tWidth*tHeight; i++) if (_textureSource[i].tIdx != -1)
				{
					FEM::SamplePoint< Real >& p = _textureSource[i];
					flowData.mesh.flow(flowData.edges, GetPointer(flowData.tFlowField), length, p, (Real)(1e-2 * frames));
					Point2D< Real > q = triangleTextures[p.tIdx * 3 + 0] * ((Real)1. - p.p[0] - p.p[1]) + triangleTextures[p.tIdx * 3 + 1] * p.p[0] + triangleTextures[p.tIdx * 3 + 2] * p.p[1];
					outputTextures[s][f][i] = Sample(textures[s], tWidth, tHeight, q, !Nearest.set);
				}
			}
		}
		FreePointer(_textureSource);
	}
};

#include <GL/glew.h>
#include <GL/glut.h>
#include <Src/SurfaceVisualization.inl>

enum{
	READ_SOURCE,
	READ_TARGET,
	READ_COUNT,
};


enum{
	SIGNAL_INPUT,
	SIGNAL_BLEND,
	SIGNAL_COUNT,
};

template< class Real, int Channels >
struct WhitneyFlowViewer
{
	static SurfaceVisualization sv;
	static  FlowData< Real, Channels > flowData;
	static bool processTexture;
	static InputTextureData< Real > inputTextureData;
	static InputGeometryData< Real > inputGeometryData;
	static VectorField<Real> * vf;
	//VertexBased
	static std::vector<Point3D<Real>> inputSignal[2];
	static std::vector<Point3D<Real>> inputAdvectedSignal[2];

	//TextureBase
	static Point3D<Real> * inputTexture[2];
	static Point3D<Real> * inputAdvectedTexture[2];

	static void IterativeOptimization();

	static int signalSource;
	static int signalMode;

	static Real scalarSmoothWeight;
	static Real vectorFieldSmoothWeight;



	static int Init();
	static void Idle(void);
	static void KeyboardFunc(unsigned char key, int x, int y);
	static void SpecialFunc(int key, int x, int y);
	static void Display(void);
	static void Reshape(int w, int h);
	static void MouseFunc(int button, int state, int x, int y);
	static void MotionFunc(int x, int y);

	static void AdvanceLevelCallBack(Visualization* v, const char* prompt);

	static void UpdateSignalVisualization();
	static void UpdateTextureVisualization();
	static void ToggleSignalModeCallBack(Visualization* v, const char* prompt);
	static void ToggleSignalSourceCallBack(Visualization* v, const char* prompt);

	static void OutputResultCallBack(Visualization* v, const char* prompt);

	static void ScalarSmoothWeightCallBack(Visualization* v, const char* prompt);
	static void VectorFieldSmoothWeightCallBack(Visualization* v, const char* prompt);

	static void MeasureVectorFieldSmoothnessCallBack(Visualization* v, const char* prompt);
};

template< class Real, int Channels > SurfaceVisualization				WhitneyFlowViewer< Real, Channels>::sv;
template< class Real, int Channels > bool								WhitneyFlowViewer< Real, Channels>::processTexture;
template< class Real, int Channels > FlowData< Real, Channels >			WhitneyFlowViewer< Real, Channels>::flowData;
template< class Real, int Channels > InputTextureData< Real >			WhitneyFlowViewer< Real, Channels>::inputTextureData;
template< class Real, int Channels > InputGeometryData< Real >			WhitneyFlowViewer< Real, Channels>::inputGeometryData;
template< class Real, int Channels > int								WhitneyFlowViewer< Real, Channels>::signalSource = READ_SOURCE;
template< class Real, int Channels > int								WhitneyFlowViewer< Real, Channels>::signalMode = SIGNAL_INPUT;
template< class Real, int Channels > Real								WhitneyFlowViewer< Real, Channels>::scalarSmoothWeight = 3e-3f;
template< class Real, int Channels > Real								WhitneyFlowViewer< Real, Channels>::vectorFieldSmoothWeight = 1e-4f;

template< class Real, int Channels > std::vector<Point3D<Real>>			WhitneyFlowViewer< Real, Channels>::inputSignal[2];
template< class Real, int Channels > std::vector<Point3D<Real>>			WhitneyFlowViewer< Real, Channels>::inputAdvectedSignal[2];

template< class Real, int Channels > Point3D<Real> *					WhitneyFlowViewer< Real, Channels>::inputTexture[2];
template< class Real, int Channels > Point3D<Real> *					WhitneyFlowViewer< Real, Channels>::inputAdvectedTexture[2];
template< class Real, int Channels > VectorField<Real> *				WhitneyFlowViewer< Real, Channels>::vf;

template< class Real, int Channels > void WhitneyFlowViewer< Real, Channels>::Idle(void){ sv.Idle(); }
template< class Real, int Channels > void WhitneyFlowViewer< Real, Channels>::KeyboardFunc(unsigned char key, int x, int y){ sv.KeyboardFunc(key, x, y); }
template< class Real, int Channels > void WhitneyFlowViewer< Real, Channels>::SpecialFunc(int key, int x, int y){ sv.SpecialFunc(key, x, y); }
template< class Real, int Channels > void WhitneyFlowViewer< Real, Channels>::Display(void){ sv.Display(); }
template< class Real, int Channels > void WhitneyFlowViewer< Real, Channels>::Reshape(int w, int h){ sv.Reshape(w, h); }
template< class Real, int Channels > void WhitneyFlowViewer< Real, Channels>::MouseFunc(int button, int state, int x, int y){ sv.MouseFunc(button, state, x, y); }
template< class Real, int Channels > void WhitneyFlowViewer< Real, Channels>::MotionFunc(int x, int y){ sv.MotionFunc(x, y); }



template< class Real, int Channels >
void WhitneyFlowViewer< Real, Channels>::MeasureVectorFieldSmoothnessCallBack(Visualization* v, const char* prompt) {
	int tCount =  flowData.mesh.tCount;
#if 0
	std::vector<Real> smoothTimeVector(vf->coeffs.size());
	vf->smoothOperator.Multiply(GetPointer(vf->coeffs), GetPointer(smoothTimeVector));
	Real dot = Dot(GetPointer(vf->coeffs), GetPointer(smoothTimeVector), vf->coeffs.size());
	printf("Vector Field Smoothness Energy %f \n", dot);
	std::vector<Point2D<Real>> tVF(tCount);
	vf->GetTriangleVectorField(tVF);
	WriteVector(tVF, "vectorField.bin");
#else
	std::vector<Point2D<Real>> tVF;
	ReadVector(tVF, "vectorField.bin");
	for (int i = 0; i < tCount; i++) vf->coeffs[2 * i] = tVF[i][0], vf->coeffs[2 * i + 1] = tVF[i][1];

	std::vector<Real> smoothTimeVector(vf->coeffs.size());
	vf->smoothOperator.Multiply(GetPointer(vf->coeffs), GetPointer(smoothTimeVector), vf->coeffs.size());
	Real dot = Dot(GetPointer(vf->coeffs), GetPointer(smoothTimeVector), vf->coeffs.size());
	printf("Vector Field Smoothness Energy %f \n", dot);

	for (int t = 0; t < tCount; t++) sv.vectorField[t] = (sv.vertices[sv.triangles[t][1]] - sv.vertices[sv.triangles[t][0]])*tVF[t][0] + (sv.vertices[sv.triangles[t][2]] - sv.vertices[sv.triangles[t][0]])*tVF[t][1];
#endif
}

template< class Real, int Channels >
void WhitneyFlowViewer< Real, Channels>::ScalarSmoothWeightCallBack(Visualization* v, const char* prompt){
	glutSetCursor(GLUT_CURSOR_WAIT);
	scalarSmoothWeight = (Real)atof(prompt);
	sprintf(sv.info[2], "Smooth Weights(Scalar - Vector Field): %g - %g", scalarSmoothWeight, vectorFieldSmoothWeight);
	glutSetCursor(GLUT_CURSOR_INHERIT);
}


template< class Real, int Channels >
void WhitneyFlowViewer< Real, Channels>::VectorFieldSmoothWeightCallBack(Visualization* v, const char* prompt){
	glutSetCursor(GLUT_CURSOR_WAIT);
	vectorFieldSmoothWeight = (Real)atof(prompt);
	sprintf(sv.info[2], "Smooth Weights(Scalar - Vector Field): %g - %g", scalarSmoothWeight, vectorFieldSmoothWeight);
	glutSetCursor(GLUT_CURSOR_INHERIT);
}

template< class Real, int Channels >
int WhitneyFlowViewer< Real, Channels>::Init(){
	std::vector< Point3D< Real > >& vertices = flowData.vertices;
	std::vector< TriangleIndex >& triangles = flowData.triangles;
	int file_type;
	if (Mesh.set)
	{
		processTexture = true;
		sv.useTexture = true;
		std::vector< PlyVertex< float > > _vertices;
		std::vector< TriangleIndexWithData< Point2D< Real > > > _texturedTriangles;
		{
			std::vector< PlyTexturedFace< float > > faces;
			if (!PlyReadPolygons(Mesh.value, _vertices, faces, PlyVertex< float >::ReadProperties, NULL, PlyVertex< float >::ReadComponents, PlyTexturedFace< float >::ReadProperties, NULL, PlyTexturedFace< float >::ReadComponents, file_type)){
				printf("ERROR: Unable to read mesh %s \n", Mesh.value);
				return 0;
			}
		
			_texturedTriangles.resize(faces.size());
			for (int i = 0; i<faces.size(); i++)
			{
				if (faces[i].nr_vertices != 3 || faces[i].nr_uv_coordinates != 6) fprintf(stderr, "[ERROR] Bad face: %d %d\n", faces[i].nr_vertices, faces[i].nr_uv_coordinates), exit(0);
				for (int j = 0; j<3; j++) _texturedTriangles[i][j] = faces[i][j], _texturedTriangles[i].data[j] = Point2D< Real >(faces[i].texture(j));
			}
		}
		Point3D< Real > minCorner = _vertices[0].point;
		Point3D< Real > maxCorner = _vertices[0].point;
		for (int v = 0; v < _vertices.size(); v++) {
			for (int c = 0; c < 3; c++) {
				minCorner[c] = std::min<Real>(minCorner[c], _vertices[v].point[c]);
				maxCorner[c] = std::max<Real>(maxCorner[c], _vertices[v].point[c]);
			}
		}
		Real  diagonalLength = Point3D< Real >::Length(maxCorner - minCorner);
		SubdivideEdgeLength.value = SubdivideEdgeLength.value * diagonalLength;
		if (SubdivideEdgeLength.value>0) Subdivide(_texturedTriangles, _vertices, (Real)SubdivideEdgeLength.value);

		if(1) printf("Num vertices %d  \n", _vertices.size());

		vertices.resize(_vertices.size());
		for (int i = 0; i<vertices.size(); i++) vertices[i] = Point3D< Real >(_vertices[i]);
		triangles.resize(_texturedTriangles.size());
		inputTextureData.triangleTextures.resize(_texturedTriangles.size() * 3);
		for (int i = 0; i<_texturedTriangles.size(); i++)
		{
			triangles[i] = _texturedTriangles[i];
			for (int j = 0; j<3; j++) inputTextureData.triangleTextures[3 * i + j] = _texturedTriangles[i].data[j];
		}
		{
			int w, h;
			char* ext = GetFileExtension(In.values[0]);
			if (!strcasecmp(ext, "png")) inputTextureData.textures[0] = PNGReadColor(In.values[0], w, h);
			else{ fprintf(stderr, "[ERROR] Unrecognized image extension: %s\n", ext); return EXIT_FAILURE; }
			delete[] ext;
			inputTextureData.tWidth = w, inputTextureData.tHeight = h;
		}
		{
			int w, h;
			char* ext = GetFileExtension(In.values[1]);
			if (!strcasecmp(ext, "png")) inputTextureData.textures[1] = PNGReadColor(In.values[1], w, h);
			else{ fprintf(stderr, "[ERROR] Unrecognized image extension: %s\n", ext); return EXIT_FAILURE; }
			delete[] ext;
			if (inputTextureData.tWidth != w || inputTextureData.tHeight != h){ fprintf(stderr, "[ERROR] Texture resolutions don't match: %d x %d != %d x %d\n", inputTextureData.tWidth, inputTextureData.tHeight, w, h); return EXIT_FAILURE; }
		}


		for (int s = 0; s<2; s++)
		{
			std::vector< Point3D< Real > > signal;
			SampleTextureToVertices(_texturedTriangles, _vertices, inputTextureData.textures[s], inputTextureData.tWidth, inputTextureData.tHeight, signal, !Nearest.set);
			flowData.signals[s].resize(signal.size());
			for (int i = 0; i<signal.size(); i++) for (int c = 0; c<3; c++) flowData.signals[s][i][c] = signal[i][c];
		}
	}
	else
	{
		processTexture = false;
		sv.useTexture = false;
		std::vector< TriangleIndex > _triangles[2];
		std::vector< PlyColorVertex< float > > _vertices[2];
		PlyReadTriangles(In.values[0], _vertices[0], _triangles[0], PlyColorVertex< float >::ReadProperties, NULL, PlyColorVertex< float >::ReadComponents, file_type);
		PlyReadTriangles(In.values[1], _vertices[1], _triangles[1], PlyColorVertex< float >::ReadProperties, NULL, PlyColorVertex< float >::ReadComponents, file_type);
		if (_vertices[0].size() != _vertices[1].size()) { fprintf(stderr, "[ERROR] Vertex counts differ: %d != %d\n", _vertices[0].size(), _vertices[1].size()); return EXIT_FAILURE; }
		if (_triangles[0].size() != _triangles[1].size()) { fprintf(stderr, "[ERROR] Different number of triangles in meshes: %d != %d\n", _triangles[0].size(), _triangles[1].size()); return EXIT_FAILURE; }
		triangles = _triangles[0];
		vertices.resize(_vertices[0].size());
		for (int i = 0; i<vertices.size(); i++) vertices[i] = Point3D< Real >(_vertices[0][i]) * ((Real)0.5) + Point3D< Real >(_vertices[1][i]) * ((Real)0.5);

		for (int i = 0; i<triangles.size(); i++) for (int j = 0; j<3; j++) if (_triangles[0][i][j] != _triangles[1][i][j])
		{
			fprintf(stderr, "[ERROR] Triangle indices don't match: [%d,%d] %d != %d\n", i, j, _triangles[0][i][j], _triangles[1][i][j]);
			return EXIT_FAILURE;
		}
		for (int s = 0; s<2; s++)
		{
			inputGeometryData.colors[s].resize(_vertices[s].size());
			inputGeometryData.vertices[s].resize(_vertices[s].size());
			flowData.signals[s].resize(_vertices[s].size());
			for (int i = 0; i<_vertices[s].size(); i++) inputGeometryData.colors[s][i] = Point3D< Real >(_vertices[s][i].color), inputGeometryData.vertices[s][i] = Point3D< Real >(_vertices[s][i].point);
			for (int i = 0; i<_vertices[s].size(); i++) for (int c = 0; c<3; c++) flowData.signals[s][i][c] = inputGeometryData.colors[s][i][c];
		}
	}
	if (Verbose.set) printf("Vertices / Triangles: %d / %d\n", vertices.size(), triangles.size());

	flowData.gradientType = DivergenceFree.set ? FEM::RiemannianMesh< Real >::HAT_ROTATED_GRADIENT : FEM::RiemannianMesh< Real >::HAT_GRADIENT_AND_ROTATED_GRADIENT;

	ConstPointer(Point3D< Real >) _vertices = (ConstPointer(Point3D< Real >))GetPointer(vertices);

	// Set the cross-edge transformations
	{
		Timer t;
		flowData.mesh.triangles = GetPointer(triangles);
		flowData.mesh.tCount = triangles.size();
		flowData.mesh.setMetricFromEmbedding(_vertices);
		flowData.mesh.makeUnitArea();
		flowData.mesh.setInverseMetric();
		flowData.edges = flowData.mesh.getEdgeXForms();
		if (Verbose.set) printf("Got edge transforms: %.2f (s)\n", t.elapsed());
	}
	// Set the system matrices
	{
		Timer t;
		flowData.sMass = flowData.mesh.scalarMassMatrix(false);
		flowData.sStiffness = flowData.mesh.scalarStiffnessMatrix();
		flowData.sM.resize(flowData.sMass.rows);
		for (int i = 0; i<flowData.sMass.rows; i++)
		{
			flowData.sM.SetRowSize(i, flowData.sMass.rowSizes[i]);
			for (int j = 0; j<flowData.sMass.rowSizes[i]; j++) flowData.sM[i][j] = MatrixEntry< Real, int >(flowData.sMass[i][j].N, (Real)0);
		}
		flowData.sSolver = new EigenCholeskySolverLLt(flowData.sM, true);
		flowData.triangleMass = SparseMatrix< Real, int >::Identity((int)triangles.size());
		for (int i = 0; i<triangles.size(); i++) flowData.triangleMass[i][0].Value = (Real)flowData.mesh.area(i);
		
		flowData.triangleArea.resize((int)triangles.size());
		for (int i = 0; i<triangles.size(); i++)flowData.triangleArea[i] = (Real)flowData.mesh.area(i);
		if (Verbose.set) printf("Got system matrices: %.2f (s)\n", t.elapsed());
	}

	if (processTexture) inputTextureData.textureSource = GetTextureSource(flowData.mesh, (ConstPointer(FEM::EdgeXForm< Real >))flowData.edges, inputTextureData.triangleTextures, inputTextureData.tWidth, inputTextureData.tHeight, PadRadius.value);

	// Set the comparison values
	if (LogSpace.set) for (int s = 0; s<2; s++) for (int i = 0; i<vertices.size(); i++) for (int c = 0; c<3; c++) flowData.signals[s][i][c] = (Real)(log(std::max< Real >((Real)1., flowData.signals[s][i][c])) * 255. / log(255.));
	if (DoGWeight.value>0)
	{
		Timer t;
		Real weight = (Real)DoGSmooth.value;
		Pointer(Real) x = AllocPointer< Real >(vertices.size());
		Pointer(Real) b = AllocPointer< Real >(vertices.size());
		EigenCholeskySolverLLt solver(flowData.sMass + flowData.sStiffness * weight);
		// Smooth and subtract off
		for (int s = 0; s<2; s++) for (int c = 0; c<3; c++)
		{
			// \int [ f(x) - \int f(x) ]^2 = \int f^2(x) + [ \int f(x) ]^2 - 2 [ \int f(x) ]^2 = \int f^2(x) - [ \int f(x) ]^2

#pragma omp parallel for num_threads( Threads.value )
			for (int i = 0; i<vertices.size(); i++) x[i] = flowData.signals[s][i][c];
			flowData.sMass.Multiply(x, b);
			Real oldAvg = flowData.mesh.getIntegral(x);
			Real oldVar = Dot((ConstPointer(Real))x, (ConstPointer(Real))b, vertices.size()) - oldAvg * oldAvg;

			solver.solve((ConstPointer(Real))b, x);

#pragma omp parallel for num_threads( Threads.value )
			for (int i = 0; i<vertices.size(); i++) x[i] = flowData.signals[s][i][c] - x[i];
			flowData.sMass.Multiply(x, b);
			Real newAvg = flowData.mesh.getIntegral(x);
			Real newVar = Dot((ConstPointer(Real))x, (ConstPointer(Real))b, vertices.size()) - newAvg * newAvg;

			Real scale = (Real)sqrt(oldVar / newVar);
			if (Channels == 6)
#pragma omp parallel for num_threads( Threads.value )
			for (int i = 0; i<vertices.size(); i++) flowData.signals[s][i][c + 3] = (x[i] - newAvg) * scale + oldAvg;
			else if (Channels == 3)
			for (int i = 0; i<vertices.size(); i++) flowData.signals[s][i][c] = (x[i] - newAvg) * scale + oldAvg;
		}
		if (Channels == 6) for (int s = 0; s<2; s++) for (int c = 0; c<3; c++) for (int i = 0; i<vertices.size(); i++) flowData.signals[s][i][c] *= (Real)(1. - DoGWeight.value), flowData.signals[s][i][c + 3] *= (Real)DoGWeight.value;
		if (Verbose.set) printf("Set comparison values: %.2f (s)\n", t.elapsed());
	}
	{
		flowData.tFlowField.resize(triangles.size());
	}

	{
		if (VectorFieldMode.value == WHITNEY_VECTOR_FIELD) vf = new WhitneyVectorField<Real>();
		else if (VectorFieldMode.value == CONFORMAL_VECTOR_FIELD) vf = new ConformalVectorField<Real>();
		else if (VectorFieldMode.value == CONNECTION_VECTOR_FIELD) vf = new ConnectionVectorField<Real>(ConnectionMode.value);
		else{
			printf("ERROR: Unsupported vector field! \n");
			return 0;
		}
		vf->Init(flowData.mesh);
	}


	int vCount = flowData.vertices.size();
	sv.vertices.resize(vCount);
	for (int v = 0; v < vCount; v++) sv.vertices[v] = Point3D<float>(flowData.vertices[v]);
	sv.triangles = flowData.triangles;
	sv.colors.resize(vCount);
	sv.vectorField.resize(sv.triangles.size());
	if (sv.useTexture){
		sv.textureWidth = inputTextureData.tWidth;
		sv.textureHeight = inputTextureData.tHeight;
		sv.texture = new unsigned char[3 * sv.textureWidth * sv.textureHeight];
		sv.textureCoordinates.resize(inputTextureData.triangleTextures.size());
		for (int i = 0; i < inputTextureData.triangleTextures.size(); i++)sv.textureCoordinates[i] = Point2D<float>(inputTextureData.triangleTextures[i][0], inputTextureData.triangleTextures[i][1]);
		for (int s = 0; s < 2; s++){
			inputAdvectedTexture[s] = new Point3D<Real>[sv.textureWidth * sv.textureHeight];
			inputTexture[s] = new Point3D<Real>[sv.textureWidth * sv.textureHeight];
			for (int j = 0; j < sv.textureHeight; j++) for (int i = 0; i < sv.textureWidth; i++) inputTexture[s][j*sv.textureWidth + i] = inputAdvectedTexture[s][j*sv.textureWidth + i] = Point3D<Real>(inputTextureData.textures[s][3 * ((sv.textureHeight - j - 1)*sv.textureWidth + i)], inputTextureData.textures[s][3 * ((sv.textureHeight - j - 1)*sv.textureWidth + i) + 1], inputTextureData.textures[s][3 * ((sv.textureHeight - j - 1)*sv.textureWidth + i) + 2]);
		}
		for (int i = 0; i < sv.textureWidth * sv.textureHeight; i++)for (int c = 0; c < 3; c++)  sv.texture[3 * i + c] = static_cast<unsigned char>(inputTexture[signalSource][i][c]);
	}

	if(!sv.useTexture) for (int v = 0; v < vCount; v++) sv.colors[v] = Point3D<float>(inputGeometryData.colors[0][v]) / 255.f;
	if (!sv.useTexture) for (int s = 0; s<2; s++) inputSignal[s] = inputAdvectedSignal[s] = inputGeometryData.colors[s];


	sv.callBacks.push_back(Visualization::KeyboardCallBack(&sv, 't', "toggle signal source", ToggleSignalSourceCallBack));
	sv.callBacks.push_back(Visualization::KeyboardCallBack(&sv, 'n', "toggle signal mode", ToggleSignalModeCallBack));
	sv.callBacks.push_back(Visualization::KeyboardCallBack(&sv, 'a', "advance level", AdvanceLevelCallBack));

	sv.callBacks.push_back(Visualization::KeyboardCallBack(&sv, 'j', "scalar smooth weight", "Value", ScalarSmoothWeightCallBack));
	sv.callBacks.push_back(Visualization::KeyboardCallBack(&sv, 'J', "vector smooth weight", "Value", VectorFieldSmoothWeightCallBack));
	sv.callBacks.push_back(Visualization::KeyboardCallBack(&sv, 'o', "export result", "File name",OutputResultCallBack));


	sv.callBacks.push_back(Visualization::KeyboardCallBack(&sv, 'S', "vf smoothness", MeasureVectorFieldSmoothnessCallBack));

	sv.info.resize(6);
	for (int i = 0; i<sv.info.size(); i++) sv.info[i] = new char[512], sv.info[i][0] = 0;

	sprintf(sv.info[0], "Vertices / Triangles (%d,%d)", sv.vertices.size(), sv.triangles.size());
	sprintf(sv.info[1], "Signal: %s - %s", "Input", "Source");
	sprintf(sv.info[2], "Smooth Weights(Scalar - Vector Field): %g - %g", scalarSmoothWeight, vectorFieldSmoothWeight);

	return 1;
}

template< class Real, int Channels >
void WhitneyFlowViewer< Real, Channels>::UpdateTextureVisualization(){
	int vCount = sv.vertices.size();
	char signalModeName[256];
	char signalSourceName[256];
	if (signalMode == SIGNAL_INPUT){
		sprintf(signalModeName, "Input");
		sprintf(signalSourceName, signalSource == READ_SOURCE ? "Source" : "Target");
		for (int i = 0; i < sv.textureWidth * sv.textureHeight; i++)for (int c = 0; c < 3; c++)  sv.texture[3 * i + c] = static_cast<unsigned char>(inputTexture[signalSource][i][c]);
		
	}
	else if (signalMode == SIGNAL_BLEND){
		sprintf(signalModeName, "Output");
		sprintf(signalSourceName, signalSource == READ_SOURCE ? "Linear Blend" : "Optical Flow");
		if (signalSource == READ_SOURCE) for (int i = 0; i < sv.textureWidth * sv.textureHeight; i++) for (int c = 0; c < 3; c++)  sv.texture[3 * i + c] = static_cast<unsigned char>((inputTexture[0][i][c] + inputTexture[1][i][c]) / 2.0);
		if (signalSource == READ_TARGET) for (int i = 0; i < sv.textureWidth * sv.textureHeight; i++) for (int c = 0; c < 3; c++)  sv.texture[3 * i + c] = static_cast<unsigned char>((inputAdvectedTexture[0][i][c] + inputAdvectedTexture[1][i][c]) / 2.0);
	}
	
	sprintf(sv.info[1], "Signal: %s - %s", signalModeName, signalSourceName);
	sv.updateTextureBuffer(false);
}

template< class Real, int Channels >
void WhitneyFlowViewer< Real, Channels>::UpdateSignalVisualization(){
	int vCount = sv.vertices.size();
	char signalModeName[256];
	char signalSourceName[256];
	if (signalMode == SIGNAL_INPUT){
		sprintf(signalModeName, "Input");
		sprintf(signalSourceName, signalSource == READ_SOURCE ? "Source" : "Target");
		for (int v = 0; v < vCount; v++) sv.colors[v] = Point3D<float>(inputSignal[signalSource][v]) / 255.f;
	}
	else if (signalMode == SIGNAL_BLEND){
		sprintf(signalModeName, "Output");
		sprintf(signalSourceName, signalSource == READ_SOURCE ? "Linear Blend" : "Optical Flow");
		if (signalSource == READ_SOURCE)for (int v = 0; v < vCount; v++) sv.colors[v] = Point3D<float>((inputSignal[0][v] + inputSignal[1][v]) / Real(2.0)) / 255.f;
		if (signalSource == READ_TARGET)for (int v = 0; v < vCount; v++) sv.colors[v] = Point3D<float>((inputAdvectedSignal[0][v] + inputAdvectedSignal[1][v]) / Real(2.0)) / 255.f;
	}
	sprintf(sv.info[1], "Signal: %s - %s", signalModeName, signalSourceName);

	sv.updateMesh(false);
}

template< class Real, int Channels >
void WhitneyFlowViewer< Real, Channels>::OutputResultCallBack(Visualization* v, const char* prompt) {
	glutSetCursor(GLUT_CURSOR_WAIT);
	if (!sv.useTexture) {
		int vCount = flowData.vertices.size();
		std::vector<Point3D<Real>> outputColors(vCount);
		for (int v = 0; v < vCount; v++) outputColors[v] = Point3D<float>((inputAdvectedSignal[0][v] + inputAdvectedSignal[1][v]) / Real(2.0));
		OutputMesh(prompt,flowData.vertices, outputColors, flowData.triangles,PLY_ASCII);
	}
	else {
		std::vector<Point3D<Real>> outputColors(inputTextureData.tWidth *  inputTextureData.tHeight);
		for (int i = 0; i <  inputTextureData.tWidth*  inputTextureData.tHeight; i++) outputColors[i] = (inputAdvectedTexture[0][i] + inputAdvectedTexture[1][i]) / 2.0;
		OutputImage(prompt, outputColors, inputTextureData.tWidth, inputTextureData.tHeight, true);
	}
	glutSetCursor(GLUT_CURSOR_INHERIT);
}


template< class Real, int Channels >
void WhitneyFlowViewer< Real, Channels>::ToggleSignalSourceCallBack(Visualization* v, const char* prompt){
	glutSetCursor(GLUT_CURSOR_WAIT);
	signalSource = (signalSource + 1) % READ_COUNT;
	if(!sv.useTexture)UpdateSignalVisualization();
	else UpdateTextureVisualization();
	glutSetCursor(GLUT_CURSOR_INHERIT);
}

template< class Real, int Channels >
void WhitneyFlowViewer< Real, Channels>::ToggleSignalModeCallBack(Visualization* v, const char* prompt){
	glutSetCursor(GLUT_CURSOR_WAIT);
	signalMode = (signalMode + 1) % SIGNAL_COUNT;
	if (!sv.useTexture) UpdateSignalVisualization();
	else UpdateTextureVisualization();
	glutSetCursor(GLUT_CURSOR_INHERIT);
}

template< class Real, int Channels >
void WhitneyFlowViewer< Real, Channels>::AdvanceLevelCallBack(Visualization* v, const char* prompt){
	glutSetCursor(GLUT_CURSOR_WAIT);

	Timer t;
	UpdateFlow(flowData,vf, scalarSmoothWeight, vectorFieldSmoothWeight);
	if (Verbose.set) printf("Got flow: %.2f (s)\n", t.elapsed());

	int tCount = sv.triangles.size();
	for (int t = 0; t < tCount; t++) sv.vectorField[t] = (sv.vertices[sv.triangles[t][1]] - sv.vertices[sv.triangles[t][0]])*flowData.tFlowField[t][0] + (sv.vertices[sv.triangles[t][2]] - sv.vertices[sv.triangles[t][0]])*flowData.tFlowField[t][1];

	if(!sv.useTexture)inputGeometryData.flow(flowData, (Real)0.5, inputAdvectedSignal, Threads.value);
	else inputTextureData.flow(flowData, (Real)0.5, inputAdvectedTexture, Threads.value);

	if (0){
		std::vector< Point< Real, Channels > > resampled[2];
		for (int s = 0; s<2; s++) ResampleSignal(flowData.mesh, (ConstPointer(Point2D< Real >))GetPointer(flowData.tFlowField), (ConstPointer(FEM::EdgeXForm< Real >))flowData.edges, flowData.signals[s], resampled[s], (Real)(s == 0 ? -0.5 : 0.5), Threads.value);
		std::vector<Point< Real, Channels >> diff(resampled[0].size());
		for (int i = 0; i < resampled[0].size(); i++) diff[i] = resampled[1][i] - resampled[0][i];
		std::vector<Point< Real, Channels >> sMassDiff(resampled[0].size());
		flowData.sMass.Multiply(GetPointer(diff), GetPointer(sMassDiff));
		Real alignmentError = 0;
		for (int i = 0; i < diff.size(); i++) alignmentError += diff[i].InnerProduct(sMassDiff[i]);
		Real vfSmoothnessValue = vf->GetVectorFieldSmoothness();
		printf("Alignment Error %f. Vector Field Smoothness %f \n", alignmentError, vfSmoothnessValue);
	}

	scalarSmoothWeight *= ScalarWeightMultiplier.value;
	vectorFieldSmoothWeight = vectorFieldSmoothWeight *  VectorFieldWeightMultiplier.value > VectorFieldSmoothWeightThreshold.value ? vectorFieldSmoothWeight *  VectorFieldWeightMultiplier.value : vectorFieldSmoothWeight;
	sprintf(sv.info[2], "Smooth Weights(Scalar - Vector Field): %g - %g", scalarSmoothWeight, vectorFieldSmoothWeight);

	signalMode = SIGNAL_BLEND;
	if(!sv.useTexture)UpdateSignalVisualization();
	else UpdateTextureVisualization();
	glutSetCursor(GLUT_CURSOR_INHERIT);
}

template< class Real, int Channels >
void WhitneyFlowViewer< Real, Channels>::IterativeOptimization(){
	for (int i = 0; i < Levels.value; i++){
		Timer t;
		UpdateFlow(flowData,vf, scalarSmoothWeight, vectorFieldSmoothWeight);
		if (Verbose.set) printf("Got flow: %.2f (s)\n", t.elapsed());
		scalarSmoothWeight *= ScalarWeightMultiplier.value;
		vectorFieldSmoothWeight = vectorFieldSmoothWeight *  VectorFieldWeightMultiplier.value > VectorFieldSmoothWeightThreshold.value ? vectorFieldSmoothWeight *  VectorFieldWeightMultiplier.value : vectorFieldSmoothWeight;
	}
	if (processTexture) {
		inputTextureData.flow(flowData, (Real)0.5, inputAdvectedTexture, Threads.value);
		for (int i = 0; i < inputTextureData.tWidth*inputTextureData.tHeight; i++)inputAdvectedTexture[0][i] = (inputAdvectedTexture[0][i] + inputAdvectedTexture[1][i]) / 2.0;
		OutputImage(Out.value, inputAdvectedTexture[0], inputTextureData.tWidth, inputTextureData.tHeight, true);
	}
	else {
		inputGeometryData.flow(flowData, (Real)0.5, inputAdvectedSignal, Threads.value);
		int vCount = flowData.vertices.size();
		std::vector<Point3D<Real>> outputColors(vCount);
		for (int v = 0; v < vCount; v++) outputColors[v] = Point3D<float>((inputAdvectedSignal[0][v] + inputAdvectedSignal[1][v]) / Real(2.0));
		OutputMesh(Out.value, flowData.vertices, outputColors, flowData.triangles, PLY_ASCII);
	}
}

template< class Real, int Channels >
int _main(int argc, char* argv[]){

	SurfaceVisualization& sv = WhitneyFlowViewer< Real, Channels >::sv;
	WhitneyFlowViewer< Real, Channels >::scalarSmoothWeight = (Real)ScalarSmoothWeight.value;
	if (VectorFieldSmoothWeight.set) {
		WhitneyFlowViewer< Real, Channels >::vectorFieldSmoothWeight = (Real)VectorFieldSmoothWeight.value;
	}
	else {
		if (VectorFieldMode.value == WHITNEY_VECTOR_FIELD) WhitneyFlowViewer< Real, Channels >::vectorFieldSmoothWeight = 3e-6;
		if (VectorFieldMode.value == CONFORMAL_VECTOR_FIELD) WhitneyFlowViewer< Real, Channels >::vectorFieldSmoothWeight = 5e-7;
		if (VectorFieldMode.value == CONNECTION_VECTOR_FIELD) WhitneyFlowViewer< Real, Channels >::vectorFieldSmoothWeight = 1e4;
	}
	if (!WhitneyFlowViewer< Real, Channels >::Init()) return 0;
	if (Out.set){
		WhitneyFlowViewer< Real, Channels>::IterativeOptimization();
	}
	else{
		glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
		glutInitWindowSize(sv.screenWidth, sv.screenHeight);
		glutInit(&argc, argv);
		char windowName[1024];
		sprintf(windowName, "Optical Flow");
		glutCreateWindow(windowName);

		if (glewInit() != GLEW_OK) fprintf(stderr, "[ERROR] glewInit failed\n"), exit(0);
		glutIdleFunc(WhitneyFlowViewer< Real, Channels>::Idle);
		glutDisplayFunc(WhitneyFlowViewer< Real, Channels>::Display);
		glutReshapeFunc(WhitneyFlowViewer< Real, Channels>::Reshape);
		glutMouseFunc(WhitneyFlowViewer< Real, Channels>::MouseFunc);
		glutMotionFunc(WhitneyFlowViewer< Real, Channels>::MotionFunc);
		glutKeyboardFunc(WhitneyFlowViewer< Real, Channels>::KeyboardFunc);
		glutSpecialFunc(WhitneyFlowViewer< Real, Channels>::SpecialFunc);
		glutMainLoop();
	}
	return EXIT_SUCCESS;
}

int main(int argc, char* argv[])
{
	cmdLineParse(argc - 1, argv + 1, params);
	if (!In.set)
	{
		ShowUsage(argv[0]);
		return EXIT_FAILURE;
	}
	if (GSSearch.value <= 0)
	{
		fprintf(stderr, "[WARNING] Search range must be positive: %g<=0\n", GSSearch.value);
		GSSearch.set = false;
		GSSearch.value = 1.f;
	}
	if (GSSearch.value> 1.f) GSSearch.value = 1.f / GSSearch.value;
	if (GSSearch.value == 1.f) GSSearch.set = false;

	DoGWeight.value = std::min< float >(1.f, std::max< float >(0.f, DoGWeight.value));
	if (DoGWeight.value>0 && DoGWeight.value<1) return _main< double, 6 >(argc, argv);
	else                                        return _main< double, 3 >(argc, argv);
}
