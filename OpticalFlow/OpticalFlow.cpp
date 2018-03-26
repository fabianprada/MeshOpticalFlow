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
#undef ARRAY_DEBUG

#define BARICENTRIC_WHITNEY 0

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
#include "MeshFlow.inl"
#include <Eigen/Sparse>
#include <Eigen/QR>
#include <Eigen/LU>
#include <Eigen/SVD>

cmdLineParameter< char* > Mesh("mesh"), Out("out");
cmdLineParameterArray< char*, 2 > In("in");
cmdLineParameter< int > Levels("iterations", 10), Threads("threads", omp_get_max_threads()), PadRadius("pad", 2), Krylov("krylov", 1);
cmdLineParameter< float > ScalarSmoothWeight("sSmooth", 3e-3f), VectorFieldSmoothWeight("vfSmooth", 1e-4f), VectorFieldSmoothWeightThreshold("vfSThreshold", 1e-8f), SubdivideEdgeLength("eLength", 0.006f), DoGWeight("dogWeight", 1.f), DoGSmooth("dogSmooth", (float)1e-4), GSSearch("search", 1.f);
cmdLineParameter< float > ScalarWeightMultiplier("sMultiply", 0.25f), VectorFieldWeightMultiplier("vMultiply", 0.25f);
cmdLineReadable DivergenceFree("divFree");
cmdLineReadable Verbose("verbose"), ShowError("error"), Nearest("nearest"), Debug("debug"), LogSpace("log"), HalfWayFitting("hwFitting");
cmdLineReadable* params[] = { &Mesh, &In, &Out, &Levels, &Threads, &ScalarSmoothWeight, &VectorFieldSmoothWeight, &Verbose, &ShowError, &ScalarWeightMultiplier, &VectorFieldWeightMultiplier,&DoGWeight, &DoGSmooth, &SubdivideEdgeLength, &Nearest, &PadRadius, &Debug, &LogSpace, &GSSearch, &DivergenceFree, &Krylov, &HalfWayFitting, NULL };

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
	printf("\t[--%s <scalar weight multiplication factor>=%0.3f]\n", ScalarWeightMultiplier.name, ScalarWeightMultiplier.value);
	printf("\t[--%s <difference of Gaussians blending weight>=%0.3f]\n", DoGWeight.name, DoGWeight.value);
	printf("\t[--%s <difference of Gaussians smoothing weight>=%0.3f]\n", DoGSmooth.name, DoGSmooth.value);

	printf("Vector Field Parameters: \n");

	printf("\t[--%s <vector field smoothing weight>=%0.3f]\n", VectorFieldSmoothWeight.name, VectorFieldSmoothWeight.value);
	printf("\t[--%s <vector field weight multiplication factor>=%0.3f]\n", VectorFieldWeightMultiplier.name, VectorFieldWeightMultiplier.value);
	printf("\t[--%s <vector field weight threshold>=%0.3f]\n", VectorFieldSmoothWeightThreshold.name, VectorFieldSmoothWeightThreshold.value);

	printf("Auxiliar Parameters: \n");
	printf("\t[--%s <parallelization threads>=%d]\n", Threads.name, Threads.value);
	printf("\t[--%s <padding radius>=%d]\n", PadRadius.name, PadRadius.value);
	printf("\t[--%s <golden secition search range multiplier>=%0.3f]\n", GSSearch.name, GSSearch.value);
	printf("\t[--%s <Krylov Subspace Dimension>=%d]\n", Krylov.name, Krylov.value);
	printf("\t[--%s <use halfway fitting error>]\n", HalfWayFitting.name);
	printf("\t[--%s]\n", DivergenceFree.name);
	printf("\t[--%s]\n", LogSpace.name);
	printf("\t[--%s]\n", Nearest.name);
	printf("\t[--%s]\n", ShowError.name);
	printf("\t[--%s]\n", Verbose.name);
	printf("\t[--%s]\n", Debug.name);

}


// Code borrowed from: https://en.wikipedia.org/wiki/Golden_section_search
template< class Real, class Functor >
Real GoldenSectionSearch(Functor& f, Real a, Real b, Real c, Real fb, Real tolerance)
{
	const static Real PHI = (Real)((1. + sqrt(5.)) / 2.);
	const static Real RES_PHI = (Real)(2. - PHI);
	Real x, fx;

	if (c - b>b - a) x = b + RES_PHI * (c - b);
	else          x = b - RES_PHI * (b - a);
#if 1
	if (fabs(c - a)<tolerance) return (c + a) / 2;
#else
	if (fabs(c - a)<tolerance * (fabs(b) + fabs(x))) return (c + a) / 2;
#endif

	// compute the error
	fx = f(x);

	if (fx<fb)
	{
		if (c - b>b - a) return GoldenSectionSearch(f, b, x, c, fx, tolerance);
		else          return GoldenSectionSearch(f, a, x, b, fx, tolerance);
	}
	else
	{
		if (c - b>b - a) return GoldenSectionSearch(f, a, b, x, fb, tolerance);
		else          return GoldenSectionSearch(f, x, b, c, fb, tolerance);
	}
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
	std::vector< Point< Real, Channels > > signals[2];
	EigenCholeskySolverLLt *pSolver;
	EigenCholeskySolverLLt *sSolver;
	EigenCholeskySolverLDLt *vfSolver;
	SparseMatrix< Real, int > sM, vfM;
	SparseMatrix< Real, int > sMass, sStiffness, vfMass, vfStiffness, triangleMass, grad;
	SparseMatrix< Real, int > biStiffness;

	SparseMatrix< Real, int > gradientAndJGradient, gradientAndJGradient_transpose;
	SparseMatrix< Real, int > gradientAndJGradientRestriction, gradientAndJGradientProlongation;
	SparseMatrix< Real, int > restriction, prolongation;
	SparseMatrix< Real, int > lowMass, lowStiffness;
	SparseMatrix< Real, int > lowBiStiffness;
	SparseMatrix< Real, int > lowVFMass, lowVFStiffness;
	SparseMatrix< Real, int > triangleBasedVFRestriction, triangleBasedVFProlongation, triangleBasedVFMass, triangleBasedVFStiffness, triangleBasedHighVFMass, triangleBasedHighVFStiffness, triangleBasedWeights;

	SparseMatrix< Real, int > signedEdge;
	SparseMatrix< Real, int > signedEdge_transpose;
	SparseMatrix< Real, int > hodgeLaplacian;
	SparseMatrix< Real, int > whitneyProlongation;
	SparseMatrix< Real, int > whitneyRestriction;
	SparseMatrix< Real, int > whitneyMass;
	std::vector<int> reducedEdgeIndex;
	std::vector<int> expandedEdgeIndex;
	std::vector<bool> positiveOrientedEdge;
	EigenCholeskySolverLLt *whitneyVFSolver;

	Pointer(FEM::EdgeXForm< Real >) edges;
	std::vector< std::vector<Real> > partial_eFlowField;
	std::vector< Real > reduced_eFlowField;
	std::vector< Real > eFlowField;
	std::vector< Real > vFlowField, dVFlowField;
	std::vector< Point2D< Real > > tFlowField, dTFlowField;
	std::vector< Point2D< Real > > old_tFlowField;


	void combineFlows(Real x)
	{
#pragma omp parallel for num_threads( Threads.value )
		for (int i = 0; i<vFlowField.size(); i++) vFlowField[i] += dVFlowField[i] * x;
		mesh.setGradient((ConstPointer(Real))GetPointer(vFlowField), gradientType, GetPointer(tFlowField));
	}
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
	Real operator()(Real x) const
	{
		std::vector< Real > _vFlowField(vFlowField.size());
		std::vector< Point2D< Real > > _tFlowField(tFlowField.size());
#pragma omp parallel for num_threads( Threads.value )
		for (int i = 0; i<vFlowField.size(); i++) _vFlowField[i] = vFlowField[i] + dVFlowField[i] * x;
		mesh.setGradient((ConstPointer(Real))GetPointer(_vFlowField), gradientType, GetPointer(_tFlowField));
		Real error = (Real)0;
#pragma omp parallel for num_threads( Threads.value ) reduction ( + : error )
		for (int i = 0; i<triangles.size(); i++)
		{
			FEM::SamplePoint< Real > p[] = { FEM::SamplePoint< Real >(i, Point2D< Real >((Real)1. / 3, (Real)1. / 3)), FEM::SamplePoint< Real >(i, Point2D< Real >((Real)1. / 3, (Real)1. / 3)) };
			FEM::SamplePoint< Real > q[] = { p[0], p[1] };
			for (int s = 0; s<2; s++) mesh.flow(edges, (ConstPointer(Point2D< Real >))GetPointer(_tFlowField), (Real)(s == 0 ? -1. : 1.), p[s], (Real)1e-2);
			error += Point< Real, Channels >::SquareNorm(Sample< Real, Point< Real, Channels > >(signals[0], triangles, q[0]) - Sample< Real, Point< Real, Channels > >(signals[1], triangles, p[1])) * triangleMass[i][0].Value;
			error += Point< Real, Channels >::SquareNorm(Sample< Real, Point< Real, Channels > >(signals[0], triangles, p[0]) - Sample< Real, Point< Real, Channels > >(signals[1], triangles, q[1])) * triangleMass[i][0].Value;
		}
		return error / (Real)2.;
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

	void projectSignal(const std::vector< Point3D<Real> > & in, std::vector< Point3D<Real> > & out, Real smoothWeight){
		int vCount = in.size();
		out.resize(vCount);
		Timer t;
		pSolver = new EigenCholeskySolverLLt(lowMass + lowStiffness*smoothWeight);
		for (int c = 0; c<3; c++)
		{
			Pointer(Real) color = AllocPointer< Real >(vCount);
			Pointer(Real) colorMass = AllocPointer< Real >(vCount);
			for (int i = 0; i<vCount; i++) color[i] = in[i][c];

			// Get the high-resolution mass
			sMass.Multiply(color, colorMass);

			// Restrict to the low-resolution mass
			{
				Pointer(Real) temp = AllocPointer< Real >(restriction.rows);
				restriction.Multiply(colorMass, temp);
				FreePointer(colorMass);
				colorMass = temp;
			}

			// Solve the low-resolution system
			{
				Pointer(Real) temp = AllocPointer< Real >(lowMass.rows);
				pSolver->solve((ConstPointer(Real))colorMass, temp);
				FreePointer(color);
				color = temp;
			}
			if (Verbose.set)
			{
				Real n = 0, e = 0;
#pragma omp parallel for reduction( + : e , n )
				for (int i = 0; i<lowStiffness.rows; i++) for (int j = 0; j<lowStiffness.rowSizes[i]; j++)
					e += lowStiffness[i][j].Value * color[i] * color[lowStiffness[i][j].N], n += lowMass[i][j].Value * color[i] * color[lowMass[i][j].N];
				printf("\t Projection Mass/Stiffness[%d]: %g/%g\n", c, n, e);
			}

			// Prolong to the high-resolution system
			{
				Pointer(Real) temp = AllocPointer< Real >(prolongation.rows);
				prolongation.Multiply(color, temp);
				FreePointer(color);
				color = temp;
			}

			for (int i = 0; i < vCount; i++) out[i][c] = color[i];

			if (Verbose.set)
			{
				Real d = 0, N = 0, E = 0;
#pragma omp parallel for reduction( + : d, N, E)
				for (int i = 0; i<sMass.rows; i++) for (int j = 0; j<sMass.rowSizes[i]; j++)
					d += sMass[i][j].Value * (in[i][c] - out[i][c]) * (in[sMass[i][j].N][c] - out[sMass[i][j].N][c]), N += sMass[i][j].Value * in[i][c] * in[sMass[i][j].N][c], E += sStiffness[i][j].Value *  in[i][c] * in[sStiffness[i][j].N][c];
				printf("\t Initial Mass/Stiffness[%d] : %g/%g\n", c, N, E);
				printf("\t Mass Difference[%d] : %g\n", c, d);
			}

			FreePointer(color);
			FreePointer(colorMass);
		}
		if (Verbose.set) printf("Project signal: %.2f (s)\n", t.elapsed());
	}

	void projectSignal(const std::vector< Point<Real, Channels> > & in, std::vector< Point<Real, Channels> > & out, Real smoothWeight){
		int vCount = in.size();
		out.resize(vCount);
		Timer t;
		pSolver = new EigenCholeskySolverLLt(lowMass + lowStiffness*smoothWeight);
		for (int c = 0; c<Channels; c++)
		{
			Pointer(Real) color = AllocPointer< Real >(vCount);
			Pointer(Real) colorMass = AllocPointer< Real >(vCount);
			for (int i = 0; i<vCount; i++) color[i] = in[i][c];

			// Get the high-resolution mass
			sMass.Multiply(color, colorMass);

			// Restrict to the low-resolution mass
			{
				Pointer(Real) temp = AllocPointer< Real >(restriction.rows);
				restriction.Multiply(colorMass, temp);
				FreePointer(colorMass);
				colorMass = temp;
			}

			// Solve the low-resolution system
			{
				Pointer(Real) temp = AllocPointer< Real >(lowMass.rows);
				pSolver->solve((ConstPointer(Real))colorMass, temp);
				FreePointer(color);
				color = temp;
			}

			if (Verbose.set)
			{
				Real n = 0, e = 0;
#pragma omp parallel for reduction( + : e , n )
				for (int i = 0; i<lowStiffness.rows; i++) for (int j = 0; j<lowStiffness.rowSizes[i]; j++)
					e += lowStiffness[i][j].Value * color[i] * color[lowStiffness[i][j].N], n += lowMass[i][j].Value * color[i] * color[lowMass[i][j].N];
				printf("\t Projection Mass/Stiffness[%d]: %g/%g\n", c, n, e);
			}

			// Prolong to the high-resolution system
			{
				Pointer(Real) temp = AllocPointer< Real >(prolongation.rows);
				prolongation.Multiply(color, temp);
				FreePointer(color);
				color = temp;
			}

			for (int i = 0; i < vCount; i++) out[i][c] = color[i];

			if (Verbose.set)
			{
				Real d = 0, N = 0, E = 0;
#pragma omp parallel for reduction( + : d, N, E)
				for (int i = 0; i<sMass.rows; i++) for (int j = 0; j<sMass.rowSizes[i]; j++)
					d += sMass[i][j].Value * (in[i][c] - out[i][c]) * (in[sMass[i][j].N][c] - out[sMass[i][j].N][c]), N += sMass[i][j].Value * in[i][c] * in[sMass[i][j].N][c], E += sStiffness[i][j].Value *  in[i][c] * in[sStiffness[i][j].N][c];
				printf("\t Initial Mass/Stiffness[%d] : %g/%g\n", c, N, E);
				printf("\t Mass Difference[%d] : %g\n", c, d);
			}

			FreePointer(color);
			FreePointer(colorMass);
		}
		if (Verbose.set) printf("Project signal: %.2f (s)\n", t.elapsed());
	}

	void projectSignal(const std::vector< Real > & in, std::vector< Real > & out, Real smoothWeight){
		int vCount = in.size();
		out.resize(vCount);
		Timer t;
		pSolver = new EigenCholeskySolverLLt(lowMass + lowStiffness*smoothWeight);
		Pointer(Real) color = AllocPointer< Real >(vCount);
		Pointer(Real) colorMass = AllocPointer< Real >(vCount);
		for (int i = 0; i<vCount; i++) color[i] = in[i];

		// Get the high-resolution mass
		sMass.Multiply(color, colorMass);

		// Restrict to the low-resolution mass
		{
			Pointer(Real) temp = AllocPointer< Real >(restriction.rows);
			restriction.Multiply(colorMass, temp);
			FreePointer(colorMass);
			colorMass = temp;
		}

		// Solve the low-resolution system
		{
			Pointer(Real) temp = AllocPointer< Real >(lowMass.rows);
			pSolver->solve((ConstPointer(Real))colorMass, temp);
			FreePointer(color);
			color = temp;
		}

		if (Verbose.set)
		{
			Real n = 0, e = 0;
#pragma omp parallel for reduction( + : e , n )
			for (int i = 0; i<lowStiffness.rows; i++) for (int j = 0; j<lowStiffness.rowSizes[i]; j++)
				e += lowStiffness[i][j].Value * color[i] * color[lowStiffness[i][j].N], n += lowMass[i][j].Value * color[i] * color[lowMass[i][j].N];
			printf("\t Projection Mass/Stiffness: %g/%g\n", n, e);
		}

		// Prolong to the high-resolution system
		{
			Pointer(Real) temp = AllocPointer< Real >(prolongation.rows);
			prolongation.Multiply(color, temp);
			FreePointer(color);
			color = temp;
		}

		for (int i = 0; i < vCount; i++) out[i] = color[i];

		if (Verbose.set)
		{
			Real d = 0, N = 0, E = 0;
#pragma omp parallel for reduction( + : d, N, E)
			for (int i = 0; i<sMass.rows; i++) for (int j = 0; j<sMass.rowSizes[i]; j++)
				d += sMass[i][j].Value * (in[i] - out[i]) * (in[sMass[i][j].N] - out[sMass[i][j].N]), N += sMass[i][j].Value * in[i] * in[sMass[i][j].N], E += sStiffness[i][j].Value *  in[i] * in[sStiffness[i][j].N];
			printf("\t Initial Mass/Stiffness : %g/%g\n", N, E);
			printf("\t Mass Difference : %g\n", d);
		}

		FreePointer(color);
		FreePointer(colorMass);
		if (Verbose.set) printf("Project signal: %.2f (s)\n", t.elapsed());
	}

	void projectVectorField(const std::vector< Real > & in, std::vector< Real > & out, Real smoothWeight){
		int vCount = in.size();
		out.resize(vCount);
		Timer t;
		EigenCholeskySolverLDLt vfPSolver(lowVFMass + lowVFStiffness*smoothWeight);
		Pointer(Real) color = AllocPointer< Real >(vCount);
		Pointer(Real) colorMass = AllocPointer< Real >(vCount);
		for (int i = 0; i<vCount; i++) color[i] = in[i];

		Real inDC = 0;
#pragma omp parallel for reduction( + : inDC)
		for (int i = 0; i < vCount; i++) inDC += in[i];

		// Get the high-resolution mass
		vfMass.Multiply(color, colorMass);

		// Restrict to the low-resolution mass
		{
			Pointer(Real) temp = AllocPointer< Real >(gradientAndJGradientRestriction.rows);
			gradientAndJGradientRestriction.Multiply(colorMass, temp);
			FreePointer(colorMass);
			colorMass = temp;
		}

		// Solve the low-resolution system
		{
			Pointer(Real) temp = AllocPointer< Real >(lowVFMass.rows);
			vfPSolver.solve((ConstPointer(Real))colorMass, temp);
			FreePointer(color);
			color = temp;
		}

		if (Verbose.set)
		{
			Real e = 0;
#pragma omp parallel for reduction( + : e)
			for (int i = 0; i < lowVFStiffness.rows; i++) for (int j = 0; j < lowVFStiffness.rowSizes[i]; j++)
				e += lowVFStiffness[i][j].Value * color[i] * color[lowVFStiffness[i][j].N];

			Real n = 0;
#pragma omp parallel for reduction( + : n)
			for (int i = 0; i < lowVFMass.rows; i++) for (int j = 0; j < lowVFMass.rowSizes[i]; j++)
				n += lowVFMass[i][j].Value * color[i] * color[lowVFMass[i][j].N];
			printf("\t Projection Mass/Stiffness: %g/%g\n", n, e);
		}

		// Prolong to the high-resolution system
		{
			Pointer(Real) temp = AllocPointer< Real >(gradientAndJGradientProlongation.rows);
			gradientAndJGradientProlongation.Multiply(color, temp);
			FreePointer(color);
			color = temp;
		}

		for (int i = 0; i < vCount; i++) out[i] = color[i];

		Real outDC = 0;
#pragma omp parallel for reduction( + : outDC)
		for (int i = 0; i < vCount; i++) outDC += out[i];

		Real offsetDC = (inDC - outDC) / Real(vCount);
#pragma omp parallel for 
		for (int i = 0; i < vCount; i++) out[i] += offsetDC;

		if (Verbose.set)
		{
			Real d = 0, N = 0;
#pragma omp parallel for reduction( + : d, N)
			for (int i = 0; i<vfMass.rows; i++) for (int j = 0; j<vfMass.rowSizes[i]; j++)
				d += vfMass[i][j].Value * (in[i] - out[i]) * (in[vfMass[i][j].N] - out[vfMass[i][j].N]), N += vfMass[i][j].Value * in[i] * in[vfMass[i][j].N];

			Real E = 0;
#pragma omp parallel for reduction( + : E)
			for (int i = 0; i<vfStiffness.rows; i++) for (int j = 0; j<vfStiffness.rowSizes[i]; j++)
				E += vfStiffness[i][j].Value *  in[i] * in[vfStiffness[i][j].N];


			printf("\t Initial Mass/Stiffness : %g/%g\n", N, E);
			printf("\t Mass Difference : %g\n", d);
		}

		FreePointer(color);
		FreePointer(colorMass);
		if (Verbose.set) printf("Project vector field: %.2f (s)\n", t.elapsed());
	}
};

template< class Real >
void projectVectorField(const std::vector<Point2D<Real>> & inVF, std::vector<Point2D<Real>> & outVF, const SparseMatrix<Real, int> & highMass, const SparseMatrix<Real, int> & lowMass, const SparseMatrix<Real, int> & lowStiffness, Real smootWeight, const SparseMatrix<Real, int> & restriction, const SparseMatrix<Real, int> & prolongation){

	int tCount = inVF.size();

	std::vector<Real> targetVectorField(2 * tCount);
	for (int i = 0; i < tCount; i++) targetVectorField[2 * i] = inVF[i][0], targetVectorField[2 * i + 1] = inVF[i][1];
	std::vector<Real> targetVectorFieldMass(2 * tCount);
	highMass.Multiply(GetPointer(targetVectorField), GetPointer(targetVectorFieldMass));
	Real tmass = Dot(GetPointer(targetVectorField), GetPointer(targetVectorFieldMass), 2 * tCount);

	std::vector<Real> targetVectorFieldRHS(restriction.rows);
	restriction.Multiply(GetPointer(targetVectorFieldMass), GetPointer(targetVectorFieldRHS));

	EigenCholeskySolverLLt solver(lowMass + lowStiffness*smootWeight);
	std::vector<Real> vectorFieldProjection(restriction.rows);
	solver.solve(GetPointer(targetVectorFieldRHS), GetPointer(vectorFieldProjection));

	Real pmass = 0;
#pragma omp parallel for reduction( + :pmass)
	for (int i = 0; i < lowMass.rows; i++) for (int j = 0; j < lowMass.rowSizes[i]; j++)
		pmass += lowMass[i][j].Value * vectorFieldProjection[i] * vectorFieldProjection[lowMass[i][j].N];

	std::vector<Real> vectorFieldProlongation(2 * tCount);
	prolongation.Multiply(GetPointer(vectorFieldProjection), GetPointer(vectorFieldProlongation));

	Real dmass = 0;
#pragma omp parallel for reduction( + :dmass)
	for (int i = 0; i < highMass.rows; i++) for (int j = 0; j < highMass.rowSizes[i]; j++)
		dmass += highMass[i][j].Value * (vectorFieldProlongation[i] - targetVectorField[i]) * (vectorFieldProlongation[highMass[i][j].N] - targetVectorField[highMass[i][j].N]);

	if (Verbose.set) printf("Target/Project/Difference Mass  %g/%g/%g \n", tmass, pmass, dmass);

	outVF.resize(tCount);
	for (int t = 0; t < tCount; t++) outVF[t][0] = vectorFieldProlongation[2 * t], outVF[t][1] = vectorFieldProlongation[2 * t + 1];
}

#if BARICENTRIC_WHITNEY
#if 1
template< class Real, int Channels >
void SetBaricentricWhitneyLinearSystem(const FlowData< Real, Channels >& flowData, std::vector< Point< Real, Channels > > values[2], SparseMatrix<Real, int> & linearSystem, std::vector<Real> & rhs){
	printf("Baricentric Whitney Basis \n");

	std::vector<TriangleIndex> triangles = flowData.triangles;
	int tCount = triangles.size();

	SparseMatrix<Real, int> tMatrix;
	tMatrix.resize(2 * tCount);
	std::vector<Real> tVector(2 * tCount, 0);

	for (int t = 0; t < tCount; t++){
		for (int j = 0; j < 2; j++) tMatrix.SetRowSize(2 * t + j, 2);
		for (int k = 0; k < 2; k++)for (int l = 0; l < 2; l++) tMatrix[2 * t + k][l] = MatrixEntry<Real, int>(2 * t + l, 0);

		Real tArea = flowData.mesh.area(t);

		Point<Real, Channels> _value[2][3];
		for (int s = 0; s < 2; s++) for (int j = 0; j < 3; j++) _value[s][j] = values[s][triangles[t][j]];
		Point<Real, Channels> _difference[3];
		for (int j = 0; j < 3; j++) _difference[j] = _value[0][j] - _value[1][j];

		for (int s = 0; s < 2; s++){
			for (int c = 0; c < Channels; c++){
				Point3D<Real> f(_value[s][0][c], _value[s][1][c], _value[s][2][c]);
				Point2D<Real> gamma = Point2D< Real >(f[1] - f[0], f[2] - f[0]);
				Real meanDifference = (_difference[0][c] + _difference[1][c] + _difference[2][c]) / 3;
				for (int k = 0; k < 2; k++)for (int l = 0; l < 2; l++) tMatrix[2 * t + k][l].Value += gamma[k] * gamma[l] * tArea;
				for (int k = 0; k < 3; k++)tVector[2 * t + k] += gamma[k] * meanDifference* tArea;
			}
		}
	}
	linearSystem = flowData.whitneyRestriction * tMatrix * flowData.whitneyProlongation;
	rhs.resize(linearSystem.rows);
	flowData.whitneyRestriction.Multiply(GetPointer(tVector), GetPointer(rhs));
}
#else
template< class Real, int Channels >
void SetBaricentricWhitneyLinearSystem(const FlowData< Real, Channels >& flowData, std::vector< Point< Real, Channels > > values[2], SparseMatrix<Real, int> & linearSystem, std::vector<Real> & rhs){
	printf("Baricentric Whitney Basis \n");
	
	std::vector<TriangleIndex> triangles = flowData.triangles;
	int tCount = triangles.size();

	Point2D< Real > grad[3];
	grad[0][0] = -1., grad[0][1] = -1.;
	grad[1][0] = 1., grad[1][1] = 0.;
	grad[2][0] = 0., grad[2][1] = 1.;

	SparseMatrix<Real, int> tMatrix;
	tMatrix.resize(3 * tCount);
	std::vector<Real> tVector(3 * tCount, 0);

	for (int t = 0; t < tCount; t++){
		for (int j = 0; j < 3; j++) tMatrix.SetRowSize(3 * t + j, 3);
		for (int k = 0; k < 3; k++)for (int l = 0; l < 3; l++) tMatrix[3 * t + k][l] = MatrixEntry<Real, int>(3 * t + l, 0);

		SquareMatrix< Real, 3 > GMatrix;
		if (!flowData.mesh.g[t].determinant()){
			fprintf(stderr, "[WARNING] Vanishing metric tensor determinant\n");
			GMatrix *= 0;
		}
		SquareMatrix< Real, 2 > iTensor = flowData.mesh.g[t].inverse();
		for (int k = 0; k < 3; k++)for (int l = 0; l < 3; l++)GMatrix(k, l) = Point2D< Real >::Dot(grad[k], iTensor * grad[l]);

		Real tArea = flowData.mesh.area(t);

		Point<Real, Channels> _value[2][3];
		for (int s = 0; s < 2; s++) for (int j = 0; j < 3; j++) _value[s][j] = values[s][triangles[t][j]];
		Point<Real, Channels> _difference[3];
		for (int j = 0; j < 3; j++) _difference[j] = _value[0][j] - _value[1][j];

		for (int s = 0; s < 2; s++){
			for (int c = 0; c < Channels; c++){
				Point3D<Real> f(_value[s][0][c], _value[s][1][c], _value[s][2][c]);
				Real meanDifference = (_difference[0][c] + _difference[1][c] + _difference[2][c]) / 3;
				Point3D<Real> Gf = GMatrix*f;
				Point3D<Real> Gf_diff;
				for (int k = 0; k < 3; k++) Gf_diff[k] = (Gf[(k + 2) % 3] - Gf[(k + 1) % 3])/3;
				for (int k = 0; k < 3; k++)for (int l = 0; l < 3; l++) tMatrix[3 * t + k][l].Value += Gf_diff[k] * Gf_diff[l] * tArea;
				for (int k = 0; k < 3; k++)tVector[3 * t + k] += Gf_diff[k] * meanDifference* tArea;
			}
		}
	}

	linearSystem = flowData.signedEdge_transpose * tMatrix * flowData.signedEdge;
	rhs.resize(linearSystem.rows);
	flowData.signedEdge_transpose.Multiply(GetPointer(tVector), GetPointer(rhs));
}
#endif
#else
template< class Real, int Channels >
void SetFullWhitneyLinearSystem(const FlowData< Real, Channels >& flowData, std::vector< Point< Real, Channels > > values[2], SparseMatrix<Real, int> & linearSystem, std::vector<Real> & rhs){

	std::vector<TriangleIndex> triangles = flowData.triangles;
	int tCount = triangles.size();

	Point2D< Real > grad[3];
	grad[0][0] = -1., grad[0][1] = -1.;
	grad[1][0] = 1., grad[1][1] = 0.;
	grad[2][0] = 0., grad[2][1] = 1.;

	SparseMatrix<Real, int> tMatrix;
	tMatrix.resize(3 * tCount);
	std::vector<Real> tVector(3 * tCount, 0);

	for (int t = 0; t < tCount; t++){
		for (int j = 0; j < 3; j++) tMatrix.SetRowSize(3 * t + j, 3);
		for (int k = 0; k < 3; k++)for (int l = 0; l < 3; l++) tMatrix[3 * t + k][l] = MatrixEntry<Real, int>(3 * t + l, 0);

		SquareMatrix< Real, 3 > GMatrix;
		if (!flowData.mesh.g[t].determinant()){
			fprintf(stderr, "[WARNING] Vanishing metric tensor determinant\n");
			GMatrix *= 0;
		}
		SquareMatrix< Real, 2 > iTensor = flowData.mesh.g[t].inverse();
		for (int k = 0; k < 3; k++)for (int l = 0; l < 3; l++)GMatrix(k, l) = Point2D< Real >::Dot(grad[k], iTensor * grad[l]);

		SquareMatrix< Real, 3 > tMass = FEM::RightTriangle< Real >::GetScalarMassMatrix(flowData.mesh.g[t], false);

		Point<Real, Channels> _value[2][3];
		for (int s = 0; s < 2; s++) for (int j = 0; j < 3; j++) _value[s][j] = values[s][triangles[t][j]];
		Point<Real, Channels> _difference[3];
		for (int j = 0; j < 3; j++) _difference[j] = _value[0][j] - _value[1][j];

		if (!HalfWayFitting.set){
			for (int s = 0; s < 2; s++){
				for (int c = 0; c < Channels; c++){
					Point3D<Real> f(_value[s][0][c], _value[s][1][c], _value[s][2][c]);
					Point3D<Real> d(_difference[0][c], _difference[1][c], _difference[2][c]);

					Point3D<Real> Gf = GMatrix*f;
					SquareMatrix< Real, 3 > GfGfT;
					for (int k = 0; k < 3; k++)for (int l = 0; l < 3; l++) GfGfT(k, l) = Gf[k] * Gf[l];

					for (int k = 0; k < 3; k++)for (int l = 0; l < 3; l++){
						Real dot = tMass((k + 1) % 3, (l + 1) % 3) * GfGfT((k + 2) % 3, (l + 2) % 3)
							- tMass((k + 1) % 3, (l + 2) % 3) * GfGfT((k + 2) % 3, (l + 1) % 3)
							- tMass((k + 2) % 3, (l + 1) % 3) * GfGfT((k + 1) % 3, (l + 2) % 3)
							+ tMass((k + 2) % 3, (l + 2) % 3) * GfGfT((k + 1) % 3, (l + 1) % 3);
						tMatrix[3 * t + k][l].Value += dot;
					}
					for (int k = 0; k < 3; k++){
						Real dot = 0;
						for (int l = 0; l < 3; l++)dot += (tMass((k + 1) % 3, l)*Gf[(k + 2) % 3] - tMass((k + 2) % 3, l)*Gf[(k + 1) % 3]) * d[l];
						tVector[3 * t + k] += dot;
					}
				}
			}
		}
		else{
			for (int c = 0; c < Channels; c++){
				Point3D<Real> f((_value[0][0][c] + _value[1][0][c]) / 2.0, (_value[0][1][c] + _value[1][1][c]) / 2.0, (_value[0][2][c] + _value[1][2][c]) / 2.0);
				Point3D<Real> d(_difference[0][c], _difference[1][c], _difference[2][c]);

				Point3D<Real> Gf = GMatrix*f;
				SquareMatrix< Real, 3 > GfGfT;
				for (int k = 0; k < 3; k++)for (int l = 0; l < 3; l++) GfGfT(k, l) = Gf[k] * Gf[l];

				for (int k = 0; k < 3; k++)for (int l = 0; l < 3; l++){
					Real dot = tMass((k + 1) % 3, (l + 1) % 3) * GfGfT((k + 2) % 3, (l + 2) % 3)
						- tMass((k + 1) % 3, (l + 2) % 3) * GfGfT((k + 2) % 3, (l + 1) % 3)
						- tMass((k + 2) % 3, (l + 1) % 3) * GfGfT((k + 1) % 3, (l + 2) % 3)
						+ tMass((k + 2) % 3, (l + 2) % 3) * GfGfT((k + 1) % 3, (l + 1) % 3);
					tMatrix[3 * t + k][l].Value += 2.0 * dot;
				}
				for (int k = 0; k < 3; k++){
					Real dot = 0;
					for (int l = 0; l < 3; l++)dot += (tMass((k + 1) % 3, l)*Gf[(k + 2) % 3] - tMass((k + 2) % 3, l)*Gf[(k + 1) % 3]) * d[l];
					tVector[3 * t + k] += 2.0 * dot;
				}
			}

		}
	}
	linearSystem = flowData.signedEdge_transpose * tMatrix *flowData.signedEdge;
	rhs.resize(linearSystem.rows);
	flowData.signedEdge_transpose.Multiply(GetPointer(tVector), GetPointer(rhs));
}
#endif

template< class Real, int Channels >
void EstimateFlowWhitneyBasis
(
FlowData< Real, Channels >& flowData,
Real scalarSmoothWeight, Real vectorSmoothWeight)
{
	std::vector< Point< Real, Channels > > smoothed[2];
	std::vector< Point< Real, Channels > > resampled[2];

	const std::vector< Point3D< Real > >& vertices = flowData.vertices;
	const std::vector< TriangleIndex >& triangles = flowData.triangles;
	ConstPointer(Point3D< Real >) _vertices = (ConstPointer(Point3D< Real >))GetPointer(vertices);

	// Resample to the midway point(and smooth?) 
	{
		Timer t;
#if BARICENTRIC_WHITNEY
		for (int s = 0; s<2; s++) ResampleSignal(flowData.mesh, (ConstPointer(Point2D< Real >))GetPointer(flowData.tFlowField), (ConstPointer(FEM::EdgeXForm< Real >))flowData.edges, flowData.signals[s], resampled[s], (Real)(s == 0 ? -0.5 : 0.5), Threads.value);
#else
		for (int s = 0; s < 2; s++) ResampleSignalWhitney(flowData.mesh, (ConstPointer(Real))GetPointer(flowData.eFlowField), (ConstPointer(FEM::EdgeXForm< Real >))flowData.edges, flowData.signals[s], resampled[s], (Real)(s == 0 ? -0.5 : 0.5), Threads.value);
#endif
		if (scalarSmoothWeight) for (int s = 0; s < 2; s++) flowData.smoothSignal(resampled[s], smoothed[s], scalarSmoothWeight);
		for (int s = 0; s < 2; s++) resampled[s] = smoothed[s];
		if (Verbose.set) printf("\tResampled signal: %.2f(s)\n", t.elapsed());
	}


	if (Debug.set)
	{
		static int count = 0;
		char fileName[512];
		sprintf(fileName, "resampled.S.%d.ply", count), OutputMesh(fileName, vertices, resampled[0], triangles, PLY_BINARY_NATIVE);
		sprintf(fileName, "resampled.T.%d.ply", count), OutputMesh(fileName, vertices, resampled[1], triangles, PLY_BINARY_NATIVE);
		count++;
	}

	Timer tls;
	SparseMatrix< Real, int > M;
	std::vector< Real > b;
#if BARICENTRIC_WHITNEY
	SetBaricentricWhitneyLinearSystem(flowData, resampled, M, b);
#else
	SetFullWhitneyLinearSystem(flowData, resampled, M, b);
#endif
	if (Verbose.set) printf("\tSet Linear System: %.2f(s)\n", tls.elapsed());

	std::vector< Real >  dampedSolution;
	dampedSolution.resize(M.rows, Real(0));

	if (Verbose.set) printf("\t Data Matrix  [%d x %d: %f]\n", M.rows, M.rows, (Real)M.Entries() / M.rows);
	if (Verbose.set) printf("\t Smoothness Matrix  [%d x %d: %f]\n", flowData.hodgeLaplacian.rows, flowData.hodgeLaplacian.rows, (Real)flowData.hodgeLaplacian.Entries() / flowData.hodgeLaplacian.rows);

	{// Solve for the flow field
		Real scale = (Real)1. / sqrt(M.SquareNorm());
		M *= scale;
		for (int i = 0; i < M.rows; i++) b[i] *= scale;
		SparseMatrix< Real, int > L = M;
		if (Verbose.set) printf("Vector Smooth Weight %g \n", vectorSmoothWeight);
		if (vectorSmoothWeight) L += flowData.hodgeLaplacian * vectorSmoothWeight;
		if (Verbose.set) printf("\tSet system: [%d x %d: %f]\n", L.rows, L.rows, (Real)L.Entries() / L.rows);
		Timer tupdate;
		flowData.whitneyVFSolver->update(L);
		if (Verbose.set) printf("\tUpdate Linear System: %.2f(s)\n", tupdate.elapsed());
		Timer tsolve;
		flowData.whitneyVFSolver->solve(GetPointer(b), GetPointer(dampedSolution));
		if (Verbose.set) printf("\tSolve Linear System: %.2f(s)\n", tsolve.elapsed());
	}


	Real startError;
#if BARICENTRIC_WHITNEY
	if (Verbose.set) startError = flowData.getError();
#else
	if (Verbose.set) startError = flowData.getWhitneySymmetricError();
#endif


	if (Krylov.value > 1){
		Timer tKrylov;
		std::vector<std::vector< Real >>  KrylovBasis;
		KrylovBasis.resize(Krylov.value, std::vector< Real >(M.rows, Real(0)));
		
		Real d0SqNorm = 0;
#pragma omp parallel for reduction( + :d0SqNorm)
		for (int i = 0; i < flowData.whitneyMass.rows; i++) for (int j = 0; j < flowData.whitneyMass.rowSizes[i]; j++)
			d0SqNorm += flowData.whitneyMass[i][j].Value * dampedSolution[i] * dampedSolution[flowData.whitneyMass[i][j].N];

#pragma omp parallel for
		for (int i = 0; i < M.rows; i++) KrylovBasis[0][i] = dampedSolution[i];

		Timer tbasis;
		for (int k = 1; k < Krylov.value; k++){
			std::vector< Real > KrylovVector(M.rows);
			flowData.whitneyMass.Multiply(GetPointer(KrylovBasis[k - 1]), GetPointer(KrylovVector));
			flowData.whitneyVFSolver->solve(GetPointer(KrylovVector), GetPointer(KrylovBasis[k]));

			Real dkSqNorm = 0;
#pragma omp parallel for reduction( + :dkSqNorm)
			for (int i = 0; i < flowData.whitneyMass.rows; i++) for (int j = 0; j < flowData.whitneyMass.rowSizes[i]; j++)
				dkSqNorm += flowData.whitneyMass[i][j].Value * KrylovBasis[k][i] * KrylovBasis[k][flowData.whitneyMass[i][j].N];

			Real normalizationFactor = sqrt(d0SqNorm / dkSqNorm);
#pragma omp parallel for
			for (int i = 0; i < M.rows; i++) KrylovBasis[k][i] *= normalizationFactor;
		}

		if (Verbose.set) printf("\tKrylov Basis Computation: %.2f(s)\n", tbasis.elapsed());

		std::vector<std::vector< Real >>  MKrylovBasis;
		MKrylovBasis.resize(Krylov.value, std::vector< Real >(M.rows, Real(0)));

#pragma omp parallel for
		for (int k = 0; k < Krylov.value; k++){
			M.Multiply(GetPointer(KrylovBasis[k]), GetPointer(MKrylovBasis[k]));
		}

		Eigen::MatrixXd KrylovSystemMatrix(Krylov.value, Krylov.value);
#pragma omp parallel for
		for (int k = 0; k < Krylov.value; k++) for (int l = 0; l < Krylov.value; l++){ //Optimize uisng symmetry of M
			Real dot = Dot(GetPointer(KrylovBasis[k]), GetPointer(MKrylovBasis[l]), M.rows);
			KrylovSystemMatrix(k,l) = dot;
		}

		Eigen::VectorXd KrylovSystemRHS(Krylov.value);
#pragma omp parallel for
		for (int k = 0; k < Krylov.value; k++){
			KrylovSystemRHS[k] = Dot(GetPointer(KrylovBasis[k]), GetPointer(b), M.rows);
		}

		Eigen::VectorXd  KrylovSystemSolution = KrylovSystemMatrix.llt().solve(KrylovSystemRHS);
		for (int k = 0; k < Krylov.value; k++){
#pragma omp parallel for
			for (int i = 0; i < M.rows; i++) dampedSolution[i] += KrylovSystemSolution[k]*KrylovBasis[k][i];
		}

		Real quadTerm =  KrylovSystemSolution.dot(KrylovSystemMatrix * KrylovSystemSolution);
		Real linearTerm = KrylovSystemSolution.dot(KrylovSystemRHS);

		if (Verbose.set) printf("\tSolve Krylov System: %.2f(s)\n", tKrylov.elapsed());
#if BARICENTRIC_WHITNEY
		std::vector< Real >  _correctionFlowField;
		_correctionFlowField.resize(2 * triangles.size());
		flowData.whitneyProlongation.Multiply(GetPointer(dampedSolution), GetPointer(_correctionFlowField));
		for (int i = 0; i<triangles.size(); i++) flowData.tFlowField[i] += Point2D<Real>(_correctionFlowField[2 * i], _correctionFlowField[2 * i + 1]);
		if (Verbose.set) printf("\tError: %g -> %g\n", startError, flowData.getError());
#else 
		for (int i = 0; i < M.rows; i++) flowData.reduced_eFlowField[i] += dampedSolution[i];
		for (int i = 0; i < triangles.size(); i++)for (int j = 0; j < 3; j++) flowData.eFlowField[3 * i + j] += flowData.positiveOrientedEdge[3 * i + j] ? dampedSolution[flowData.reducedEdgeIndex[3 * i + j]] : -dampedSolution[flowData.reducedEdgeIndex[3 * i + j]];
		if (Verbose.set){
			printf("\tError: %g -> %g\n", startError, flowData.getWhitneySymmetricError());
			Real correctionFlowNorm = 0;
#pragma omp parallel for reduction( + :correctionFlowNorm)
			for (int i = 0; i < flowData.whitneyMass.rows; i++) for (int j = 0; j < flowData.whitneyMass.rowSizes[i]; j++)
				correctionFlowNorm += flowData.whitneyMass[i][j].Value * dampedSolution[i] * dampedSolution[flowData.whitneyMass[i][j].N];
			printf("\tCorrection Flow Norm: %g\n", correctionFlowNorm);
		}
#endif
	}
	else{
		Real step = Real(1.0);
		{// Solve for step
			if (vectorSmoothWeight){
				std::vector<Real> MdampedSolution(M.rows);
				M.Multiply(GetPointer(dampedSolution), GetPointer(MdampedSolution));
				Real denom = Dot(GetPointer(dampedSolution), GetPointer(MdampedSolution), M.rows);
				Real num = Dot(GetPointer(dampedSolution), GetPointer(b), M.rows);


				if (denom) step = num / denom;
				else step = 0.0;
				if (step) for (int i = 0; i < dampedSolution.size(); i++)dampedSolution[i] *= step;
			}
		}
#if BARICENTRIC_WHITNEY
		std::vector< Real >  _correctionFlowField;
		_correctionFlowField.resize(2 * triangles.size());
		flowData.whitneyProlongation.Multiply(GetPointer(dampedSolution), GetPointer(_correctionFlowField));
		for (int i = 0; i<triangles.size(); i++) flowData.tFlowField[i] += Point2D<Real>(_correctionFlowField[2 * i], _correctionFlowField[2 * i + 1]);
		if (Verbose.set) printf("\tError: (%g) %g -> %g\n", step, startError, flowData.getError());
#else 
		for (int i = 0; i < M.rows; i++) flowData.reduced_eFlowField[i] += dampedSolution[i];
		for (int i = 0; i < triangles.size(); i++)for (int j = 0; j < 3; j++) flowData.eFlowField[3 * i + j] += flowData.positiveOrientedEdge[3 * i + j] ? dampedSolution[flowData.reducedEdgeIndex[3 * i + j]] : -dampedSolution[flowData.reducedEdgeIndex[3 * i + j]];
		if (Verbose.set){
			printf("\tError: (%g) %g -> %g\n", step, startError, flowData.getWhitneySymmetricError());
			Real correctionFlowNorm = 0;
#pragma omp parallel for reduction( + :correctionFlowNorm)
			for (int i = 0; i < flowData.whitneyMass.rows; i++) for (int j = 0; j < flowData.whitneyMass.rowSizes[i]; j++)
				correctionFlowNorm += flowData.whitneyMass[i][j].Value * dampedSolution[i] * dampedSolution[flowData.whitneyMass[i][j].N];
			printf("\tCorrection Flow Norm: %g\n", correctionFlowNorm);
		}
#endif
	}
}

// We assume that we are given a flow field which when applied forward (approximately) takes the source to the target
// and when applied backwards (approximately) takes the target to the source
template< class Real, int Channels >
void EstimateFlow
(
FlowData< Real, Channels >& flowData,
Real scalarSmoothWeight, Real vectorSmoothWeight
)
{
	std::vector< Point< Real, Channels > > smoothed[2];
	std::vector< Point< Real, Channels > > resampled[2];

	const std::vector< Point3D< Real > >& vertices = flowData.vertices;
	const std::vector< TriangleIndex >& triangles = flowData.triangles;
	ConstPointer(Point3D< Real >) _vertices = (ConstPointer(Point3D< Real >))GetPointer(vertices);

	// Resample to the midway point and smooth the signal 
	{
		Timer t;
		for (int s = 0; s<2; s++) ResampleSignal(flowData.mesh, (ConstPointer(Point2D< Real >))GetPointer(flowData.tFlowField), (ConstPointer(FEM::EdgeXForm< Real >))flowData.edges, flowData.signals[s], resampled[s], (Real)(s == 0 ? -0.5 : 0.5), Threads.value);
		for (int s = 0; s<2; s++) flowData.smoothSignal(resampled[s], smoothed[s], scalarSmoothWeight);
		for (int s = 0; s < 2; s++) resampled[s] = smoothed[s];

		if (Verbose.set) printf("\tSmoothed and resampled signal: %.2f(s)\n", t.elapsed());
	}


	if (Debug.set)
	{
		static int count = 0;
		char fileName[512];
		sprintf(fileName, "resampled.S.%d.ply", count), OutputMesh(fileName, vertices, resampled[0], triangles, PLY_BINARY_NATIVE);
		sprintf(fileName, "resampled.T.%d.ply", count), OutputMesh(fileName, vertices, resampled[1], triangles, PLY_BINARY_NATIVE);
		count++;
	}

	// Solve for the flow field
	{
		Timer t;
		SparseMatrix< Real, int > _M;
		_M.resize(2 * triangles.size());
		for (int i = 0; i<triangles.size(); i++) for (int j = 0; j<2; j++)
		{
			_M.SetRowSize(2 * i + j, 2);
			for (int k = 0; k<2; k++) _M[2 * i + j][k] = MatrixEntry< Real, int >(2 * i + k, (Real)0);
		}
		Pointer(Real) _b = AllocPointer< Real >(_M.rows);
		memset(_b, 0, sizeof(Real)* _M.rows);
		SparseMatrix< Real, int >& M = flowData.vfM; //Where is flowData.vfM used again?
#pragma omp parallel for num_threads( Threads.value )
		for (int i = 0; i<M.rows; i++) for (int j = 0; j<M.rowSizes[i]; j++) M[i][j].Value = (Real)0;
		Pointer(Real) b = AllocPointer< Real >(M.rows);
		memset(b, 0, sizeof(Real)* M.rows);

		Pointer(Real) bm = AllocPointer< Real >(flowData.gradientAndJGradientRestriction.rows);
		memset(bm, 0, sizeof(Real)*flowData.gradientAndJGradientRestriction.rows);
		SparseMatrix< Real, int > Mm;

		// We want the vector field X to explain the difference between the target and source and we want its negative to explain the difference between the source and target
		// [SOURCE]
		//		\sum_{c} G(deformed_source[c])^t * D * G(deformed_source[c]) * X + \epsilon * ( ...regularization... ) * X = \sum_{c} G(deformed_source[c])^t * D * ( deformed_source - deformed_target )[c]
		// [TARGET]
		//		\sum_{c} G(deformed_target[c])^t * D * G(deformed_target[c]) * X + \epsilon * ( ...regularization... ) * X = \sum_{c} G(deformed_target[c])^t * D * ( deformed_source - deformed_target )[c]
		// [SOURCE+TARGET]
		//		\sum_{c} [ G(deformed_source[c])^t * D * G(deformed_source[c]) +  G(deformed_target[c])^t * D * G(deformed_target[c]) + \epsilon * ( ...regularization... ) ] * X =
		//				= \sum_{c} [ G(deformed_source[c])^t * D  + G(deformed_target[c])^t * D ] * ( deformed_source - deformed_target )[c]
		{
			Pointer(Real) _dValues = AllocPointer< Real >(triangles.size());							// The per-triangle difference in color values
			Pointer(Real) _sValues = AllocPointer< Real >(vertices.size());								// The source values
			Pointer(Point2D< Real >) _gValues = AllocPointer< Point2D< Real > >(triangles.size());		// The gradient of the source values
			for (int c = 0; c<Channels; c++)
			{
#pragma omp parallel for num_threads( Threads.value )
				for (int i = 0; i<triangles.size(); i++)
				{
					Real source = resampled[0][triangles[i][0]][c] + resampled[0][triangles[i][1]][c] + resampled[0][triangles[i][2]][c];
					Real target = resampled[1][triangles[i][0]][c] + resampled[1][triangles[i][1]][c] + resampled[1][triangles[i][2]][c];
					_dValues[i] = (source - target) / 3.f;
				}
				for (int s = 0; s<2; s++)
				{
					for (int i = 0; i<vertices.size(); i++) _sValues[i] = resampled[s][i][c];
					flowData.grad.Multiply(_sValues, (Pointer(Real))_gValues);
#pragma omp parallel for num_threads( Threads.value )
					for (int i = 0; i<triangles.size(); i++)
					{
						SquareMatrix< Real, 2 > m = flowData.mesh.vectorFieldDotMass(i, _gValues[i]);
						for (int j = 0; j<2; j++) for (int k = 0; k<2; k++) _M[2 * i + j][k].Value += m(j, k);
					}
					flowData.mesh.setVectorFieldDotDual(_gValues, _dValues, (Pointer(Point2D< Real >))_b, FEM::SYSTEM_ADD);
				}
			}
			FreePointer(_dValues);
			FreePointer(_sValues);
			FreePointer(_gValues);
			Real scale = (Real)1. / sqrt(_M.SquareNorm());
			_M *= scale;
			for (int i = 0; i<_M.rows; i++) _b[i] *= scale;
			flowData.gradientAndJGradient_transpose.Multiply(_b, b);
			M = flowData.gradientAndJGradient_transpose * _M * flowData.gradientAndJGradient;
			if (vectorSmoothWeight) M += flowData.vfStiffness * vectorSmoothWeight;
		}
		{
			Timer t;

			if (Verbose.set) printf("\tSet system: %.2f (s) [%d x %d: %f]\n", t.elapsed(), Mm.rows, Mm.rows, (Real)Mm.Entries() / Mm.rows);
			flowData.vfSolver = new EigenCholeskySolverLDLt(Mm);
			std::vector< Real >  dVFlowFieldm;
			dVFlowFieldm.resize(flowData.gradientAndJGradientRestriction.rows, Real(0));
			flowData.vfSolver->solve((ConstPointer(Real))bm, GetPointer(dVFlowFieldm));
			flowData.gradientAndJGradientProlongation.Multiply(GetPointer(dVFlowFieldm), GetPointer(flowData.dVFlowField));

			if (Verbose.set) printf("\tSolved system: %.2f (s)\n", t.elapsed());
		}
		FreePointer(b);
		FreePointer(bm);
		FreePointer(_b);
	}
	{
		Timer t;

		Real step = (Real)0;
		Real startError;
		if (ShowError.set) startError = flowData.getError();
		{
			// Ignore the regularization term and find the scalar s that minimizes the energy:
			//	E(s)	= \sum_{T} || < s*X , \nabla deformed_{source/target} > - delta ||^2 * |T|
			//			= \sum_{T} [ s^2 < X , \nabla deformed_{source/target} >^2 - 2 * s * delta * < X , \nabla deformed_{source/target} > + delta^2 ] * |T|
			//	=> 0	= \sum_{T} s [ < X , \nabla deformed_{source/target} >^2 - delta * < X , \nabla deformed_{source/target } > ] * |T|
			//	=> s	= ( \sum_{T} delta * < X , \nabla deformed_{source/target } > ) * |T| / ( \sum_{T} < X , \nabla deformed_{source/target} >^2 * |T| )
			Real num = 0, denom = 0;
			flowData.gradientAndJGradient.Multiply((ConstPointer(Real))GetPointer(flowData.dVFlowField), (Pointer(Real))GetPointer(flowData.dTFlowField));
			Pointer(Real) _dValues = AllocPointer< Real >(triangles.size());							// The per-triangle difference in color values
			Pointer(Real) _sValues = AllocPointer< Real >(vertices.size());								// The source values
			Pointer(Point2D< Real >) _gValues = AllocPointer< Point2D< Real > >(triangles.size());		// The gradient of the source values
			for (int c = 0; c<Channels; c++)
			{
				for (int i = 0; i<triangles.size(); i++)
				{
					Real source = resampled[0][triangles[i][0]][c] + resampled[0][triangles[i][1]][c] + resampled[0][triangles[i][2]][c];
					Real target = resampled[1][triangles[i][0]][c] + resampled[1][triangles[i][1]][c] + resampled[1][triangles[i][2]][c];
					_dValues[i] = (source - target) / 3.f;
				}
				for (int s = 0; s<2; s++)
				{
					for (int i = 0; i<vertices.size(); i++) _sValues[i] = resampled[s][i][c];
					flowData.grad.Multiply(_sValues, (Pointer(Real))_gValues);
					for (int i = 0; i<triangles.size(); i++)
					{
						Real area = flowData.mesh.area(i);
						Real dot = Point2D< Real >::Dot(flowData.mesh.g[i] * _gValues[i], flowData.dTFlowField[i]);
						num += _dValues[i] * dot * area, denom += dot * dot * area;
					}
				}
			}
			FreePointer(_dValues);
			FreePointer(_sValues);
			FreePointer(_gValues);
			step = num / denom;
		}
		if (GSSearch.set)
		{
#pragma omp parallel for
			for (int i = 0; i<flowData.dVFlowField.size(); i++) flowData.dVFlowField[i] *= step;
		}
		else
		{
			flowData.combineFlows(step);
			if (0){//project flow filed
				std::vector<Point2D<Real>> pFlowField;
				projectVectorField(flowData.tFlowField, pFlowField, flowData.triangleBasedHighVFMass, flowData.triangleBasedVFMass, flowData.triangleBasedVFStiffness, Real(0), flowData.triangleBasedVFRestriction, flowData.triangleBasedVFProlongation);
				flowData.tFlowField = pFlowField;
			}
			if (ShowError.set) printf("\tError: (%g) %g -> %g\t%.2f (s)\n", step, startError, flowData.getError(), t.elapsed());
		}
	}
	if (GSSearch.set)
	{
		Timer t;
		Real startError;
		if (ShowError.set) startError = flowData(0);
		Real step = GoldenSectionSearch(flowData, (Real)GSSearch.value, (Real)1., (Real)(1.f / GSSearch.value), flowData((Real)1.), (Real)1e-3);
		flowData.combineFlows(step);
		if (ShowError.set) printf("\tError: (%g) %g -> %g\t%.2f (s)\n", step, startError, flowData.getError(), t.elapsed());
	}

	if (Verbose.set)
	{
		ConstPointer(Real) x = (ConstPointer(Real))GetPointer(flowData.vFlowField);
		Pointer(Real) Mx = AllocPointer< Real >(flowData.vfMass.rows);
		flowData.vfMass.Multiply(x, Mx);
		Real m = Dot((ConstPointer(Real))Mx, (ConstPointer(Real))x, flowData.vfMass.rows);
		flowData.vfStiffness.Multiply(x, Mx);
		Real s = Dot((ConstPointer(Real))Mx, (ConstPointer(Real))x, flowData.vfMass.rows);
		printf("\tFlow-field mass/stiffness: %g / %g -> %g\n", sqrt(m), sqrt(s), sqrt(s / m));
		FreePointer(Mx);
	}

	if (Debug.set)
	{
		for (int s = 0; s<2; s++) ResampleSignal(flowData.mesh, (ConstPointer(Point2D< Real >))GetPointer(flowData.tFlowField), (ConstPointer(FEM::EdgeXForm< Real >))flowData.edges, flowData.signals[s], resampled[s], (Real)(s == 0 ? -0.5 : 0.5), Threads.value);
		static int count = 0;
		char fileName[512];
		sprintf(fileName, "midpoint.S.%d.ply", count), OutputMesh(fileName, vertices, resampled[0], triangles, PLY_BINARY_NATIVE);
		sprintf(fileName, "midpoint.T.%d.ply", count), OutputMesh(fileName, vertices, resampled[1], triangles, PLY_BINARY_NATIVE);
		count++;
	}
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
#if BARICENTRIC_WHITNEY
			ResampleSignal(flowData.mesh, (ConstPointer(Point2D< Real >))GetPointer(flowData.tFlowField), (ConstPointer(FEM::EdgeXForm< Real >))flowData.edges, colors[s], outputColors[s], length, threads);
#else
			ResampleSignalWhitney(flowData.mesh, (ConstPointer(Real))GetPointer(flowData.eFlowField), (ConstPointer(FEM::EdgeXForm< Real >))flowData.edges, colors[s], outputColors[s], length, threads);
#endif
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

#if BARICENTRIC_WHITNEY
				flowData.mesh.flow(flowData.edges, GetPointer(flowData.tFlowField), length, p, (Real)1e-2);
#else
				flowData.mesh.whitneyFlow(flowData.edges, GetPointer(flowData.eFlowField), length, p, (Real)1e-2);
#endif
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
#include "SurfaceVisualization.inl"



template< class Real >
void projectSignal(const std::vector<Real> & in, std::vector< Real> & out, const SparseMatrix<Real, int> & lowMass, const SparseMatrix<Real, int> & highMass, const SparseMatrix<Real, int> & lowStiffness, const SparseMatrix<Real, int> & highStiffness, const SparseMatrix<Real, int> & lowBiStiffness, const SparseMatrix<Real, int> & highBiStiffness, const SparseMatrix<Real, int> & restriction, const SparseMatrix<Real, int> & prolongation, const Real smoothWeight, const Real biSmoothWeight, bool fitGradient, bool fitLaplacian){
	int vCount = in.size();
	out.resize(vCount);
	Timer t;
	SparseMatrix<Real, int> M = lowMass;
	if (smoothWeight) M += lowStiffness*smoothWeight;
	if (biSmoothWeight) M += lowBiStiffness*biSmoothWeight;
	EigenCholeskySolverLLt pSolver(M);

	Pointer(Real) signal = AllocPointer< Real >(vCount);
	Pointer(Real) signalMass = AllocPointer< Real >(vCount);
	for (int i = 0; i<vCount; i++) signal[i] = in[i];

	// Get the high-resolution mass
	highMass.Multiply(signal, signalMass);
	if (fitGradient && smoothWeight){
		Pointer(Real) signalStifness = AllocPointer< Real >(vCount);
		highStiffness.Multiply(signal, signalStifness);
		for (int i = 0; i < vCount; i++) signalMass[i] += smoothWeight * signalStifness[i];
		FreePointer(signalStifness);
	}

	if (fitLaplacian &&  biSmoothWeight){
		Pointer(Real) signalBiStifness = AllocPointer< Real >(vCount);
		highBiStiffness.Multiply(signal, signalBiStifness);
		for (int i = 0; i < vCount; i++) signalMass[i] += biSmoothWeight * signalBiStifness[i];
		FreePointer(signalBiStifness);
	}

	// Restrict to the low-resolution mass
	{
		Pointer(Real) temp = AllocPointer< Real >(restriction.rows);
		restriction.Multiply(signalMass, temp);
		FreePointer(signalMass);
		signalMass = temp;
	}

	// Solve the low-resolution system
	{
		Pointer(Real) temp = AllocPointer< Real >(lowMass.rows);
		pSolver.solve((ConstPointer(Real))signalMass, temp);
		FreePointer(signal);
		signal = temp;
	}

	Real n = 0, e = 0, b = 0;
	if (Verbose.set)
	{
#pragma omp parallel for reduction( + : e , n )
		for (int i = 0; i<lowStiffness.rows; i++) for (int j = 0; j<lowStiffness.rowSizes[i]; j++)
			e += lowStiffness[i][j].Value * signal[i] * signal[lowStiffness[i][j].N], n += lowMass[i][j].Value * signal[i] * signal[lowMass[i][j].N];

#pragma omp parallel for reduction( + : b)
		for (int i = 0; i < lowBiStiffness.rows; i++) for (int j = 0; j < lowBiStiffness.rowSizes[i]; j++)
			b += lowBiStiffness[i][j].Value * signal[i] * signal[lowBiStiffness[i][j].N];

	}

	// Prolong to the high-resolution system
	{
		Pointer(Real) temp = AllocPointer< Real >(prolongation.rows);
		prolongation.Multiply(signal, temp);
		FreePointer(signal);
		signal = temp;
	}

	for (int i = 0; i < vCount; i++) out[i] = signal[i];

	if (Verbose.set)
	{
		Real B = 0;
#pragma omp parallel for reduction( + : B)
		for (int i = 0; i<highBiStiffness.rows; i++) for (int j = 0; j<highBiStiffness.rowSizes[i]; j++)
			B += highBiStiffness[i][j].Value *  signal[i] * signal[highBiStiffness[i][j].N];

		Real d = 0;
#pragma omp parallel for reduction( + : d)
		for (int i = 0; i<highMass.rows; i++) for (int j = 0; j<highMass.rowSizes[i]; j++)
			d += highMass[i][j].Value * (in[i] - out[i]) * (in[highMass[i][j].N] - out[highMass[i][j].N]);
		printf("HBiStiffness/LBiStiffness/Stiffness/Mass/Difference : %g/%g/%g/%g/%g\n", B, b, e, n, d);
	}

	FreePointer(signal);
	FreePointer(signalMass);
}

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

template< class Real >
struct ExteriorOperators{
	SparseMatrix<Real, int> d0, d1;
	SparseMatrix<Real, int> m0, m1, m2;
	SparseMatrix<Real, int> m0_inv, m1_inv, m2_inv;
	SparseMatrix<Real, int> rotationalEnergy;
	SparseMatrix<Real, int> divergenceEnergy;
	SparseMatrix<Real, int> hodgeLaplacian;
	SparseMatrix<Real, int> vfMass;
	SparseMatrix<Real, int> signedEdge;
	std::vector<int> reducedEdgeIndex;
	std::vector<int> expandedEdgeIndex;
	std::vector<bool> positiveOrientedEdge;
};
template< class Real, int Channels >
struct WhitneyFlowViewer
{
	static SurfaceVisualization sv;
	static  FlowData< Real, Channels > flowData;
	static bool processTexture;
	static InputTextureData< Real > inputTextureData;
	static InputGeometryData< Real > inputGeometryData;

	//VertexBased
	static std::vector<Point3D<Real>> inputSignal[2];
	static std::vector<Point3D<Real>> inputAdvectedSignal[2];

	//TextureBase
	static Point3D<Real> * inputTexture[2];
	static Point3D<Real> * inputAdvectedTexture[2];


	//Exterior Operators
	static ExteriorOperators<Real> exteriorOperators;
	static void SetExteriorOperators();

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

	static void Advance();
	static void AdvanceLevelCallBack(Visualization* v, const char* prompt);

	static void RepeatLevelCallBack(Visualization* v, const char* prompt);
	static void UpdateSignalVisualization();
	static void UpdateTextureVisualization();
	static void ToggleSignalModeCallBack(Visualization* v, const char* prompt);
	static void ToggleSignalSourceCallBack(Visualization* v, const char* prompt);
	static void OutputResultCallBack(Visualization* v, const char* prompt);

	static void ScalarSmoothWeightCallBack(Visualization* v, const char* prompt);
	static void VectorFieldSmoothWeightCallBack(Visualization* v, const char* prompt);
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

template< class Real, int Channels > ExteriorOperators<Real>			WhitneyFlowViewer< Real, Channels>::exteriorOperators;

template< class Real, int Channels > void WhitneyFlowViewer< Real, Channels>::Idle(void){ sv.Idle(); }
template< class Real, int Channels > void WhitneyFlowViewer< Real, Channels>::KeyboardFunc(unsigned char key, int x, int y){ sv.KeyboardFunc(key, x, y); }
template< class Real, int Channels > void WhitneyFlowViewer< Real, Channels>::SpecialFunc(int key, int x, int y){ sv.SpecialFunc(key, x, y); }
template< class Real, int Channels > void WhitneyFlowViewer< Real, Channels>::Display(void){ sv.Display(); }
template< class Real, int Channels > void WhitneyFlowViewer< Real, Channels>::Reshape(int w, int h){ sv.Reshape(w, h); }
template< class Real, int Channels > void WhitneyFlowViewer< Real, Channels>::MouseFunc(int button, int state, int x, int y){ sv.MouseFunc(button, state, x, y); }
template< class Real, int Channels > void WhitneyFlowViewer< Real, Channels>::MotionFunc(int x, int y){ sv.MotionFunc(x, y); }

template< class Real, int Channels >
void WhitneyFlowViewer< Real, Channels>::SetExteriorOperators(){
	//Set edge indices
	Timer tm;

	SparseMatrix<Real, int> & d0 = exteriorOperators.d0;
	SparseMatrix<Real, int> & d1 = exteriorOperators.d1;

	SparseMatrix<Real, int> & m0 = exteriorOperators.m0;
	SparseMatrix<Real, int> & m1 = exteriorOperators.m1;
	SparseMatrix<Real, int> & m2 = exteriorOperators.m2;
		
	SparseMatrix<Real, int> & m0_inv = exteriorOperators.m0_inv;
	SparseMatrix<Real, int> & m1_inv = exteriorOperators.m1_inv;
	SparseMatrix<Real, int> & m2_inv = exteriorOperators.m2_inv;

	SparseMatrix<Real, int> & rotationalEnergy = exteriorOperators.rotationalEnergy;
	SparseMatrix<Real, int> & divergenceEnergy = exteriorOperators.divergenceEnergy;
	SparseMatrix<Real, int> & hodgeLaplacian = exteriorOperators.hodgeLaplacian;

	std::vector<TriangleIndex> triangles = flowData.triangles;
	int tCount = triangles.size();

	std::vector<int> & reducedEdgeIndex = exteriorOperators.reducedEdgeIndex;
	std::vector<int> & expandedEdgeIndex = exteriorOperators.expandedEdgeIndex;
	std::vector<bool> & positiveOrientedEdge = exteriorOperators.positiveOrientedEdge;

	reducedEdgeIndex.resize(3 * tCount, -1);
	expandedEdgeIndex.reserve(3 * tCount);
	positiveOrientedEdge.resize(3 * tCount, true);
	
	int currentIndex = 0;
	for (int i = 0; i < tCount; i++)for (int j = 0; j < 3; j++){
		if (reducedEdgeIndex[3 * i + j] == -1){
			expandedEdgeIndex.push_back(3 * i + j);
			reducedEdgeIndex[3 * i + j] = currentIndex;
			int oppositeEdge = flowData.edges[3 * i + j].oppositeEdge;
			if (oppositeEdge != -1){
				reducedEdgeIndex[oppositeEdge] = currentIndex;
				positiveOrientedEdge[oppositeEdge] = false;
			}
			currentIndex++;
		}
	}
	

	//d0
	int eCount = expandedEdgeIndex.size();
	d0.resize(eCount);
	for (int i = 0; i < eCount; i++){
		d0.SetRowSize(i,2);
		int t = expandedEdgeIndex[i] / 3;
		int v = expandedEdgeIndex[i] % 3;
		d0[i][0] = MatrixEntry< Real, int >(triangles[t][(v + 1) % 3] , Real(-1));
		d0[i][1] = MatrixEntry< Real, int >(triangles[t][(v + 2) % 3], Real(1));
	}


	//d1
	d1.resize(tCount);
	for (int i = 0; i < tCount; i++){
		d1.SetRowSize(i, 3);
		for (int j = 0; j < 3; j++){
			d1[i][j] = MatrixEntry< Real, int >(reducedEdgeIndex[3 * i + j], positiveOrientedEdge[3 * i + j] ? Real(1) : Real(-1));
		}
	}
	
	//m0
	//Barycentric areas
	int vCount = flowData.vertices.size();
	std::vector<Real> baricentryArea(vCount,Real(0));
	for (int t = 0; t < tCount; t++)for (int v = 0; v < 3; v++){
		Real area = flowData.mesh.area(t)/Real(3);
		baricentryArea[triangles[t][v]] += area;
	}
	m0.resize(vCount);
	m0_inv.resize(vCount);
	for (int i = 0; i < vCount; i++){
		m0.SetRowSize(i, 1);
		m0[i][0] = MatrixEntry< Real, int >(i, baricentryArea[i]);
		
		m0_inv.SetRowSize(i, 1);
		m0_inv[i][0] = MatrixEntry< Real, int >(i, 1.0 / baricentryArea[i]);
	}


	//m1 
	m1.resize(eCount);
	m1_inv.resize(eCount);
	Point2D< Real > grad[3] = { Point2D< Real >(-1.0, -1.0), Point2D< Real >(1.0, 0.0), Point2D< Real >(0.0, 1.0) };
	for (int i = 0; i < eCount; i++){
		int t = expandedEdgeIndex[i] / 3;
		int v = expandedEdgeIndex[i] % 3;

		Real r = -flowData.mesh.area(t) * Point2D< Real >::Dot(grad[(v + 1) % 3], flowData.mesh.g[t].inverse() * grad[(v + 2) % 3]);

		int oppositeEdge = flowData.edges[3 * t + v].oppositeEdge;
		if (oppositeEdge != -1){
			int tt = oppositeEdge / 3;
			int vv = oppositeEdge % 3;
			r += -flowData.mesh.area(tt) * Point2D< Real >::Dot(grad[(vv + 1) % 3], flowData.mesh.g[tt].inverse() * grad[(vv + 2) % 3]);
		}

		m1.SetRowSize(i, 1);
		m1[i][0] = MatrixEntry< Real, int >(i, r);

		m1_inv.SetRowSize(i, 1);
		m1_inv[i][0] = MatrixEntry< Real, int >(i,  r ? 1/r : 0);
	}

	
	//m2 Inverse Triangle areas
	
	m2.resize(tCount);
	m2_inv.resize(tCount);
	for (int i = 0; i < tCount; i++){
		m2.SetRowSize(i, 1);
		m2[i][0] = MatrixEntry< Real, int >(i, 1.0 / flowData.mesh.area(i));

		m2_inv.SetRowSize(i, 1);
		m2_inv[i][0] = MatrixEntry< Real, int >(i, flowData.mesh.area(i));
	}

	//Set differential opertors
	rotationalEnergy = d1.transpose() * m2 * d1;
	divergenceEnergy = m1 * d0 * m0_inv * d0.transpose() * m1;
	hodgeLaplacian = rotationalEnergy + divergenceEnergy;


	SparseMatrix<Real, int> signedEdge;
	signedEdge.resize(3 * tCount);
	for (int t = 0; t < tCount; t++){
		for (int j = 0; j < 3; j++) signedEdge.SetRowSize(3 * t + j, 1);
		for (int j = 0; j < 3; j++){
			signedEdge[3 * t + j][0] = MatrixEntry<Real, int>(reducedEdgeIndex[3 * t + j], positiveOrientedEdge[3 * t + j] ? 1 : -1);
		}
	}
	exteriorOperators.signedEdge = signedEdge;
	
	flowData.signedEdge = signedEdge;
	flowData.signedEdge_transpose = signedEdge.transpose();
	flowData.hodgeLaplacian = hodgeLaplacian;
	flowData.reducedEdgeIndex = reducedEdgeIndex;
	flowData.expandedEdgeIndex = expandedEdgeIndex;
	flowData.positiveOrientedEdge = positiveOrientedEdge;
	flowData.reduced_eFlowField.resize(eCount,0);
	flowData.eFlowField.resize(3 * tCount,0);

	//Whitney Prolongation
	SparseMatrix<Real, int> whitneyProlongation;
	whitneyProlongation.resize(2 * tCount);
	for (int t = 0; t < tCount; t++){
		whitneyProlongation.SetRowSize(2 * t, 3);
		whitneyProlongation.SetRowSize(2 * t + 1, 3);
		if (!flowData.mesh.g[t].determinant()){
			fprintf(stderr, "[WARNING] Vanishing metric tensor determinant\n");
		}
		SquareMatrix< Real, 2 > iTensor = flowData.mesh.g[t].inverse();
		for (int k = 0; k < 3; k++){
			Point2D<Real> gradDiff = iTensor*((grad[(k + 2) % 3] - grad[(k + 1) % 3])/3.0);
			if (!positiveOrientedEdge[3 * t + k]) gradDiff *= -1;
			whitneyProlongation[2 * t][k] = MatrixEntry<Real, int>(reducedEdgeIndex[3 * t + k], gradDiff[0]);
			whitneyProlongation[2 * t + 1][k] = MatrixEntry<Real, int>(reducedEdgeIndex[3 * t + k], gradDiff[1]);
		}
	}

	flowData.whitneyProlongation = whitneyProlongation;
	flowData.whitneyRestriction = whitneyProlongation.transpose();
	flowData.whitneyVFSolver = new EigenCholeskySolverLLt(flowData.hodgeLaplacian, true);
	flowData.whitneyMass = flowData.whitneyProlongation.transpose() * flowData.mesh.vectorFieldMassMatrix()* flowData.whitneyProlongation;
	exteriorOperators.vfMass = flowData.whitneyProlongation.transpose() * flowData.mesh.vectorFieldMassMatrix()* flowData.whitneyProlongation;

	if (Verbose.set) printf("Setting Exterior Operators : %.2f (s)\n", tm.elapsed());
}

template< class Real>
void UpdateGradientRestrictionAndProlongation(const int vCount, const SparseMatrix< Real, int > & restriction, SparseMatrix< Real, int > & gradientAndJGradientRestriction, SparseMatrix< Real, int > & gradientAndJGradientProlongation){
	size_t dim = restriction.rows;
	gradientAndJGradientRestriction.resize(2 * dim);
	for (int k = 0; k < dim; k++){
		gradientAndJGradientRestriction.SetRowSize(k, restriction.rowSizes[k]);
		gradientAndJGradientRestriction.SetRowSize(k + dim, restriction.rowSizes[k]);
		for (int j = 0; j < restriction.rowSizes[k]; j++){
			gradientAndJGradientRestriction[k][j] = restriction[k][j];
			gradientAndJGradientRestriction[k + dim][j] = MatrixEntry< Real, int >(restriction[k][j].N + (int)vCount, restriction[k][j].Value);
		}
	}
	gradientAndJGradientProlongation = gradientAndJGradientRestriction.transpose();
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
void WhitneyFlowViewer< Real, Channels>::Advance(){
	Timer t;

	//Update Vector Field And Flow
	EstimateFlow(flowData, scalarSmoothWeight, vectorFieldSmoothWeight);
	if (Verbose.set) printf("Got flow: %.2f (s)\n", t.elapsed());
	int tCount = sv.triangles.size();
	
	for (int t = 0; t < tCount; t++) sv.vectorField[t] = (sv.vertices[sv.triangles[t][1]] - sv.vertices[sv.triangles[t][0]])*flowData.tFlowField[t][0] + (sv.vertices[sv.triangles[t][2]] - sv.vertices[sv.triangles[t][0]])*flowData.tFlowField[t][1];
	inputGeometryData.flow(flowData, (Real)0.5, inputAdvectedSignal, Threads.value);

	scalarSmoothWeight *= ScalarWeightMultiplier.value;
	vectorFieldSmoothWeight = vectorFieldSmoothWeight *  VectorFieldWeightMultiplier.value > VectorFieldSmoothWeightThreshold.value ? vectorFieldSmoothWeight *  VectorFieldWeightMultiplier.value : vectorFieldSmoothWeight;

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
			

			//char ** comments = new char *[1];
			//char atlas_comment[256];
			//sprintf(atlas_comment, "TextureFile %s", "A.png");
			//comments[0] = atlas_comment;
			//PlyWritePolygons("newMesh.ply", _vertices, faces, PlyVertex< float >::WriteProperties, PlyVertex< float >::WriteComponents, PlyTexturedFace< float >::WriteProperties, PlyTexturedFace< float >::WriteComponents, PLY_ASCII, comments, 1);


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
		flowData.grad = flowData.mesh.gradientMatrix(FEM::RiemannianMesh< Real >::HAT_GRADIENT);
		flowData.vfMass = flowData.mesh.gradientMassMatrix(flowData.gradientType);
		flowData.vfStiffness = flowData.mesh.gradientStiffnessMatrix(flowData.gradientType);
		flowData.gradientAndJGradient = flowData.mesh.gradientMatrix(flowData.gradientType);
		flowData.gradientAndJGradient_transpose = flowData.gradientAndJGradient.transpose();
		flowData.vfM = flowData.vfStiffness + flowData.gradientAndJGradient_transpose * flowData.mesh.vectorFieldMassMatrix() * flowData.gradientAndJGradient;
		flowData.vfSolver = new EigenCholeskySolverLDLt(flowData.vfM, true);
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
		flowData.tFlowField.resize(triangles.size()), flowData.vFlowField.resize(DivergenceFree.set ? vertices.size() : vertices.size() * 2, (Real)0);
		flowData.dTFlowField.resize(triangles.size()), flowData.dVFlowField.resize(DivergenceFree.set ? vertices.size() : vertices.size() * 2);
	}

	{
		SetExteriorOperators();
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
	sv.callBacks.push_back(Visualization::KeyboardCallBack(&sv, 'r', "repeat level", RepeatLevelCallBack));

	sv.callBacks.push_back(Visualization::KeyboardCallBack(&sv, 'j', "scalar smooth weight", "Value", ScalarSmoothWeightCallBack));
	sv.callBacks.push_back(Visualization::KeyboardCallBack(&sv, 'J', "vector smooth weight", "Value", VectorFieldSmoothWeightCallBack));
	sv.callBacks.push_back(Visualization::KeyboardCallBack(&sv, 'o', "export result", "File name",OutputResultCallBack));

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
	//sv.updateMesh(false);
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
    flowData.old_tFlowField = flowData.tFlowField;
	
	Timer t;
	EstimateFlowWhitneyBasis(flowData, scalarSmoothWeight, vectorFieldSmoothWeight);
	if (Verbose.set) printf("Got flow: %.2f (s)\n", t.elapsed());

	int tCount = sv.triangles.size();
#if BARICENTRIC_WHITNEY
	for (int t = 0; t < tCount; t++) sv.vectorField[t] = (sv.vertices[sv.triangles[t][1]] - sv.vertices[sv.triangles[t][0]])*flowData.tFlowField[t][0] + (sv.vertices[sv.triangles[t][2]] - sv.vertices[sv.triangles[t][0]])*flowData.tFlowField[t][1];
#else
	std::vector< Real > tFlowField;
	tFlowField.resize(2 * tCount);
	flowData.whitneyProlongation.Multiply(GetPointer(flowData.reduced_eFlowField), GetPointer(tFlowField));
	for (int t = 0; t < tCount; t++) sv.vectorField[t] = (sv.vertices[sv.triangles[t][1]] - sv.vertices[sv.triangles[t][0]])*tFlowField[2 * t] + (sv.vertices[sv.triangles[t][2]] - sv.vertices[sv.triangles[t][0]])*tFlowField[2 * t + 1];
#endif

	if(!sv.useTexture)inputGeometryData.flow(flowData, (Real)0.5, inputAdvectedSignal, Threads.value);
	else inputTextureData.flow(flowData, (Real)0.5, inputAdvectedTexture, Threads.value);

	scalarSmoothWeight *= ScalarWeightMultiplier.value;
	vectorFieldSmoothWeight = vectorFieldSmoothWeight *  VectorFieldWeightMultiplier.value > VectorFieldSmoothWeightThreshold.value ? vectorFieldSmoothWeight *  VectorFieldWeightMultiplier.value : vectorFieldSmoothWeight;
	sprintf(sv.info[2], "Smooth Weights(Scalar - Vector Field): %g - %g", scalarSmoothWeight, vectorFieldSmoothWeight);

	signalMode = SIGNAL_BLEND;
	if(!sv.useTexture)UpdateSignalVisualization();
	else UpdateTextureVisualization();
	glutSetCursor(GLUT_CURSOR_INHERIT);
}

template< class Real, int Channels >
void WhitneyFlowViewer< Real, Channels>::RepeatLevelCallBack(Visualization* v, const char* prompt) {
	glutSetCursor(GLUT_CURSOR_WAIT);
	flowData.old_tFlowField = flowData.tFlowField;

	Timer t;
	EstimateFlowWhitneyBasis(flowData, scalarSmoothWeight, vectorFieldSmoothWeight);
	if (Verbose.set) printf("Got flow: %.2f (s)\n", t.elapsed());

	int tCount = sv.triangles.size();
#if BARICENTRIC_WHITNEY
	for (int t = 0; t < tCount; t++) sv.vectorField[t] = (sv.vertices[sv.triangles[t][1]] - sv.vertices[sv.triangles[t][0]])*flowData.tFlowField[t][0] + (sv.vertices[sv.triangles[t][2]] - sv.vertices[sv.triangles[t][0]])*flowData.tFlowField[t][1];
#else
	std::vector< Real > tFlowField;
	tFlowField.resize(2 * tCount);
	flowData.whitneyProlongation.Multiply(GetPointer(flowData.reduced_eFlowField), GetPointer(tFlowField));
	for (int t = 0; t < tCount; t++) sv.vectorField[t] = (sv.vertices[sv.triangles[t][1]] - sv.vertices[sv.triangles[t][0]])*tFlowField[2 * t] + (sv.vertices[sv.triangles[t][2]] - sv.vertices[sv.triangles[t][0]])*tFlowField[2 * t + 1];
#endif

	if (!sv.useTexture)inputGeometryData.flow(flowData, (Real)0.5, inputAdvectedSignal, Threads.value);
	else inputTextureData.flow(flowData, (Real)0.5, inputAdvectedTexture, Threads.value);

	sprintf(sv.info[2], "Smooth Weights(Scalar - Vector Field): %g - %g", scalarSmoothWeight, vectorFieldSmoothWeight);

	signalMode = SIGNAL_BLEND;
	if (!sv.useTexture)UpdateSignalVisualization();
	else UpdateTextureVisualization();
	glutSetCursor(GLUT_CURSOR_INHERIT);
}

template< class Real, int Channels >
void WhitneyFlowViewer< Real, Channels>::IterativeOptimization(){
	for (int i = 0; i < Levels.value; i++){
		Timer t;
		EstimateFlowWhitneyBasis(flowData, scalarSmoothWeight, vectorFieldSmoothWeight);
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
	WhitneyFlowViewer< Real, Channels >::vectorFieldSmoothWeight = (Real)VectorFieldSmoothWeight.value;
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
