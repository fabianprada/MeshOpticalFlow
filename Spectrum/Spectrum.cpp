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
#include <Src/VectorLaplacianSpectrum.inl>
#include <Src/VectorIO.h>
#include <Src/MetricFace.h>


cmdLineParameter< char* > Mesh("mesh");
cmdLineParameter< int > VectorFieldMode("vfMode", WHITNEY_VECTOR_FIELD), ConnectionMode("cMode", PROJECTED_BARICENTRIC_WEIGHTS), NumEigenVector("eigenVectors",20);
cmdLineParameter< float > SubdivideEdgeLength("eLength", 0);
cmdLineReadable MetricFromEdgeLength("edgeMetric");
cmdLineReadable* params[] = { &Mesh, &VectorFieldMode, &ConnectionMode, &SubdivideEdgeLength,&MetricFromEdgeLength, NULL };

void ShowUsage(const char* ex)
{
	printf("Usage %s:\n", ex);
	printf("I/O Parameters: \n");
	printf("\t[--%s <input geometry (.ply)>]\n", Mesh.name);

	printf("Processing Parameters: \n");
	printf("\t[--%s <number of eigenvector> = %02d]\n", NumEigenVector.name, NumEigenVector.value);
	printf("\t[--%s <subdivide edges up to this diagonal fraction> = %0.3f]\n", SubdivideEdgeLength.name, SubdivideEdgeLength.value);
	printf("\t[--%s <metric from edge length>]\n", MetricFromEdgeLength.name);
	printf("Vector Field Parameters: \n");
	printf("\t[--%s <vector field mode >=%d]\n", VectorFieldMode.name, VectorFieldMode.value);
	printf("\t \t [%d] Whitney \n", WHITNEY_VECTOR_FIELD);
	printf("\t \t [%d] Conformal \n", CONFORMAL_VECTOR_FIELD);
	printf("\t \t [%d] Connection \n", CONNECTION_VECTOR_FIELD);
	printf("\t[--%s <connection mode >=%d]\n", ConnectionMode.name, ConnectionMode.value);
	printf("\t \t [%d] Projected baricentric \n", PROJECTED_BARICENTRIC_WEIGHTS);
	printf("\t \t [%d] Baricentric dual \n", BARICENTRIC_WEIGHTS);
	printf("\t \t [%d] Inverse cotangents \n", INVERSE_COTANGENT_WEIGHTS);
}

#include <GL/glew.h>
#include <GL/glut.h>
#include <Src/SurfaceVisualization.inl>

struct SpectrumViewer
{
	static SurfaceVisualization sv;
	static FEM::RiemannianMesh< double > mesh;
	static VectorField<double> * vf;
	
	static int Init();
	static void Idle(void);
	static void KeyboardFunc(unsigned char key, int x, int y);
	static void SpecialFunc(int key, int x, int y);
	static void Display(void);
	static void Reshape(int w, int h);
	static void MouseFunc(int button, int state, int x, int y);
	static void MotionFunc(int x, int y);

	static void NextEigenVectorCallBack(Visualization* v, const char* prompt);
	static void PreviousEigenVectorCallBack(Visualization* v, const char* prompt);
	static std::vector<std::vector<Point2D<double>>> laplaceEigenVectors;
	static int currentEigenvector;
};

 SurfaceVisualization				SpectrumViewer::sv;

 FEM::RiemannianMesh< double >		SpectrumViewer::mesh;
 VectorField<double> *				SpectrumViewer::vf;


 std::vector<std::vector<Point2D<double>>>	SpectrumViewer::laplaceEigenVectors;
 int										SpectrumViewer::currentEigenvector = 0;

 void SpectrumViewer::Idle(void){ sv.Idle(); }
 void SpectrumViewer::KeyboardFunc(unsigned char key, int x, int y){ sv.KeyboardFunc(key, x, y); }
 void SpectrumViewer::SpecialFunc(int key, int x, int y){ sv.SpecialFunc(key, x, y); }
 void SpectrumViewer::Display(void){ sv.Display(); }
 void SpectrumViewer::Reshape(int w, int h){ sv.Reshape(w, h); }
 void SpectrumViewer::MouseFunc(int button, int state, int x, int y){ sv.MouseFunc(button, state, x, y); }
 void SpectrumViewer::MotionFunc(int x, int y){ sv.MotionFunc(x, y); }



void SpectrumViewer::NextEigenVectorCallBack(Visualization* v, const char* prompt){
	glutSetCursor(GLUT_CURSOR_WAIT);
	currentEigenvector = currentEigenvector <  laplaceEigenVectors.size() - 1 ? currentEigenvector + 1 : 0;
	printf("Eigenvector %d of %d \n", currentEigenvector + 1, laplaceEigenVectors.size());
	int tCount = sv.triangles.size();
	for (int t = 0; t < tCount; t++) sv.vectorField[t] = (sv.vertices[sv.triangles[t][1]] - sv.vertices[sv.triangles[t][0]])*laplaceEigenVectors[currentEigenvector][t][0] + (sv.vertices[sv.triangles[t][2]] - sv.vertices[sv.triangles[t][0]])*laplaceEigenVectors[currentEigenvector][t][1];
	glutSetCursor(GLUT_CURSOR_INHERIT);
}


void SpectrumViewer::PreviousEigenVectorCallBack(Visualization* v, const char* prompt) {
	glutSetCursor(GLUT_CURSOR_WAIT);
	currentEigenvector = currentEigenvector > 0 ? currentEigenvector - 1 :  laplaceEigenVectors.size() - 1;
	printf("Eigenvector %d of %d \n", currentEigenvector + 1, laplaceEigenVectors.size());
	int tCount = sv.triangles.size();
	for (int t = 0; t < tCount; t++) sv.vectorField[t] = (sv.vertices[sv.triangles[t][1]] - sv.vertices[sv.triangles[t][0]])*laplaceEigenVectors[currentEigenvector][t][0] + (sv.vertices[sv.triangles[t][2]] - sv.vertices[sv.triangles[t][0]])*laplaceEigenVectors[currentEigenvector][t][1];
	glutSetCursor(GLUT_CURSOR_INHERIT);
}

int SpectrumViewer::Init(){


	std::vector< PlyVertex< double > > _vertices;
	std::vector<double> edgeLengths;
	std::vector< TriangleIndex > triangles;
	int file_type;
	if (MetricFromEdgeLength.set) {
		std::vector< PlyMetricFace< double > > _faces;
		if (!PlyReadPolygons(Mesh.value, _vertices, _faces, PlyVertex< double >::ReadProperties, NULL, PlyVertex< double >::ReadComponents, PlyMetricFace<double>::Properties, NULL, PlyMetricFace<double>::Components, file_type)) {
			printf("Unable to read %s. Check the file name and format!", Mesh.value);
			return 0;
		}
		triangles.resize(_faces.size());
		for (int i = 0; i<_faces.size(); i++) triangles[i] = TriangleIndex(_faces[i][0], _faces[i][1], _faces[i][2]);
		edgeLengths.resize(3 * triangles.size());
		for (int i = 0; i<triangles.size(); i++) for (int j = 0; j<3; j++) edgeLengths[3 * i + j] = sqrt(_faces[i].square_length(j));
	}
	else {
		if(!PlyReadTriangles(Mesh.value, _vertices, triangles, PlyVertex< double >::ReadProperties, NULL, PlyVertex< double >::ReadComponents, file_type)){
			printf("Unable to read %s. Check the file name and format!", Mesh.value);
			return 0;
		}
	}

	std::vector< Point3D< double > > vertices(_vertices.size());
	for (int i = 0; i<vertices.size(); i++) vertices[i] = Point3D< double >(_vertices[i]);
	
	mesh.triangles = GetPointer(triangles);
	mesh.tCount = triangles.size();
	if (MetricFromEdgeLength.set) mesh.setMetricFromEdgeLengths(GetPointer(edgeLengths));
	else mesh.setMetricFromEmbedding(GetPointer(vertices));
	mesh.makeUnitArea();
	mesh.setInverseMetric();
	
	if (VectorFieldMode.value == WHITNEY_VECTOR_FIELD) vf = new WhitneyVectorField<double>();
	else if (VectorFieldMode.value == CONFORMAL_VECTOR_FIELD) vf = new ConformalVectorField<double>();
	else if (VectorFieldMode.value == CONNECTION_VECTOR_FIELD) vf = new ConnectionVectorField<double>(ConnectionMode.value);
	else{
		printf("ERROR: Unsupported vector field! \n");
		return 0;
	}
	vf->Init(mesh);
	ComputeSpectrum(vf, mesh, NumEigenVector.value,laplaceEigenVectors);
	for (int i = 0; i <  NumEigenVector.value; i++){
		char vectorName[256];
		sprintf(vectorName, "eigenvector-%03d.bin", i + 1);
		WriteVector(laplaceEigenVectors[i], vectorName);
	}

	int vCount = vertices.size();
	sv.vertices.resize(vCount);
	for (int v = 0; v < vCount; v++) sv.vertices[v] = Point3D<float>(vertices[v]);
	sv.triangles = triangles;
	sv.colors.resize(vCount);
	sv.vectorField.resize(sv.triangles.size());
	int tCount = sv.triangles.size();
	
#if 1
	Point3D< float > center;
	float area = 0.f;
	for (int i = 0; i < triangles.size(); i++)
	{
		Point3D< float > n = Point3D< float >::CrossProduct(vertices[triangles[i][1]] - vertices[triangles[i][0]], vertices[triangles[i][2]] - vertices[triangles[i][0]]);
		Point3D< float > c = (vertices[triangles[i][0]] + vertices[triangles[i][1]] + vertices[triangles[i][2]]) / 3.f;
		float a = (float)Length(n);
		center += c*a, area += a;
	}
	center /= area;
	float max = 0.f;
	for (int i = 0; i < vertices.size(); i++) max = std::max< float >(max, (float)Point3D< float >::Length(vertices[i] - center));

	for (int t = 0; t < tCount; t++) sv.vectorField[t] = (sv.vertices[sv.triangles[t][1]] - sv.vertices[sv.triangles[t][0]])*laplaceEigenVectors[currentEigenvector][t][0] + (sv.vertices[sv.triangles[t][2]] - sv.vertices[sv.triangles[t][0]])*laplaceEigenVectors[currentEigenvector][t][1];
	for (int t = 0; t < tCount; t++) sv.vectorField[t] = (sv.vectorField[t] -center ) / max;
#else
	for (int t = 0; t < tCount; t++) sv.vectorField[t] = (sv.vertices[sv.triangles[t][1]] - sv.vertices[sv.triangles[t][0]])*laplaceEigenVectors[currentEigenvector][t][0] + (sv.vertices[sv.triangles[t][2]] - sv.vertices[sv.triangles[t][0]])*laplaceEigenVectors[currentEigenvector][t][1];
#endif
	for (int v = 0; v < vCount; v++) sv.colors[v] = Point3D<float>(0.8,0.8,0.8);

	sv.callBacks.push_back(Visualization::KeyboardCallBack(&sv, 'b', "previous eigenvector", PreviousEigenVectorCallBack));
	sv.callBacks.push_back(Visualization::KeyboardCallBack(&sv, 'n', "next eigenvector", NextEigenVectorCallBack));
	sv.useTexture = false;
	sv.showVectors = true;
	sv.vectorScale = 0.01;
	sv.useLight = true;
	return 1;
}

int main(int argc, char* argv[]){

	cmdLineParse(argc - 1, argv + 1, params);
	if (!Mesh.set){
		ShowUsage(argv[0]);
		return EXIT_FAILURE;
	}

	SurfaceVisualization& sv = SpectrumViewer::sv;
	if (!SpectrumViewer::Init()) return 0;
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutInitWindowSize(sv.screenWidth, sv.screenHeight);
	glutInit(&argc, argv);
	char windowName[1024];
	sprintf(windowName, "Spectrum");
	glutCreateWindow(windowName);

	if (glewInit() != GLEW_OK) fprintf(stderr, "[ERROR] glewInit failed\n"), exit(0);
	glutIdleFunc(SpectrumViewer::Idle);
	glutDisplayFunc(SpectrumViewer::Display);
	glutReshapeFunc(SpectrumViewer::Reshape);
	glutMouseFunc(SpectrumViewer::MouseFunc);
	glutMotionFunc(SpectrumViewer::MotionFunc);
	glutKeyboardFunc(SpectrumViewer::KeyboardFunc);
	glutSpecialFunc(SpectrumViewer::SpecialFunc);
	glutMainLoop();
	return EXIT_SUCCESS;
}
