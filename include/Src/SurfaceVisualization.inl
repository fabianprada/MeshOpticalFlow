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


#include <Misha/Timer.h>
#include <Misha/Visualization.h>
#include <Misha/Camera.h>
#include <Misha/Image.h>

#ifndef M_PI
#define M_PI		3.14159265358979323846
#endif // M_PI

struct SurfaceVisualization : public Visualization
{
	bool useTexture;
	GLuint textureBuffer;
	unsigned char* texture;
	int textureWidth;
	int textureHeight;
	GLuint coordinateBuffer;
	std::vector< Point2D< float > > textureCoordinates;
	void updateTextureBuffer( bool updateCoordinates = false);

	std::vector< TriangleIndex > triangles;
	std::vector< Point3D< float > > vertices , colors;
	std::vector< Point3D< float > > vectorField;
	bool showMesh;

	void SetupOffScreenBuffer();
	void RenderOffScreenBuffer(const char * fileName);
	void RenderOffScreenBuffer(Image<Point3D<float>> & colorBuffer, Image<float> & depthBuffer);
	GLuint offscreen_depth_texture;
	GLuint offscreen_color_texture;
	GLuint offscreen_framebuffer_handle;
	int offscreen_frame_width, offscreen_frame_height;


	float vectorScale , vectorCount;
	Camera camera;
	float zoom;
	Point3D< float > translate;
	float scale;
	GLuint vbo , ebo;
	GLfloat lightAmbient[4] , lightDiffuse[4] , lightSpecular[4] , shapeSpecular[4] , shapeSpecularShininess;
	int oldX , oldY , newX , newY;
	float imageZoom , imageOffset[2];
	bool rotating , scaling , panning;
	bool useLight , showEdges , showVectors , hasColor;

	SurfaceVisualization( void );
	//bool init( const char* fileName , int subdivide=0 );
	void initMesh( void );
	void updateMesh( bool newPositions );
	bool select( int x , int y , Point3D< float >& p );

	void idle( void );
	void keyboardFunc( unsigned char key , int x , int y );
	void specialFunc( int key, int x, int y );
	void display( void );
	void mouseFunc( int button , int state , int x , int y );
	void motionFunc( int x , int y );

	static void WriteSceneConfigurationCallBack(Visualization* v, const char* prompt);
	static void ReadSceneConfigurationCallBack(Visualization* v, const char* prompt);
	static void     ToggleLightCallBack(Visualization* v, const char*) { ((SurfaceVisualization*)v)->useLight = !((SurfaceVisualization*)v)->useLight; }
	static void     ToggleEdgesCallBack( Visualization* v , const char* ){ ( (SurfaceVisualization*)v)->showEdges   = !( (SurfaceVisualization*)v)->showEdges;   }
	static void     ToggleVectorsCallBack( Visualization* v , const char* ){ ( (SurfaceVisualization*)v)->showVectors = !( (SurfaceVisualization*)v)->showVectors; }
	static void     SaveOffscreenBufferCallBack(Visualization* v, const char* prompt){
		SurfaceVisualization* sv = (SurfaceVisualization*)v;
		sv->RenderOffScreenBuffer(prompt);
	}
	bool setPosition( int x , int y , Point3D< double >& p );
	bool setPosition( int x , int y , Point3D< float >& p );
};


void SurfaceVisualization::WriteSceneConfigurationCallBack(Visualization* v, const char* prompt) {
	const SurfaceVisualization* av = (SurfaceVisualization*)v;
	FILE * file;
	file = fopen(prompt, "wb");
	fwrite(&av->camera.position, sizeof(Point3D<double>), 1, file);
	fwrite(&av->camera.forward, sizeof(Point3D<double>), 1, file);
	fwrite(&av->camera.right, sizeof(Point3D<double>), 1, file);
	fwrite(&av->camera.up, sizeof(Point3D<double>), 1, file);
	fwrite(&av->zoom, sizeof(float), 1, file);
	fclose(file);
}

void SurfaceVisualization::ReadSceneConfigurationCallBack(Visualization* v, const char* prompt) {
	SurfaceVisualization* av = (SurfaceVisualization*)v;
	FILE * file;
	file = fopen(prompt, "rb");
	if (!file) {
		printf("Camera Configuration File Not Valid \n");
	}
	else {
		fread(&av->camera.position, sizeof(Point3D<double>), 1, file);
		fread(&av->camera.forward, sizeof(Point3D<double>), 1, file);
		fread(&av->camera.right, sizeof(Point3D<double>), 1, file);
		fread(&av->camera.up, sizeof(Point3D<double>), 1, file);
		fread(&av->zoom, sizeof(float), 1, file);
		fclose(file);
	}
}


SurfaceVisualization::SurfaceVisualization( void )
{
	zoom = 1.05f;
	lightAmbient [0] = lightAmbient [1] = lightAmbient [2] = 0.25f , lightAmbient [3] = 1.f;
	lightDiffuse [0] = lightDiffuse [1] = lightDiffuse [2] = 0.70f , lightDiffuse [3] = 1.f;
	lightSpecular[0] = lightSpecular[1] = lightSpecular[2] = 1.00f , lightSpecular[3] = 1.f;
	shapeSpecular[0] = shapeSpecular[1] = shapeSpecular[2] = 1.00f , shapeSpecular[3] = 1.f;
	shapeSpecularShininess = 128;
	oldX , oldY , newX , newY;
	imageZoom = 1.f , imageOffset[0] = imageOffset[1] = 0.f;
	rotating = scaling = panning = false;
	useLight = false;
	showMesh = true;
	showEdges = showVectors = false;
	vectorScale = 1.f;
	vectorCount = 100000.f;
	offscreen_frame_width = offscreen_frame_height = 1024;
	vbo = ebo = 0;
	textureBuffer = coordinateBuffer = 0;



	
	callBacks.push_back(KeyboardCallBack(this, 'l', "toggle light", ToggleLightCallBack));
	callBacks.push_back( KeyboardCallBack( this , 'e' , "toggle edges" , ToggleEdgesCallBack ) );
	callBacks.push_back( KeyboardCallBack( this , 'v' , "toggle vectors" , ToggleVectorsCallBack ) );
	callBacks.push_back(KeyboardCallBack(this, 'K', "save screen", "File Name", SaveOffscreenBufferCallBack));
	callBacks.push_back(KeyboardCallBack(this, 'R', "read camera", "File Name", ReadSceneConfigurationCallBack));
	callBacks.push_back(KeyboardCallBack(this, 'W', "write camera", "File Name", WriteSceneConfigurationCallBack));
}


void SurfaceVisualization::SetupOffScreenBuffer(){
	// The depth buffer texture
	glGenTextures(1, &offscreen_depth_texture);
	glBindTexture(GL_TEXTURE_2D, offscreen_depth_texture);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH_COMPONENT24, offscreen_frame_width, offscreen_frame_height);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

	// The color buffer texture
	glGenTextures(1, &offscreen_color_texture);
	glBindTexture(GL_TEXTURE_2D, offscreen_color_texture);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, offscreen_frame_width, offscreen_frame_height);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);


	// Create and set up the FBO
	glGenFramebuffers(1, &offscreen_framebuffer_handle);
	glBindFramebuffer(GL_FRAMEBUFFER, offscreen_framebuffer_handle);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, offscreen_depth_texture, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, offscreen_color_texture, 0);
	GLenum drawBuffers[] = { GL_COLOR_ATTACHMENT0 };
	glDrawBuffers(1, drawBuffers);

	if (0) {
		GLenum result = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		if (result == GL_FRAMEBUFFER_COMPLETE) {
			printf("Framebuffer is complete.\n");
		}
		else {
			printf("Framebuffer is not complete.\n");
		}
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void SurfaceVisualization::RenderOffScreenBuffer(Image<Point3D<float>> & colorBuffer, Image<float> & depthBuffer){
	if (!offscreen_framebuffer_handle) SetupOffScreenBuffer();

	glViewport(0, 0, offscreen_frame_width, offscreen_frame_height);
	glBindFramebuffer(GL_FRAMEBUFFER, offscreen_framebuffer_handle);
	display();
	glFlush();

	//Save color buffer to image
	Pointer(float) GLColorBuffer = AllocPointer< float >(sizeof(float)* 3 * offscreen_frame_width * offscreen_frame_height);
	glReadBuffer(GL_COLOR_ATTACHMENT0);
	glReadPixels(0, 0, offscreen_frame_width, offscreen_frame_height, GL_RGB, GL_FLOAT, GLColorBuffer);
	glFinish();
	colorBuffer.resize(offscreen_frame_width, offscreen_frame_height);
	for (int j = 0; j<offscreen_frame_height; j++) for (int i = 0; i<offscreen_frame_width; i++) for (int c = 0; c<3; c++){
		colorBuffer(i, j)[c] = GLColorBuffer[c + i * 3 + (offscreen_frame_height - 1 - j) * offscreen_frame_width * 3];
	}
	FreePointer(GLColorBuffer);

	Pointer(float) GLDepthBuffer = AllocPointer< float >(sizeof(float)* offscreen_frame_width * offscreen_frame_height);
	glReadPixels(0, 0, offscreen_frame_width, offscreen_frame_height, GL_DEPTH_COMPONENT, GL_FLOAT, GLDepthBuffer);
	glFinish();
	depthBuffer.resize(offscreen_frame_width, offscreen_frame_height);
	for (int j = 0; j < offscreen_frame_height; j++) for (int i = 0; i<offscreen_frame_width; i++) depthBuffer(i, j) = GLDepthBuffer[(offscreen_frame_height - 1 - j)*offscreen_frame_width + i];
	FreePointer(GLDepthBuffer);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, screenWidth, screenHeight);
}

void SurfaceVisualization::RenderOffScreenBuffer(const char * fileName){
	if (!offscreen_framebuffer_handle) SetupOffScreenBuffer();

	glViewport(0, 0, offscreen_frame_width, offscreen_frame_height);
	glBindFramebuffer(GL_FRAMEBUFFER, offscreen_framebuffer_handle);
	glClearColor(1, 1, 1, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	display();
	glFlush();

	Pointer(float) pixels = AllocPointer< float >(sizeof(float)* 3 * offscreen_frame_width * offscreen_frame_height);
	Pointer(unsigned char) _pixels = AllocPointer< unsigned char >(sizeof(unsigned char)* 3 * offscreen_frame_width * offscreen_frame_height);
	glReadBuffer(GL_COLOR_ATTACHMENT0);
	glReadPixels(0, 0, offscreen_frame_width, offscreen_frame_height, GL_RGB, GL_FLOAT, pixels);
	glFinish();
	for (int j = 0; j<offscreen_frame_height; j++) for (int i = 0; i<offscreen_frame_width; i++) for (int c = 0; c<3; c++)
	{
		int ii = int(pixels[c + i * 3 + (offscreen_frame_height - 1 - j) * offscreen_frame_width * 3] * 256);
		if (ii<  0) ii = 0;
		if (ii>255) ii = 255;
		_pixels[c + i * 3 + j * offscreen_frame_width * 3] = static_cast<unsigned char>(ii);
	}
	FreePointer(pixels);
	char* ext = GetFileExtension(fileName);
	if (!strcasecmp(ext, "png"))                        PNGWriteColor(fileName, _pixels, offscreen_frame_width, offscreen_frame_height);
	else fprintf(stderr, "[WARNING] Unrecognized file extension: %s\n", ext);
	delete[] ext;
	FreePointer(_pixels);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, screenWidth, screenHeight);
}

bool SurfaceVisualization::setPosition( int x , int y , Point3D< double >& p )
{
	double _x =(double)x / screenWidth - 0.5 , _y = 1. - (double)y/screenHeight - 0.5;
	_x *= 2. , _y *= 2;
	_x *= zoom , _y *= zoom;
	double r = _x*_x + _y*_y;
	if( r<1 )
	{
		p = camera.forward * ( -sqrt( 1-r ) ) + camera.right * _x + camera.up * _y;
		return true;
	}
	return false;
}
bool SurfaceVisualization::setPosition( int x , int y , Point3D< float >& p )
{
	Point3D< double > _p;
	bool ret = setPosition( x , y , _p );
	p = Point3D< float >( (float)_p[0] , (float)_p[1] , (float)_p[2] );
	return ret;
}

void SurfaceVisualization::updateTextureBuffer( bool updateCoordinates){


	if (!glIsBuffer(coordinateBuffer)){
		glGenBuffers(1, &coordinateBuffer);
	}

	if (updateCoordinates){
		glBindBuffer(GL_ARRAY_BUFFER, coordinateBuffer);
		glBufferData(GL_ARRAY_BUFFER, 3 * triangles.size() * sizeof(Point2D<float>), &textureCoordinates[0], GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	if (!glIsBuffer(textureBuffer)){
		glGenTextures(1, &textureBuffer);
	}

	glBindTexture(GL_TEXTURE_2D, textureBuffer);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, textureWidth,textureHeight, 0, GL_RGB,GL_UNSIGNED_BYTE, (GLvoid*)texture);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void SurfaceVisualization::updateMesh( bool newPositions )
{
	Point3D< float > *_vertices = new Point3D< float >[ triangles.size()*9 ];
	Point3D< float > *_normals = _vertices + triangles.size()*3;
	Point3D< float > *_colors = _vertices + triangles.size()*6;

	Point3D< float > center;
	float area = 0.f;
	for( int i=0 ; i<triangles.size() ; i++ )
	{
		Point3D< float > n = Point3D< float >::CrossProduct( vertices[ triangles[i][1] ] - vertices[ triangles[i][0] ] , vertices[ triangles[i][2] ] - vertices[ triangles[i][0] ] );
		Point3D< float > c = ( vertices[ triangles[i][0] ] + vertices[ triangles[i][1] ] + vertices[ triangles[i][2] ] ) / 3.f;
		float a = (float)Length(n);
		center += c*a , area += a;
	}
	center /= area;
	float max = 0.f;
	for( int i=0 ; i<vertices.size() ; i++ ) max = std::max< float >( max , (float)Point3D< float >::Length( vertices[i]-center ) );

	for( int i=0 ; i<triangles.size() ; i++ )
	{
		Point3D< float > n = Point3D< float >::CrossProduct( vertices[ triangles[i][1] ] - vertices[ triangles[i][0] ] , vertices[ triangles[i][2] ] - vertices[ triangles[i][0] ] );
		n /= Length( n );
		for( int j=0 ; j<3 ; j++ )
		{
			_vertices[3*i+j] = ( vertices[ triangles[i][j] ] - center ) / max;
			_colors[3*i+j] = colors[ triangles[i][j] ];
			_normals[3*i+j] = n;
		}
	}
	if( newPositions )
	{
		translate = -center;
		scale = 1.f/max;
		for( int i=0 ; i<vertices.size() ; i++ ) vertices[i] = ( vertices[i]+translate ) * scale;
	}
	glBindBuffer( GL_ARRAY_BUFFER , vbo );
	glBufferData( GL_ARRAY_BUFFER , 9 * triangles.size() * sizeof( Point3D< float > ) , _vertices , GL_DYNAMIC_DRAW );
	glBindBuffer( GL_ARRAY_BUFFER , 0 );

	delete[] _vertices;
	glutPostRedisplay();
}
void SurfaceVisualization::initMesh( void )
{
	TriangleIndex *_triangles = new TriangleIndex[ triangles.size() ];

	for( int i=0 ; i<triangles.size() ; i++ ) for( int j=0 ; j<3 ; j++ ) _triangles[i][j] = 3*i+j;

	glGenBuffers( 1 , &ebo );
	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER , ebo );
	glBufferData( GL_ELEMENT_ARRAY_BUFFER , triangles.size() * sizeof( int ) * 3 , _triangles , GL_STATIC_DRAW );
	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER , 0 );

	glGenBuffers(1, &vbo);

	updateMesh( true );

	delete[] _triangles;
}
bool SurfaceVisualization::select( int x , int  y , Point3D< float >& out )
{
	bool ret = false;
	Pointer( float ) depthBuffer = AllocPointer< float >( sizeof(float) * screenWidth * screenHeight );
	glReadPixels( 0 , 0 , screenWidth , screenHeight , GL_DEPTH_COMPONENT , GL_FLOAT , depthBuffer );
	float ar = (float)screenWidth/(float)screenHeight ;
	float _screenWidth , _screenHeight;
	if( screenWidth>screenHeight ) _screenWidth =  screenWidth * ar , _screenHeight = screenHeight;
	else                           _screenWidth =  screenWidth , _screenHeight = screenHeight / ar;
	{
		double _x =(double)x/screenWidth - 0.5 , _y = 1. - (double)y/screenHeight - 0.5 , _z;
		if( screenWidth>screenHeight ) _x *= zoom*ar , _y *= zoom;
		else                           _x *= zoom , _y *= zoom/ar;
		_x *= 2. , _y *= 2;
		int x1 = (int)floor(x) , y1 = (int)floor(y) , x2 = x1+1 , y2 = y1+1;
		float dx = x-x1 , dy = y-y1;
		x1 = std::max< int >( 0.f , std::min< int >( x1 , screenWidth -1 ) );
		y1 = std::max< int >( 0.f , std::min< int >( y1 , screenHeight-1 ) );
		x2 = std::max< int >( 0.f , std::min< int >( x2 , screenWidth -1 ) );
		y2 = std::max< int >( 0.f , std::min< int >( y2 , screenHeight-1 ) );
		_z =
			depthBuffer[ (screenHeight-1-y1)*screenWidth+x1 ] * (1.f-dx) * (1.f-dy) +
			depthBuffer[ (screenHeight-1-y1)*screenWidth+x2 ] * (    dx) * (1.f-dy) +
			depthBuffer[ (screenHeight-1-y2)*screenWidth+x1 ] * (1.f-dx) * (    dy) +
			depthBuffer[ (screenHeight-1-y2)*screenWidth+x2 ] * (    dx) * (    dy) ;
		if( _z<1 ) out = Point3D< float >( camera.forward * ( -1.5 + 3. * _z ) + camera.right * _x + camera.up * _y + camera.position ) , ret = true;
	}
	FreePointer( depthBuffer );
	return ret;
}
void SurfaceVisualization::display( void )
{
	if( !vbo && !ebo ) initMesh();
	if (useTexture && !textureBuffer && !coordinateBuffer) updateTextureBuffer(true);
	glEnable( GL_CULL_FACE );

	//useTexture = false;
	//useLight = true;
	
	
	glMatrixMode( GL_PROJECTION );
	glLoadIdentity();
	float ar = (float)screenWidth/(float)screenHeight , ar_r = 1.f/ar;
	if( screenWidth>screenHeight ) glOrtho( -ar*zoom , ar*zoom , -zoom , zoom , -1.5 , 1.5 );
	else                           glOrtho( -zoom , zoom , -ar_r*zoom , ar_r*zoom , -1.5 , 1.5 );
	glMatrixMode( GL_MODELVIEW );

	glLoadIdentity();

	camera.draw();

	GLfloat lPosition[4];

	{
		Point3D< float > d = camera.up + camera.right - camera.forward*5;
		lPosition[0] = d[0] , lPosition[1] = d[1] , lPosition[2] = d[2];
	}
	lPosition[3] = 0.0;
	glLightModeli( GL_LIGHT_MODEL_LOCAL_VIEWER , GL_FALSE );
	glLightModeli( GL_LIGHT_MODEL_TWO_SIDE , GL_TRUE );
	glLightfv( GL_LIGHT0 , GL_AMBIENT , lightAmbient );
	glLightfv( GL_LIGHT0 , GL_DIFFUSE , lightDiffuse );
	glLightfv( GL_LIGHT0 , GL_SPECULAR , lightSpecular );
	glLightfv( GL_LIGHT0 , GL_POSITION , lPosition );
	glEnable( GL_LIGHT0 );
	if( useLight ) glEnable ( GL_LIGHTING );
	else           glDisable( GL_LIGHTING );
	glColorMaterial( GL_FRONT_AND_BACK , GL_AMBIENT_AND_DIFFUSE );
	glEnable( GL_COLOR_MATERIAL );

	glEnable( GL_DEPTH_TEST );
	glMaterialfv( GL_FRONT_AND_BACK , GL_SPECULAR  , shapeSpecular );
	glMaterialf ( GL_FRONT_AND_BACK , GL_SHININESS , shapeSpecularShininess );

	if (showMesh){
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glEnableClientState(GL_VERTEX_ARRAY);
		glVertexPointer(3, GL_FLOAT, 0, NULL);

		if (useLight) {
			glNormalPointer(GL_FLOAT, 0, (GLubyte*)NULL + sizeof(Point3D< float >) * triangles.size() * 3);
			glColor3f(0.75f, 0.75f, 0.75f);
			glEnableClientState(GL_NORMAL_ARRAY);
		}
		else {
			if (!useTexture) {
				glColorPointer(3, GL_FLOAT, 0, (GLubyte*)NULL + sizeof(Point3D< float >) * triangles.size() * 6);
				glEnableClientState(GL_COLOR_ARRAY);
			}
			else {
				glBindBuffer(GL_ARRAY_BUFFER, coordinateBuffer);
				glEnableClientState(GL_TEXTURE_COORD_ARRAY);
				glTexCoordPointer(2, GL_FLOAT, 0, NULL);

				glEnable(GL_TEXTURE_2D);
				glBindTexture(GL_TEXTURE_2D, textureBuffer);
				glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			}
		}

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
		glDrawElements(GL_TRIANGLES, (GLsizei)(triangles.size() * 3), GL_UNSIGNED_INT, NULL);

		if (useLight) {
			glDisableClientState(GL_NORMAL_ARRAY);
		}
		else {
			if (!useTexture) {
				glDisableClientState(GL_COLOR_ARRAY);
			}
			else {
				glDisableClientState(GL_TEXTURE_COORD_ARRAY);
				glDisable(GL_TEXTURE_2D);
				glBindTexture(GL_TEXTURE_2D, 0);
			}
		}
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}

	if( showEdges )
	{
		GLint src , dst;
		glGetIntegerv( GL_BLEND_SRC , &src );
		glGetIntegerv( GL_BLEND_DST , &dst );
		Point3D< float > f = camera.forward / 256;
		glPushMatrix();
		glTranslatef( -f[0] , -f[1] , -f[2] );
		glColor3f( 0.125 , 0.125 , 0.125 );
		glBlendFunc( GL_SRC_ALPHA , GL_ONE_MINUS_SRC_ALPHA );
		glEnable( GL_BLEND );
		glEnable( GL_LINE_SMOOTH );
		glLineWidth( 0.25f );
		glPolygonMode( GL_FRONT_AND_BACK , GL_LINE );
		glBindBuffer( GL_ARRAY_BUFFER , vbo );
		glBindBuffer( GL_ELEMENT_ARRAY_BUFFER , ebo );
		glEnableClientState( GL_VERTEX_ARRAY );
		glVertexPointer( 3 , GL_FLOAT , 0 , NULL );
		glDrawElements( GL_TRIANGLES , (GLsizei)(triangles.size()*3) , GL_UNSIGNED_INT , NULL );
		glBindBuffer( GL_ARRAY_BUFFER , 0 );
		glBindBuffer( GL_ELEMENT_ARRAY_BUFFER , 0 );
		glPolygonMode( GL_FRONT_AND_BACK , GL_FILL );
		glDisable( GL_LINE_SMOOTH );
		glPopMatrix();
		glDisable( GL_BLEND );
		glBlendFunc( src , dst );
	}
	if( showVectors )
	{
		static std::vector< float > random;
		static bool firstTime = true;
		static double area = 0;
		if( firstTime )
		{
			srand( 0 );
			random.resize( triangles.size() );
			for( int i=0 ; i<triangles.size() ; i++ )
			{
				random[i] = (float)Random< double >();
				Point3D< float > n = Point3D< float >::CrossProduct( vertices[ triangles[i][1] ] - vertices[ triangles[i][0] ] , vertices[ triangles[i][2] ] - vertices[ triangles[i][0] ] );
				area += Length(n);
			}
			firstTime = false;
		}

		Point3D< float > f = camera.forward / 256;
		glDisable( GL_LIGHTING );
		glPushMatrix();
		glTranslatef( -f[0] , -f[1] , -f[2] );
		glBegin( GL_TRIANGLES );
		double delta = area / (int)( vectorCount + 0.5f );
		for( int i=0 ; i<triangles.size() ; i++ )
		{
			Point3D< float > n = Point3D< float >::CrossProduct( vertices[ triangles[i][1] ] - vertices[ triangles[i][0] ] , vertices[ triangles[i][2] ] - vertices[ triangles[i][0] ] );
			double a = Length(n) , r = random[i];
			n /= a;
			//if( a>r*delta )
			//{
				Point3D< float > p = ( vertices[ triangles[i][0] ] + vertices[ triangles[i][1] ] + vertices[ triangles[i][2] ] ) / 3.f;
				Point3D< float > d = vectorField[i] * vectorScale;
				Point3D< float > _n = Point3D< float >::CrossProduct( d , n );
				_n /= 20;
				glColor3f( 0.0f , 0.0f , 0.0f ) , glVertex3f( p[0]+d[0] , p[1]+d[1] , p[2]+d[2] );
				glColor3f( 0.0f, 0.0f, 0.0f), glVertex3f(p[0] - _n[0], p[1] - _n[1], p[2] - _n[2]), glVertex3f(p[0] + _n[0], p[1] + _n[1], p[2] + _n[2]);
			//}
		}
		glEnd();
		glPopMatrix();
	}
}
void SurfaceVisualization::mouseFunc( int button , int state , int x , int y )
{
	newX = x ; newY = y;

	rotating = scaling = panning = false;
	if( button==GLUT_LEFT_BUTTON  )
		if( glutGetModifiers() & GLUT_ACTIVE_CTRL ) panning = true;
		else                                        rotating = true;
	else if( button==GLUT_RIGHT_BUTTON ) scaling = true;
}
void SurfaceVisualization::motionFunc( int x , int y )
{
	oldX = newX , oldY = newY , newX = x , newY = y;

	int imageSize = std::min< int >( screenWidth , screenHeight );
	float rel_x = (newX - oldX) / (float)imageSize * 2;
	float rel_y = (newY - oldY) / (float)imageSize * 2;
	float pRight = -rel_x * zoom , pUp = rel_y * zoom;
	float sForward = rel_y*4;
	float rRight = rel_y , rUp = rel_x;

	if     ( rotating ) camera.rotateUp( rUp ) , camera.rotateRight( rRight );
	else if( scaling  ) zoom *= (float)pow( 0.9 , (double)sForward );
	else if( panning  ) camera.translate( camera.right * pRight + camera.up * pUp );

	glutPostRedisplay();
}

void SurfaceVisualization::idle( void ){ if( !promptCallBack ){ ; } }

void SurfaceVisualization::keyboardFunc( unsigned char key , int x , int y )
{
	switch( key )
	{
		case '-': vectorScale /= 1.1f ; break;
		case '+': vectorScale *= 1.1f ; break;
	}
}
 
void SurfaceVisualization::specialFunc( int key, int x, int y )
{
	float stepSize = 10.f / ( screenWidth + screenHeight );
	if( glutGetModifiers()&GLUT_ACTIVE_CTRL ) stepSize /= 16;
	float panSize = stepSize*2 , scaleSize = stepSize*2;

	switch( key )
	{
#if 1
	case KEY_UPARROW:    zoom *= 0.98f ; break;
	case KEY_DOWNARROW:  zoom /= 0.98f ; break;
#else
	case KEY_UPARROW:    camera.translate(  camera.forward*scaleSize ) ; break;
	case KEY_DOWNARROW:  camera.translate( -camera.forward*scaleSize ) ; break;
#endif
	case KEY_LEFTARROW:  camera.translate(  camera.right * panSize ) ; break;
	case KEY_RIGHTARROW: camera.translate( -camera.right * panSize ) ; break;
	case KEY_PGUP:       camera.translate( -camera.up    * panSize ) ; break;
	case KEY_PGDN:       camera.translate(  camera.up    * panSize ) ; break;
	}
	glutPostRedisplay();
}
