#ifndef VECTOR_IO_INCLUDED
#define VECTOR_IO_INCLUDED
#include<cstdlib>
#include<cstdio>
#include<vector>
#include<Misha\Image.h>

template<typename T>
int ReadVector(std::vector<T> & vec, const char * fileName){

	FILE * file;
	file = fopen(fileName, "rb");
	if (!file) { printf("Unable to read %s \n", fileName); return 0; }
	int vecSize;
	fread(&vecSize, sizeof(int), 1, file);
	vec.resize(vecSize);
	fread(&vec[0], sizeof(T), vecSize, file);
	fclose(file);
	return 1;
}

template<typename T>
void WriteVector(const std::vector<T> & vec, const char * fileName){

	FILE * file;
	file = fopen(fileName, "wb");
	int vecSize = (int)vec.size();
	fwrite(&vecSize, sizeof(int), 1, file);
	fwrite(&vec[0], sizeof(T), vecSize, file);
	fclose(file);
}


template <class T>
void WriteBinaryImage(const Image<T> & image, const char * fileName) {
	FILE * file;
	file = fopen(fileName, "wb");
	int width = image.width();
	int height = image.height();
	fwrite(&width, sizeof(int), 1, file);
	fwrite(&height, sizeof(int), 1, file);
	fwrite(&image[0], sizeof(T), width*height, file);
	fclose(file);
}

template <class T>
int ReadBinaryImage(Image<T> & image, const char * fileName) {

	FILE * file;
	file = fopen(fileName, "rb");
	if (!file) { printf("Unable to read %s \n", fileName); return 0; }
	int width, height;
	fread(&width, sizeof(int), 1, file);
	fread(&height, sizeof(int), 1, file);
	image.resize(width, height);
	fread(&image[0], sizeof(T), width*height, file);
	fclose(file);
	return 1;
}


#endif //VECTOR_IO_INCLUDED