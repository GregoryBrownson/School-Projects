#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <bitset>
#include <sstream>
#include <cuda.h>

using namespace std;

inline void HandleError(cudaError_t cudaStatus, const char *file, int line)
{
	if (cudaStatus != cudaSuccess)
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(cudaStatus), file, line);
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define PI acos(-1.0f)

#define BLOCK_SIZE 32

__global__ void ProcessImageKernel(unsigned char *data, unsigned char *out_img, int columns, int rows, int sigma)
{
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	int x = index % columns;
	int y = index / columns;

	int half_width = sigma * 3;
	
	float** weights = new float*[half_width];

	int i = 0;
	while (i < 2 * (half_width - 1))
	{
		weights[i] = new float[half_width];
		int j = 0;
		while (j < 2 * (half_width - 1))
		{
			int py = i - half_width + 1;
			int px = j - half_width + 1;
			weights[i][j] = (1.0f / (2.0f * PI * sigma * sigma)) * expf(-(px * px + py * py) / (2.0f * sigma * sigma));
		}
	}

	int temp = data[index] * weights[half_width / 2][half_width / 2];

	if (x == 0)
	{
		temp += data[index];
		if (y == 0)
		{

		}
		else if (y == rows - 1)
		{

		}
		else
		{

		}

	}
	else if (x == columns - 1)
	{
		if (y == 0)
		{

		}
		else if (y == rows - 1)
		{

		}
		else
		{

		}
	}
	else
	{
		if (y == 0)
		{

		}
		else if (y == rows - 1)
		{

		}
		else
		{

		}
	}


	__syncthreads();
}


__global__ void RGBtoGrayKernel(unsigned char *rgb, unsigned char *gray, int pixels)
{
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	int Gindex = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	if (Gindex >= pixels) return;

	int Cindex = 3 * Gindex;
	unsigned char temp = 0.2989 * rgb[Cindex] + 0.5870 * rgb[Cindex + 1] + 0.1140 * rgb[Cindex + 2];
	gray[Gindex] = temp;

	__syncthreads();
}

int main()
{
	int gX = 0, gY = 0;
	int bX = BLOCK_SIZE, bY = BLOCK_SIZE;
	unsigned char *rgb_img, *g_img, *f_img;
	unsigned char *d_imgin, *d_imgout;
	int x, y;

	string type;
	int rows, columns, rgb_val;
	int pixels;

	

	ifstream fin("ChicagoSunset.ppm", ios::in | ios::binary);

	if (fin.is_open())
	{
		int size = 0;
		char *buffer;
		bool error;
		string line;
		istringstream is;

		getline(fin, line, '\n');

		is.str(line);

		is >> type;

		is.clear();

		getline(fin, line, '\n');

		while (line[0] == '#')
		{
			getline(fin, line, '\n');
		}

		is.str(line);

		is >> columns >> rows;

		is.clear();

		getline(fin, line, '\n');

		is.str(line);

		is >> rgb_val;

		is.clear();

		streampos begin, end;

		pixels = rows * columns;

		size = 3 * pixels;

		rgb_img = (unsigned char*)malloc(size * sizeof(unsigned char));
		g_img = (unsigned char*)malloc(pixels * sizeof(unsigned char));

		HANDLE_ERROR(cudaMalloc((void **)&d_imgin, size * sizeof(unsigned char)));
		HANDLE_ERROR(cudaMalloc((void **)&d_imgout, pixels * sizeof(unsigned char)));

		buffer = new char[size];

		fin.read(buffer, size);

		error = fin.eof();

		if (error)
		{
			cout << "GOOD";
		}

		memcpy(rgb_img, buffer, size * sizeof(unsigned char));

		ofstream fout("gray.pgm");

		fout << "P2\n" << columns << " " << rows << '\n' << rgb_val << '\n';

		x = 0;
		y = 0;
		while (x < rows)
		{
			y = 0;
			int index = 0;
			int cindex = 0;
			while (y < columns - 1)
			{
				index = x * columns + y;
				cindex = 3 * index;
				g_img[index] = 0.2989 * rgb_img[cindex] + 0.5870 * rgb_img[cindex + 1] + 0.1140 * rgb_img[cindex + 2];
				fout << +g_img[index] << " ";
				++y;
			}
			index = x * columns + y;
			g_img[index] = 0.2989 * rgb_img[cindex] + 0.5870 * rgb_img[cindex + 1] + 0.1140 * rgb_img[cindex + 2];
			fout << +g_img[index] << '\n';
			++x;
		}

		fin.close();

		fout.close();

		HANDLE_ERROR(cudaMemcpy(d_imgin, rgb_img, size * sizeof(unsigned char), cudaMemcpyHostToDevice));

		free(buffer);
	}
	else
	{
		cout << "File not found!\n";
		exit(-1);
	}

	gX = ceil(sqrt((pixels + BLOCK_SIZE * BLOCK_SIZE - 1) / (BLOCK_SIZE * BLOCK_SIZE)));

	gY = gX;

	dim3 gridSize(gX, gY);

	dim3 blockSize(bX, bY);


	RGBtoGrayKernel<<<gridSize, blockSize>>>(d_imgin, d_imgout, pixels);

	cudaError_t err = cudaGetLastError();

	HANDLE_ERROR(err);

	HANDLE_ERROR(cudaMemcpy(g_img, d_imgout, pixels * sizeof(unsigned char), cudaMemcpyDeviceToHost));

	ofstream fout("gray_image.pgm", ios::out);

	fout << "P2\n";
	fout << columns << " " << rows << '\n';
	fout << "255\n";

	x = 0;
	y = 0;
	while (x < rows)
	{
		y = 0;
		int index = 0;
		while (y < columns - 1)
		{
			index = x * columns + y;
			fout << +g_img[index] << ' ';
			++y;
		}
		index = x * columns + y;
		fout << +g_img[index] << '\n';
		++x;
	}

	fout.close();

	/*if (cudaStatus != cudaSuccess)
	{
	fprintf(stderr, "addWithCuda failed!");
	return 1;
	}
	*/

	free(g_img); free(rgb_img);
	cudaFree(d_imgin); cudaFree(d_imgout);

	return 0;
}