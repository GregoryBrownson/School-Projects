#include "Matrix.cuh"
extern "C"
{
#include "Globals.h"
}

#include <cuda.h>
#include <cassert>
#include <iomanip>
#include <algorithm>
#include <omp.h>

inline void HandleError(cudaError_t cudaStatus, const char *file, int line)
{
	if (cudaStatus != cudaSuccess)
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(cudaStatus), file, line);
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

/************************************************************************
*																		*
*	Function:	Matrix()												*
*																		*
*	Description: Default constructor for Matrix class					*
*																		*
************************************************************************/

Matrix::Matrix()
{
	Rows = 0;
	Columns = 0;
	num_elements = 0;
	mat = nullptr;
}

/************************************************************************
*																		*
*	Function:	Matrix(int, int)										*
*																		*
*	Description: Basic Constructor for Matrix class that allocates		*
*				memory for an m x n matrix								*
*																		*
************************************************************************/

Matrix::Matrix(int r, int c)
{
	assert(r > -1 && c > -1);
	Rows = r;
	Columns = c;
	num_elements = Rows * Columns;
	init();
}

/************************************************************************
*																		*
*	Function:	Matrix(const Matrix&)									*
*																		*
*	Description:  Copy constructor for Matrix class						*
*																		*
************************************************************************/

Matrix::Matrix(const Matrix& m)
{
	Rows = m.getRows();
	Columns = m.getColumns();
	num_elements = Rows * Columns;
	init(m);
}

/************************************************************************
*																		*
*	Function:	~Matrix()
*																		*
*	Description: Destructor for Matrix class
*
************************************************************************/

Matrix::~Matrix()
{
	empty();
}

/************************************************************************
*																		*
*	Function:	init()													*
*																		*
*	Description: Allocates memory for Matrix and sets values to 0		*
*																		*
************************************************************************/

void Matrix::init()
{
	try
	{
		mat = new float*[Rows];
	}
	catch (bad_alloc& ba)
	{
		cerr << "exception caught:  " << ba.what() << '\n';
	}

	try
	{
		mat[0] = new float[Rows * Columns];
		
	}
	catch (bad_alloc& ba)
	{
		cerr << "exception caught:  " << ba.what() << '\n';
	}

	int j = 0;
	while (j < Columns)
	{
		mat[0][j] = 0.0;
		++j;
	}

	int i = 1;
	while (i < Rows)
	{
		mat[i] = mat[0] + i * Columns;
		j = 0;
		while (j < Columns)
		{
			mat[i][j] = 0.0;
			++j;
		}
		++i;
	}
}

/************************************************************************
*																		*
*	Function:	init(const Matrix&)										*
*																		*
*	Description: Allocates memory for Matrix and copies values from		*
*				another Matrix											*
*																		*
************************************************************************/

void Matrix::init(const Matrix & m)
{
	try
	{
		mat = new float*[Rows];
	}
	catch (bad_alloc& ba)
	{
		cerr << "exception caught:  " << ba.what() << '\n';
	}

	try
	{
		mat[0] = new float[Rows * Columns];
	}
	catch (bad_alloc& ba)
	{
		cerr << "exception caught:  " << ba.what() << '\n';
	}

	int j = 0;
	while (j < Columns)
	{
		mat[0][j] = m[0][j];
		++j;
	}

	int i = 1;
	while (i < Rows)
	{
		mat[i] = mat[0] + i * Columns;

		int j = 0;
		while (j < Columns)
		{
			mat[i][j] = m[i][j];
			++j;
		}
		++i;
	}
}

/************************************************************************
*																		*
*	Function:	Empty()													*
*																		*
*	Description: Deallocates memory when Matrix is destroyed			*
*																		*
************************************************************************/

void Matrix::empty()
{

	int i = 0;

	delete mat[0];
	
	delete mat;
	Rows = 0;
	Columns = 0;
	num_elements = 0;
}

/************************************************************************
*																		*
*	Function:	getRows()												*
*																		*
*	Description: Returns number of Rows									*
*																		*
************************************************************************/

int Matrix::getRows() const
{
	return Rows;
}

/************************************************************************
*																		*
*	Function:	getColumns()											*
*																		*
*	Description: Returns number of Columns								*
*																		*
************************************************************************/

int Matrix::getColumns() const
{
	return Columns;
}

/************************************************************************
*																		*
*	Function:	get(int, int)											*
*																		*
*	Description: Returns value at specified position					*
*																		*
************************************************************************/

float Matrix::get(int r, int c) const
{
	assert(r > -1 && c > -1);
	assert(r < Rows && c < Columns);
	return mat[r][c];
}

/************************************************************************
*																		*
*	Function:	set(int, int, value)									*
*																		*
*	Description: Sets the value of a certain position to the specified	*
*				value													*
*																		*
************************************************************************/

void Matrix::set(int r, int c, float val)
{
	assert(r > -1 && c > -1);
	assert(r < Rows && c < Columns);
	mat[r][c] = val;
}

/************************************************************************
*																		*
*	Function:	Transpose()												*
*																		*
*	Description: Computes and returns transpose of a Matrix				*
*																		*
************************************************************************/

__global__ void MTranspose(float *d_mat, float *a_mat, int rows, int columns)
{
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	int x = index % columns;
	int y = index / columns;

	int aindex = x * rows + y;

	a_mat[aindex] = d_mat[index];
}

Matrix Matrix::Transpose() const
{
	float *d_mat, *a_mat;
	float *h_mat = *mat;
	int gX, gY;
	int bX = BLOCK_SIZE, bY = BLOCK_SIZE;
	size_t i_pitch, o_pitch;
	Matrix temp(Rows, Columns);

	HANDLE_ERROR(cudaMallocPitch(&d_mat, &i_pitch, Columns * sizeof(float), Rows));
	HANDLE_ERROR(cudaMallocPitch(&a_mat, &o_pitch, Rows * sizeof(float), Columns));

	HANDLE_ERROR(cudaMemcpy2D(d_mat, i_pitch, h_mat, Columns * sizeof(float), Columns * sizeof(float), Rows, cudaMemcpyHostToDevice));

	gX = ceil(sqrt(((Rows * Columns + BLOCK_SIZE - 1) / BLOCK_SIZE)));

	gY = gX;

	dim3 gridSize(gX, gY);

	dim3 blockSize(bX, bY);

	MTranspose<<<gridSize,blockSize>>>(d_mat, a_mat, Rows, Columns);

	h_mat = NULL;

	h_mat = temp[0];

	HANDLE_ERROR(cudaMemcpy2D(h_mat, Rows * sizeof(float), a_mat, o_pitch, Rows * sizeof(float), Columns, cudaMemcpyDeviceToHost));

	return temp;
}

/************************************************************************
*																		*
*	Function:	Inverse()												*
*																		*
*	Description: Computes and returns inverse of Matrix					*
*																		*
************************************************************************/

__global__ void MRowReduce(float *d_mat, float *i_mat, int rows, int columns, int i)
{
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	int x = index % columns;
	int y = index / columns;

	int pindex = i * Columns + i;
	int cindex = y * Columns + i;
	int prindex = i * columns + x;

	scalar = d_mat[cindex] / d_mat[pindex];

	if (y > i)
	{
		d_mat[index] += scalar * d_mat[prindex];
		i_mat[index] += scalar * i_mat[prindex];
	}
}

__global__ void MUnitReduce(float *d_mat, float *i_mat, int rows, int columns, int i)
{
	int cindex = blockIdx.x * blockDim.x + threadIdx.x
	int pindex = i * Columns + i;

	scalar = 1.0f / d_mat[pindex];

	if (y > i)
	{
		d_mat[cindex] *= scalar
		i_mat[cindex] *= scalar
	}
}


Matrix Matrix::Inverse() const
{
	assert(Rows == Columns);

	Matrix I = this->Identity();

	float *d_mat, *di_mat, *a_mat;
	float *h_mat = *mat;
	float *hi_mat = I[0];
	int gX, gY;
	int bX = BLOCK_SIZE, bY = BLOCK_SIZE;
	size_t i_pitch, o_pitch, di_pitch;	

	int num_threads = MAX_THREADS;

	int i = 0;
	while (i < Rows)
	{
		float maxEl = abs(G[i][i]);
		int maxRow = i;
		int j = i + 1;
		// Search for maximum column
		while (j < Rows)
		{
			if (abs(G[j][i]) > maxEl)
			{
				maxEl = abs(G[j][i]);
				maxRow = i;
			}
			++j;
		}

		if (maxEl == 0.0)
		{
			cout << "Error, Matrix is singular!\n";
			break;
		}

		// Swap maximum row with current row

#pragma omp parallel firstprivate(j)
		{
			j = omp_get_thread_num();
			while (j < Columns)
			{
				float tmp = G[i][j];
				float tmp2 = I[i][j];
				G[i][j] = G[maxRow][j];
				I[i][j] = I[maxRow][j];
				G[maxRow][j] = tmp;
				I[maxRow][j] = tmp2;
				j += num_threads;
			}
		}

		HANDLE_ERROR(cudaMallocPitch(&d_mat, &i_pitch, Columns * sizeof(float), Rows));
		HANDLE_ERROR(cudaMallocPitch(&di_mat, &di_pitch, Columns * sizeof(float), Rows));
		HANDLE_ERROR(cudaMallocPitch(&a_mat, &o_pitch, Columns * sizeof(float), Rows));

		HANDLE_ERROR(cudaMemcpy2D(d_mat, i_pitch, h_mat, Columns * sizeof(float), Columns * sizeof(float), Rows, cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy2D(di_mat, i_pitch, hi_mat, Columns * sizeof(float), Columns * sizeof(float), Rows, cudaMemcpyHostToDevice));

		gX = ceil(sqrt(((Rows * Columns + BLOCK_SIZE - 1) / BLOCK_SIZE)));

		gY = gX;

		dim3 gridSize(gX, gY);

		dim3 blockSize(bX, bY);

		MRowReduce<<<gridSize, blockSize>>>(d_mat, i_mat, Rows, Columns, i);


		MUnitReduce<<<((Columns + 256 - 1) / 256), 256>>>(d_mat, i_mat, Rows, Columns, i);

		h_mat = NULL;

		h_mat = I[0];

		HANDLE_ERROR(cudaMemcpy2D(h_mat, Rows * sizeof(float), a_mat, o_pitch, Rows * sizeof(float), Columns, cudaMemcpyDeviceToHost));

		

//#pragma omp parallel firstprivate(j)
//		{
//			j = i + omp_get_thread_num() + 1;
//			while (j < Rows)
//			{
//				float scalar = -(G[j][i] / G[i][i]);
//				int k = 0;
//				while (k < Columns)
//				{
//					G[j][k] += scalar * G[i][k];
//					I[j][k] += scalar * I[i][k];
//					++k;
//				}
//				j += num_threads;
//			}
//		}

		//float scalar = 1.0 / G[i][i];

		// Reduce to unit matrix


//#pragma omp parallel firstprivate(j)
//		{
//			j = omp_get_thread_num();
//			while (j < Columns)
//			{
//				G[i][j] *= scalar;
//				I[i][j] *= scalar;
//				j += num_threads;
//			}
//		}

//#pragma omp parallel firstprivate(j, scalar)
//		{
//			j = i - 1 - omp_get_thread_num();
//			while (j >= 0)
//			{
//				scalar = -G[j][i];
//				int k = 0;
//				while (k < Columns)
//				{
//					G[j][k] += scalar * G[i][k];
//					I[j][k] += scalar * I[i][k];
//					++k;
//				}
//				j -= num_threads;
//			}
//		}

		++i;
	}

	// This section is part of LU decomposition method
	// To be completed

	/*
	Matrix L, R, I;
	L = this->Lower();
	L.display(cout, " ");
	R = this->Transpose();
	R.display(cout, " ");
	I = this->Identity();
	I.display(cout, " ");
	*/

	return I;
}

/************************************************************************
*																		*
*	Function:	Identity()												*
*																		*
*	Description: Returns the identity Matrix							*
*																		*
************************************************************************/

Matrix Matrix::Identity() const
{
	assert(Rows == Columns);
	Matrix I(Rows, Columns);

	int num_threads = MAX_THREADS;
#pragma omp parallel
	{	
		int i = omp_get_thread_num();
		while (i < Rows)
		{
			I[i][i] = 1.0;
			i += num_threads;
		}
	}

	return I;
}

/************************************************************************
*																		*
*	Function:	Determinant												*
*																		*
*	Description: Computes and returns the determinant of a Matrix		*
*																		*
************************************************************************/

float Matrix::Determinant() const
{
	float Det = 0;
	//	Calculate via first row expansion
	if (Rows == 1)
		return mat[0][0];
	else if (Rows == 2)
		return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
	else
	{
#pragma omp parallel
		{
			int num_threads = omp_get_num_threads();
			int i = omp_get_thread_num();
			while (i < Columns)
			{
				Det += pow(-1, i) * mat[0][i] * Minor(0, i).Determinant();
				i += num_threads;
			}
		}
	}
	return Det;
}

/************************************************************************
*																		*
*	Function:	Gauss()													*
*																		*
*	Description: Solves system of equations with Gaussian Elimination	*
*																		*
************************************************************************/

void Matrix::Gauss() const
{
	int num_threads = omp_get_max_threads();

	int minimum = min(Rows, Columns);
	int i = 0;
	while (i < minimum)
	{
		float maxEl = abs(mat[i][i]);
		int maxRow = i;
		int j = i + 1;

		// Search for maximum column
		while (j < Rows)
		{
			if (abs(mat[j][i]) > maxEl)
			{
				maxEl = abs(mat[j][i]);
				maxRow = i;
			}
			++j;
		}

		if (maxEl == 0.0)
		{
			cout << "Error, Matrix is singular!\n";
			break;
		}

		// Swap maximum row with current row
#pragma omp parallel firstprivate(j)
		{
			int c = Columns;
			j = i + omp_get_thread_num();
			while (j < c)
			{
				float tmp = mat[i][j];
				mat[i][j] = mat[maxRow][j];
				mat[maxRow][j] = tmp;
				j += num_threads;
			}
		}

		// Make all rows below this one 0 in current column
#pragma omp parallel firstprivate(j)
		{
			j = i + omp_get_thread_num() + 1;
			while (j < Rows)
			{
				float scalar = -(mat[j][i] / mat[i][i]);
				int k = i;
				while (k < Columns)
				{
					if (i == k)
						mat[j][k] = 0.0;
					else
						mat[j][k] += scalar * mat[i][k];
					++k;
				}
				j += num_threads;
			}
		}


		// Reduce to unit matrix
#pragma omp parallel firstprivate(j)
		{
			float scalar = 1.0 / mat[i][i];
			int c = Columns;
			j = i + omp_get_thread_num();
			while (j < c)
			{
				mat[i][j] *= scalar;
				j += num_threads;
			}
		}
		++i;
	}


	// RRE Form
	i = minimum - 1;
	while (i >= 0)
	{
#pragma omp parallel 
		{
			int j = i - 1 - omp_get_thread_num();
			while (j >= 0)
			{
				float scalar = -mat[j][i];
				int k = i;
				while (k < Columns)
				{
					mat[j][k] += scalar * mat[i][k];
					++k;
				}
				j -= num_threads;
			}
		}
		--i;
	}
}

/************************************************************************
*																		*
*	Function:	display(ostream&, string)								*
*																		*
*	Description: Outputs Matrix to a specified output with delimited	*
*				values													*
*																		*
************************************************************************/

void Matrix::display(ostream& out, string delim) const
{
	int i = 0;
	while (i < Rows)
	{
		int j = 0;
		while (j < Columns)
		{
			out << fixed << setprecision(4) << mat[i][j] << delim;
			++j;
		}
		out << '\n';
		++i;
	}
	out << '\n';
}

/************************************************************************
*																		*
*	Function:	operator+												*
*																		*
*	Description: Addition operator for Matrix							*
*																		*
************************************************************************/

Matrix Matrix::operator+(const Matrix& m)
{
	assert(Rows == m.getRows() && Columns == m.getColumns());
	Matrix temp(Rows, Columns);
#pragma omp parallel
	{
		int ID = omp_get_thread_num();
		int num_threads = omp_get_num_threads();
		int i = ID;
		while (i < Rows)
		{
			int j = 0;
			while (j < Columns)
			{
				temp[i][j] = mat[i][j] + m[i][j];
				++j;
			}
			i += num_threads;
		}
	}

	return temp;
}

/************************************************************************
*																		*
*	Function: operator-													*
*																		*
*	Description: Subtraction operator for Matrix						*
*																		*
************************************************************************/

Matrix Matrix::operator-(const Matrix& m)
{
	assert(Rows == m.getRows() && Columns == m.getColumns());
	Matrix temp(Rows, Columns);
#pragma omp parallel
	{
		int ID = omp_get_thread_num();
		int num_threads = omp_get_num_threads();
		int i = ID;
		while (i < Rows)
		{
			int j = 0;
			while (j < Columns)
			{
				temp[i][j] = mat[i][j] - m[i][j];
				++j;
			}
			i += num_threads;
		}
	}

	return temp;
}

/************************************************************************
*																		*
*	Function:	operator*(const Matrix&)								*
*																		*
*	Description: Matrix multiplication operator							*
*																		*
************************************************************************/

Matrix Matrix::operator*(const Matrix& m)
{
	assert(Columns == m.getRows());
	Matrix temp(Rows, m.getColumns());
#pragma omp parallel
	{
		int ID = omp_get_thread_num();
		int num_threads = omp_get_num_threads();
		int i = ID;
		while (i < Rows)
		{
			int j = 0;
			while (j < Columns)
			{
				int c = 0;
				int r = 0;
				while (c < Columns)
				{
					temp[i][j] += mat[i][c] * m[r][j];
					++r;
					++c;
				}
				++j;
			}

			i += num_threads;
		}
	}

	return temp;
}

/************************************************************************
*																		*
*	Function:	operator*(const float&)								*
*																		*
*	Description: Scalar multiplication operator							*
*																		*
************************************************************************/

Matrix Matrix::operator*(const float & s)
{
	Matrix temp(Rows, Columns);
#pragma omp parallel
	{
		int ID = omp_get_thread_num();
		int num_threads = omp_get_num_threads();
		int i = ID;
		while (i < Rows)
		{
			int j = 0;
			while (j < Columns)
			{
				temp[i][j] += mat[i][j] * s;
				++j;
			}
			i += num_threads;
		}
	}

	return temp;
}

/************************************************************************
*																		*
*	Function:	operator=												*
*																		*
*	Description: Assignment operator for Matrix							*
*																		*
************************************************************************/

void Matrix::operator=(const Matrix& m)
{
#if !DEBUG
	empty();
#endif
	Rows = m.getRows();
	Columns = m.getColumns();
	num_elements = Rows * Columns;
	init();
	assert(Rows == m.getRows() && Columns == m.getColumns());
#pragma omp parallel
	{
		int ID = omp_get_thread_num();
		int num_threads = omp_get_num_threads();
		int i = ID;
		while (i < Rows)
		{
			int j = 0;
			while (j < Columns)
			{
				mat[i][j] = m[i][j];
				++j;
			}
			i += num_threads;
		}
	}
}

/************************************************************************
*																		*
*	Function:	operator+=(const float&)								*
*																		*
*	Description: Add AND assignment operator							*
*																		*
************************************************************************/

void Matrix::operator+=(const Matrix & m)
{
	assert(Rows == m.getRows() && Columns == m.getColumns());

#pragma omp parallel
	{
		int ID = omp_get_thread_num();
		int num_threads = omp_get_num_threads();
		int i = ID;
		while (i < Rows)
		{
			int j = 0;
			while (j < Columns)
			{
				mat[i][j] += m[i][j];
				++j;
			}
			i += num_threads;
		}
	}
}

/************************************************************************
*																		*
*	Function:	operator-=(const Matrux&)								*
*																		*
*	Description: Subtract AND assignment operator						*
*																		*
************************************************************************/

void Matrix::operator-=(const Matrix & m)
{
	assert(Rows == m.getRows() && Columns == m.getColumns());

#pragma omp parallel
	{
		int ID = omp_get_thread_num();
		int num_threads = omp_get_num_threads();
		int i = ID;
		while (i < Rows)
		{
			int j = 0;
			while (j < Columns)
			{
				mat[i][j] -= m[i][j];
				++j;
			}
			i += num_threads;
		}
	}

}

/************************************************************************
*																		*
*	Function:	operator*=(const Matrix&)								*
*																		*
*	Description: Matrix multiply AND assignment operator				*
*																		*
************************************************************************/

void Matrix::operator*=(const Matrix & m)
{
	assert(Columns == m.getRows());

	if (Rows == Columns)
	{
#pragma omp parallel
		{
			int num_threads = omp_get_num_threads();
			int i = omp_get_thread_num();
			while (i < Rows)
			{
				int j = 0;
				while (j < Columns)
				{
					int c = 0;
					int r = 0;
					float val = 0.0;
					while (c < Columns)
					{
						val += mat[i][c] * m[r][j];
						++r;
						++c;
					}
					mat[i][j] = val;
					++j;
				}

				i += num_threads;
			}
		}
	}
	else
	{
		float** temp = mat;

		Columns = m.getColumns();

		init();

#pragma omp parallel
		{
			int num_threads = omp_get_num_threads();
			int i = omp_get_thread_num();
			while (i < Rows)
			{
				int j = 0;
				while (j < Columns)
				{
					int c = 0;
					int r = 0;
					float val;
					while (c < Columns)
					{
						val += temp[i][c] * m[r][j];
						++r;
						++c;
					}
					mat[i][j] = val;
					++j;
				}

				i += num_threads;
			}
		}

		int i = 0;

		delete temp[0];

		delete temp;
	}
}

/************************************************************************
*																		*
*	Function:	operator*=(const float&)								*
*																		*
*	Description: Scalar multiply AND assignment operator				*
*																		*
************************************************************************/

void Matrix::operator*=(const float & s)
{
#pragma omp parallel
	{
		int num_threads = omp_get_num_threads();
		int i = omp_get_thread_num();
		while (i < Rows)
		{
			int j = 0;
			while (j < Columns)
			{
				mat[i][j] *= s;
				++j;
			}
			i += num_threads;
		}
	}
}

/************************************************************************
*																		*
*	Function:	operator[]												*
*																		*
*	Description: l-val subscript operator for Matrix					*
*																		*
************************************************************************/

float* Matrix::operator[](unsigned i)
{
	assert(i < Rows);
	return mat[i];
}

/************************************************************************
*																		*
*	Function:	operator[]												*
*																		*
*	Description: r-val subscript operator for Matrix					*
*																		*
************************************************************************/

float* Matrix::operator[](unsigned i) const
{
	assert(i < Rows);
	return mat[i];
}

/************************************************************************
*																		*
*	Function:	Minor(int, int)											*
*																		*
*	Description: Computes and returns minor Matrix for specified point	*
*																		*
************************************************************************/

Matrix Matrix::Minor(int r, int c) const
{
	Matrix m(Rows - 1, Columns - 1);
#pragma omp parallel
	{
		int num_threads = omp_get_num_threads();
		int i = omp_get_thread_num();
		int x = omp_get_thread_num();
		while (x < Rows - 1)
		{
			if (i == r)
				i += num_threads;
			int j = 0;
			int y = 0;
			while (y < Columns - 1)
			{
				if (j == c)
					++j;
				m[x][y] = mat[i][j];
				++j;
				++y;
			}
			i += num_threads;
			x += num_threads;
		}
	}
	return m;
}

/************************************************************************
*																		*
*	Function:	Lower()													*
*																		*
*	Description: Computes and returns lower triangle Matrix for LU		*
*				decomposition											*
*																		*
************************************************************************/

Matrix Matrix::Lower() const
{
	assert(Rows == Columns);
	Matrix Low(*this);

	int minimum = min(Rows, Columns);
	int i = Rows - 1;
	while (i < Rows)
	{
		// Make all rows below this one 0 in current column

#pragma omp parallel
		{
			int num_threads = omp_get_num_threads();
			int j = i - omp_get_thread_num() - 1;
			while (j >= 0)
			{
				float scalar = -(Low[j][i] / Low[i][i]);
				int k = i;
				while (k >= 0)
				{
					if (i == k)
						Low[j][k] = 0.0;
					else
						Low[j][k] += scalar * Low[i][k];
					--k;
				}
				j -= num_threads;
			}

		}

		// Reduce to unit matrix
#pragma omp parallel
		{
			int num_threads = omp_get_num_threads();
			float scalar = 1.0 / Low[i][i];
			int j = i - omp_get_thread_num();
			while (j >= 0)
			{
				Low[i][j] *= scalar;
				j -= num_threads;
			}
		}
		++i;
	}

	return Low;
}

/************************************************************************
*																		*
*	Function: Upper()													*
*																		*
*	Description: Computes and returns upper triangle Matrix for LU		*
*				decomposition											*
*																		*
************************************************************************/

Matrix Matrix::Upper() const
{
	assert(Rows == Columns);

	Matrix Up(*this);


	int minimum = min(Rows, Columns);
	int i = 0;
	while (i < Rows)
	{
		// Make all rows below this one 0 in current column
#pragma omp parallel
		{
			int num_threads = omp_get_num_threads();
			int j = i + omp_get_thread_num() + 1;
			while (j < Rows)
			{
				float scalar = -(Up[j][i] / Up[i][i]);
				int k = i;
				while (k < Columns)
				{
					if (i == k)
						Up[j][k] = 0.0;
					else
						Up[j][k] += scalar * Up[i][k];
					++k;
				}
				j += num_threads;
			}
		}
	}
	return Up;
}

/************************************************************************
*																		*
*	Function:	Cofactor()												*
*																		*
*	Description: Computes and returns the cofactor Matrix				*
*																		*
************************************************************************/

Matrix Matrix::Cofactor() const
{
	Matrix temp(Rows, Columns);
#pragma omp parallel
	{
		int num_threads = omp_get_num_threads();
		int i = omp_get_thread_num();
		while (i < Rows)
		{
			int j = 0;
			while (j < Columns)
			{
				temp[i][j] = pow(-1.0, float(i + j)) * Minor(i, j).Determinant();
				++j;
			}
			i += num_threads;
		}
	}
	return temp;
}

/************************************************************************
*																		*
*	Function:	operator<<												*
*																		*
*	Description: ostream operator										*
*																		*
************************************************************************/

ostream& operator<<(ostream& out, const Matrix& m)
{
	m.display(out, "\t");
	return out;
}

/************************************************************************
*																		*
*	Function:	operator*(float, const Matrix&)						*
*																		*
*	Description: Scalar multiplication operator							*
*
************************************************************************/

Matrix operator*(float s, const Matrix & m)
{
	Matrix temp(m.Rows, m.Columns);

#pragma omp parallel
	{
		int ID = omp_get_thread_num();
		int num_threads = omp_get_num_threads();
		int i = ID;
		while (i < m.Rows)
		{
			int j = 0;
			while (j < m.Columns)
			{
				temp[i][j] += m.mat[i][j] * s;
				++j;
			}
			i += num_threads;
		}
	}

	return temp;
}