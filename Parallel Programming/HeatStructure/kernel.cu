#include <cuda.h>
#include <stdio.h>

#include "PPMImage.cuh"
#include "Matrix.cuh"

int main()
{
	//ofstream fout("out.tab");
	/*ofstream fout1("out1.tab");
	ofstream fout2("out2.tab");*/

	int length = 3;
	int height = 3;

	int rows = length * height;
	float top = 70.0;
	float bottom = 50.0;
	float left = 60.0, right = 60.0;

	Matrix m(rows, rows);
	Matrix D(rows, 1);
	Matrix PPmat(height, length);

	int i = 0;
	while (i < rows)
	{
		m[i][i] = -4.0;
		D[i][0] = 0.0;

		if (i / length == 0)
		{
			D[i][0] -= top;
			m[i][i + length] = 1.0;
		}
		else if (i / length == height - 1)
		{
			D[i][0] -= bottom;
			m[i][i - length] = 1.0;
		}
		else
		{
			m[i][i + length] = 1.0;
			m[i][i - length] = 1.0;
		}

		if (i % length == 0)
		{
			D[i][0] -= left;
			m[i][i + 1] = 1.0;
		}
		else if (i % length == length - 1)
		{
			D[i][0] -= right;
			m[i][i - 1] = 1.0;
		}
		else
		{
			m[i][i + 1] = 1.0;
			m[i][i - 1] = 1.0;
		}
		++i;
	}

	cout << m;

	m = m.Transpose();

	cout << m;


	/*m = m.Inverse();

	m *= D;

	cout << m;

	i = 0;
	while (i < height)
	{
		int y = 0;
		while (y < length)
		{
			int x = length * i;
			PPmat[i][y] = m[x + y][0];
			++y;
		}
		++i;
	}

	fout << PPmat;

	fout.close();

	PPMImage ppm("Exercise3", PPmat);

	ppm.PrintImage();*/


	return 0;
}