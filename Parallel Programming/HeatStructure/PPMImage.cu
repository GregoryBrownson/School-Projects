#include "PPMImage.cuh"

#include <omp.h>

PPMImage::PPMImage(void)
{

}

PPMImage::PPMImage(string fname, const Matrix& m) 
{
	filename = fname.append(".ppm");
	Rows = m.getRows();
	Columns = m.getColumns();
	setRange(m);

	int num_threads = omp_get_max_threads();

	RGB_data = new RGB[Columns*Rows];

	int x = 0;
	while (x < Rows)
	{	
		int y = 0;
		while (y < Columns)
		{
			int index = y + x * Columns;
			RGB_data[index] = getColors(m[x][y]);
			++y;
		}
		++x;
	}
}

PPMImage::PPMImage(string fname, int row, int col)
{
	filename = fname.append(".ppm");
	Rows = row;
	Columns = col;
}

PPMImage::~PPMImage()
{

}

void PPMImage::setColors(int r, int c, const Matrix& m)
{
	Rows = r;
	Columns = c;

	int x = 0;
	while (x < Rows)
	{		
		int y = 0;
		while (y < Columns)
		{
			int index = y + Columns * x;
			RGB_data[index] = getColors(m[x][y]);
			++y;
		}
		++x;
	}
}

void PPMImage::setRange(float r)
{
	Range = r;
}

void PPMImage::setMin(float m)
{
	Min = m;
}

void PPMImage::PrintImage()
{
	ofstream fout(filename.c_str());
	char w = ' ';	// Whitespace
	fout << "P3" << '\n';
	fout << "# " << filename << '\n';
	fout << Columns << w << Rows << '\n';
	fout << "255" << "\n";
	int i = 0;
	while (i < Rows)
	{
		int j = 0;
		int index = j + Columns * i;
		fout << RGB_data[index][0] << w << RGB_data[index][1] << w << RGB_data[index][2];
		++j;
		while (j < Columns)
		{
			int index = j + Columns * i;
			fout << RGB_data[index][0] << w << RGB_data[index][1] << w << RGB_data[index][2];
			++j;
		}
		fout << '\n';
		++i;
	}
	fout.close();
}

ostream& operator<<(ostream& out, const PPMImage& p)
{
	
	return out;
}

RGB PPMImage::getColors(float val)
{
	RGB x;
	float tran = (val - Min) / Range;
	if (tran < 0.25)
	{
		x[0] = 51;
		x[1] = 51 + (int)(204 * tran * 4.0);
		x[2] = 255;
	}
	else if (tran < 0.50)
	{
		x[0] = 51;
		x[1] = 255;
		x[2] = 255 - (int)(204 * 4 * (tran - 0.25)) ;
	}
	else if (tran < 0.75)
	{
		x[0] = 51 + (int)(204 * 4 * (tran - 0.5));
		x[1] = 255;
		x[2] = 51;
	}
	else
	{
		x[0] = 255;
		x[1] = 255 - (int)(204 * 4 * (tran - 0.75));
		x[2] = 51;
	}

	return x;
}

void PPMImage::setRange(const Matrix& m)
{
	int num_threads = omp_get_max_threads();

	float min = m[0][0];
	float max = m[0][0];

	float* _min = new float[num_threads];
	float* _max = new float[num_threads];
	
	int i = 0;
	while (i < num_threads)
	{
		_min[i] = min;
		_max[i] = max;
		++i;
	}

	#pragma omp parallel
	{
		int ID = omp_get_thread_num();
		int i = ID;
		while (i < Rows)
		{
			int j = 0;
			while (j < Columns)
			{
				if (m[i][j] < _min[ID])
				{
					_min[ID] = m[i][j];
				}
				else if (m[i][j] > _max[ID])
				{
					_max[ID] = m[i][j];
				}
				++j;
			}

			i += num_threads;
		}
	}

	i = 0;
	while (i < num_threads)
	{
		if (_min[i] < min)
		{
			min = _min[i];
		}
		if (_max[i] > max)
		{
			max = _max[i];
		}
		++i;
	}


	delete _min;
	delete _max;

	Min = min;

	Range = (max - min);
}