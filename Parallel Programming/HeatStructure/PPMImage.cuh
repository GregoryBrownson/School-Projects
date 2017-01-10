#pragma once

#ifndef PPMIMAGE_H
#define PPMIMAGE_H

#include <iostream>
#include <fstream>
#include <string>
#include <list>
#include <map>
#include <cuda.h>

#include "Matrix.cuh"

using namespace std;


struct RGB
{
	RGB() {};
	RGB(int x, int y, int z)
	{
		rgb_vals[0] = x;
		rgb_vals[1] = y;
		rgb_vals[2] = z;
	};


	inline int& operator[](unsigned i)
	{
		return rgb_vals[i];
	};
	inline int operator[](unsigned i) const
	{
		return rgb_vals[i];
	};

private:
	int rgb_vals[3];
};

class PPMImage
{
public:
	PPMImage(void);
	PPMImage(string fname, const Matrix& m);
	PPMImage(string fname, int row, int col);

	~PPMImage();

	void setColors(int r, int c, const Matrix& m);
	void setRange(float r);
	void setMin(float m);

	void PrintImage();

	friend ostream& operator<<(ostream& out, const PPMImage&);

private:
	RGB getColors(float val);
	void setRange(const Matrix& m);

private:
	string filename;
	float Range;
	float Min;
	int Rows;
	int Columns;
	RGB *RGB_data;
};
#endif