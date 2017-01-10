#pragma once


#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>

#include <iostream>
#include <string>

using namespace std;

class Matrix
{
public:
	Matrix();
	Matrix(int r, int c);
	Matrix(const Matrix& m);
	~Matrix();

	void init();
	void init(const Matrix& m);

	void empty();

	int getRows() const;
	int getColumns() const;
	float get(int r, int c) const;
	bool isIdentity();

	void set(int r, int c, float val);
	void resetMat(float* m);

	Matrix Transpose() const;
	Matrix Inverse() const;
	Matrix Identity() const;
	float Determinant() const;
	void Gauss() const;

	void display(ostream& out, string delim) const;

	Matrix operator+(const Matrix& m);
	Matrix operator-(const Matrix& m);
	Matrix operator*(const Matrix& m);
	Matrix operator*(const float& s);

	void operator=(const Matrix& m);
	void operator+=(const Matrix& m);
	void operator-=(const Matrix& m);
	void operator*=(const Matrix& m);
	void operator*=(const float& s);

	float* operator[](unsigned i);
	float* operator[](unsigned i) const;

	float** Mat() const;

	friend ostream& operator<<(ostream& out, const Matrix&);
	friend Matrix operator*(float s, const Matrix&);

private:

	Matrix Minor(int r, int c) const;
	Matrix Lower() const;
	Matrix Upper() const;
	Matrix Cofactor() const;

private:
	unsigned int Rows;
	unsigned int Columns;
	unsigned int num_elements;
	float **mat;
};

#endif