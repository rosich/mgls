#!/bin/bash
cd ./src &&

f2py2.7 -m -c mGLS mGLS.f95  -llapack  &&
mv mGLS*.so ../bin

f2py2.7 -m -c mMGLS mMGLS.f95 --opt=-O3 -llapack  &&
f2py2.7 -m -c mMGLS_AR mMGLS_AR.f95 --opt=-O3 -llapack &&
mv mMGLS*.so ../bin 
