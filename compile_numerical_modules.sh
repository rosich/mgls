#!/bin/bash
cd ./src &&
f2py2.7  -m -c mMGLS mMGLS.f95 --opt=-O3 -llapack  &&
mv mMGLS*.so ../bin 
