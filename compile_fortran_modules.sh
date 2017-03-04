#!/bin/bash
cd ./src &&

f2py -m -c mGLS mGLS.f95  -llapack  &&
mv mGLS.so ../bin

f2py -m -c mMGLS mMGLS.f95 --opt=-O3 -llapack  &&
f2py -m -c mMGLS_AR mMGLS_AR.f95 --opt=-O3 -llapack &&
mv mMGLS.so ../bin &&
mv mMGLS_AR.so ../bin
