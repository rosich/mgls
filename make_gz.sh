#!/usr/bin/bash

tar -cf mgls.tar  ./bin/*  make_gz.sh compile_* mgls.py  ./src/* ./data/* gen* mgls_pre* 
gzip -9 mgls.tar
