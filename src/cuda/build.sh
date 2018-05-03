scons -c
scons -Q
python combine.py
mv libchlwcuda.a ../../lib
mv chlwang_cuda.h ../../include
