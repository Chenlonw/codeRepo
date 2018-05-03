#directively copy c program from c folder
for i in $(ls ../c/*.c)
do
	echo 'copying file' $i
	b=${i%%.c}
	c=${b##*/}
	cp $i ./$c".cu"
done
scons -c
scons -Q
python combine.py
mv libchlwcuda.a ../../lib
mv chlwang_cuda.h ../../include
