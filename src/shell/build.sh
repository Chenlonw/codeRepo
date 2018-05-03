#!/bin/bash
DIR="$( cd "$( dirname "$0" )" && pwd )"

echo $DIR

touch tmp.file
echo " " >>tmp.file
echo "# The following alias are automatically appended by CHLWREPO" >>tmp.file

for i in $(find . -name "*.sh")
do
	b=${i#*/}
	echo "alias" ${b%%.sh}=\"$DIR"/"$b\" >>tmp.file 
done

cat tmp.file >> ~/.zshrc
rm tmp.file
