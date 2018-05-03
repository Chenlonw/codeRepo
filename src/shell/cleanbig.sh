#!/bin/bash

#########################################################################
# File Name: clean.sh
# Description: Clean all the data files (*.dat/*.bin)
# Author: Chenlong Wang
# mail: clwang88@gmail.com
# Created Time: on. 23. mars 2016 kl. 15.50 +0800
#########################################################################

echo "remove file bigger than (M)?"
read size
touch list
echo "########################################" >list
echo "    files has been found, Check!!!" >>list
echo "########################################\n" >>list
find . -size +$size'M' >list1
cat list1 >>list
less list
echo -n 'confirm or not? [y/n] >'
read choose
case "$choose" in
  [yY])  echo -n 'arguments:' && 
	  read index &&
	  find . -size +$size'M' -exec rm -$index {} \; && 
	  rm list1 list &&
	  exit;;
  [nN]) rm list1 list && echo "Exit" && exit;;
  * ) rm list1  list && echo "wrong input" && exit;;
esac
echo job done!
