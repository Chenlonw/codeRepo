#########################################################################
# File Name: GNSYNC.sh
# Description: Geeknote automatically sync script
# Author: Chenlong Wang
# mail: clwang88@gmail.com
# Created Time: ti. 03. nov. 2015 kl. 16.32 +0100
#########################################################################
#!/bin/bash

jupyter-nbconvert --to slides $1 --reveal-prefix ../../build/reveal.js
