#!/bin/bash
# Wrapper for clusrec.py
# Receives:
#   $1 - number of human model
#   $2 - number of clothes model

python clusrec.py -i test_tie.txt -hm human_detection/multi_label/output/human/m$1_model.h5 -hl human_detection/multi_label/output/human/m$1_lbin.pickle -cm clothes_detection/output/clothes/d$2_model.h5 -cl clothes_detection/output/clothes/d$2_lbin.pickle
