import os
import numpy as np
import re
import random

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

inp_file = "list_F.txt"
out_name = "_F1"
tr_num = 48
ts_num = 75

f_inp = open(inp_file,'r')
f_tr = open("tr_list"+out_name+".txt",'w')
f_ts = open("ts_list"+out_name+".txt",'w')

list_ = []
while True:
    line = f_inp.readline().split()
    if not line: break
    list_.append(line[0])
assert len(list_) == tr_num+ts_num

random.shuffle(list_)

tr_list = []
for i in range(tr_num):
    tr_list.append(list_[i])
tr_list.sort(key=natural_keys)
ts_list = []
for i in range(ts_num):
    ts_list.append(list_[-i-1])
ts_list.sort(key=natural_keys)

for i in range(tr_num):
    print(tr_list[i]+'\t'+str(i),file=f_tr)
for i in range(ts_num):
    print(ts_list[i]+'\t'+str(i),file=f_ts)

f_inp.close()
f_tr.close()
f_ts.close()
