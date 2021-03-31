import re
import numpy as np


def openFile(input_file): 
    with open(input_file) as f:
         contents = f.readlines()
    return contents


def readFile(input_file):
    contents = openFile(input_file)    
    _sequence = [re.split(r'\s+',line)[0] for line in contents]
    _value = [re.split(r'\s+',line)[1] for line in contents]
    _value = list(map(float, _value))
    _length = [len(eachseq) for eachseq in _sequence]   
    if len(set(_length)) == 1:
       return _sequence, _value 
    else:
       print ('Warning!!! Inconsistent sequence length')

def readFeature(input_file):
    contents = openFile(input_file)    
    _value = [re.split(r'\s+',line)[0] for line in contents]
    _feature = [re.split(r'\s+',line)[2:len(line)] for line in contents]
    _feature = [[i for i in line if(len(str(i))!=0)] for line in _feature]
    _value = list(map(float, _value))
    _feature = np.array(_feature)
    return _value, _feature

