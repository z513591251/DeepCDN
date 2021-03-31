"""
PDMNOnehot    : Position-Dependent MonoNucleotides Onehot (DNA or RNA)
PDMNDict      : Position-Dependent MonoNucleotides Dictionary (DNA or RNA)
PDDNOnehot    : Position-Dependent DiNucleotides Onehot (DNA or RNA)
PDDNDict      : Position-Dependent DiNucleotides Dictionary(DNA or RNA)
PDTNOnehot    : Position-Dependent TriNucleotides Onehot (DNA or RNA)
PDTNDict      : Position-Dependent TriNucleotides Dictionary (DNA or RNA)
DNAComposition: DNA Composition
GCcount       : The number of Gs and Cs
SpeKmercount  : The number of specific k-mer sub-sequence
DNPCP         : DiNucleotides PhysicoChemical Properties
TNPCP         : TriNucleotides PhysicoChemical Properties
DNAshape      : DNA structural features
RFHC          : Rings,Functional groups and Hydrogen bonds Composition
"""
import numpy as np
import sys
import itertools
import math
sys.path.append('.//indices')
from Wordvec import *
from Dinucleotide_indices import *
from Trinucleotide_indices import *
from Shape import *


def PDMNOnehot(inpSeq):
    _onehot = []
    for eachSeq in inpSeq:
        _onehot.append([OnehotMonoNuc.get(base) for base in eachSeq])  
    _onehot = np.array(_onehot).reshape(len(inpSeq),len(eachSeq)*4)
    return _onehot


def PDMNDict(inpSeq):
    _dict = []
    for eachSeq in inpSeq:
        _dict.append([DictMonoNuc.get(base) for base in eachSeq])
    _dict = np.array(_dict).reshape(len(inpSeq),len(eachSeq))
    return _dict


def PDDNOnehot(inpSeq):
    _onehot = []
    for eachSeq in inpSeq:
        _onehot.append([OnehotDiNuc.get(eachSeq[num:num+2]) for num in range(len(eachSeq)-1)]) 
    _onehot = np.array(_onehot).reshape(len(inpSeq),(len(eachSeq)-1)*16)
    return _onehot


def PDDNDict(inpSeq):
    _dict = []
    for eachSeq in inpSeq:
        _dict.append([DictDiNuc.get(eachSeq[num:num+2]) for num in range(len(eachSeq)-1)])
    _dict = np.array(_dict).reshape(len(inpSeq),(len(eachSeq)-1))
    return _dict

def PDTNOnehot(inpSeq):
    _onehot = []
    for eachSeq in inpSeq:
        _onehot.append([OnehotTriNuc.get(eachSeq[num:num+3]) for num in range(len(eachSeq)-2)])
    _onehot = np.array(_onehot).reshape(len(inpSeq),(len(eachSeq)-2)*64)
    return _onehot


def PDTNDict(inpSeq):
    _dict = []
    for eachSeq in inpSeq:
        _dict.append([DictTriNuc.get(eachSeq[num:num+3]) for num in range(len(eachSeq)-2)])
    _dict = np.array(_dict).reshape(len(inpSeq),(len(eachSeq)-2))
    return _dict

def DNAComposition(inpSeq,kmer):
# kmer = 1: Mononucleotide composition
# kmer = 2: DiMononucleotide composition
# kmer = 3: TriMononucleotide composition
# ...
# kmer = 5: PentaMononucleotide composition

    _composition = [] 
    _kmer = list(map(''.join,itertools.product('ATCG',repeat=kmer)))
    for eachSeq in inpSeq:
        _segment = [eachSeq[num:num+kmer] for num in range(len(eachSeq)-kmer+1)]
        _composition.append([(_segment.count(seg)/(len(eachSeq)-kmer+1)) for seg in _kmer])
    _composition = np.array(_composition).reshape(len(inpSeq),4**kmer)
    return _composition
 
def GCcount(inpSeq,start,end):
    _count = []
    for eachSeq in inpSeq:
        subSeq = eachSeq [start-1:end]
        _count.append(subSeq.count('G')+subSeq.count('C'))
    _count = np.array(_count).reshape(len(inpSeq),1)
    return _count


def SpeKmercount(inpSeq,Kmer,start,end):
    _count = [] 
    length = end - start
    if length < len(Kmer):
       print ('Wrong!!!, the length of sequence should be larger than that of Kmer subsequence')
    else:
       for eachSeq in inpSeq:
           subSeq = eachSeq [start-1:end]
           _segment = [subSeq[num:num+len(Kmer)] for num in range(len(subSeq)-len(Kmer)+1)]
           _count.append(_segment.count(Kmer))
       _count = np.array(_count).reshape(len(inpSeq),1)
       return _count    


def DNPCP(inpSeq,indice,windows):
    _properties = []
    _average = []
    if len(indice.keys()) == 16:
       for eachSeq in inpSeq:
           _properties.append([indice.get(eachSeq[num:num+2]) for num in range(len(eachSeq)-1)])
       _properties = np.array(_properties).reshape(len(inpSeq),(len(eachSeq)-1))
       for eachPro in _properties:
           _average.append([np.mean(eachPro[num:num+windows]) for num in range(len(eachPro)-windows+1)])
       _average = np.array(_average).reshape(len(_properties),len(eachPro)-windows+1)
       return _average
    else:
        print ('Wrong name of physicochemical indices!!! Please see the Dinucleotide_indices') 


def TNPCP(inpSeq,indice,windows): 
    _properties = []
    _average = []
    if len(indice.keys()) == 64:
       for eachSeq in inpSeq:
           _properties.append([indice.get(eachSeq[num:num+3]) for num in range(len(eachSeq)-2)])
       _properties = np.array(_properties).reshape(len(inpSeq),(len(eachSeq)-2))
       for eachPro in _properties:
           _average.append([np.mean(eachPro[num:num+windows]) for num in range(len(eachPro)-windows+1)])
       _average = np.array(_average).reshape(len(_properties),len(eachPro)-windows+1)
       return _average
    else:
       print ('Wrong name of physicochemical indices!!! Please see the Trinucleotide_indices') 


def DNAshape(inpSeq,indice): 
    _shape = []
    if len(indice.keys()) == 1024:
       for eachSeq in inpSeq:
           _shape.append([indice.get(eachSeq[num:num+5]) for num in range(len(eachSeq)-4)])
       _shape = np.array(_shape).reshape(len(inpSeq),(len(eachSeq)-4))
       return _shape
    else:
       print ('Wrong name of DNA shape!!! Please see the Shape') 


def RFHC(inpSeq):
    _composition = [] 
    RFC = {'A':[1,1,1],
           'C':[0,0,1],
           'G':[1,0,0],
           'T':[0,1,0]
           }
    for eachSeq in inpSeq:
        _composition.append([RFC.get(base) for base in eachSeq])  
    _composition = np.array(_composition).reshape(len(inpSeq),len(eachSeq)*3)
    return _composition


 
                
  
