# -*- coding: utf-8 -*-
"""
Created on Mon May 21 23:36:08 2018

@author: ALVARO
"""
"""
af3=2 
af4=3
f7=4
f3=5
f4=6
f8=7
fc5=8
fc6=9
truth = 0
lie = 1
d16 -> text_value_emotiuon
"""
import re

truths = ['A001.csv','C006.csv','E002.csv','E003.csv','E004.csv','E006.csv','E007.csv','E011.csv','E013.csv','E014.csv','E016.csv','E018.csv','E021.csv','E027.csv','E039.csv','E041.csv','E052.csv','E056.csv','E060.csv','E061.csv']
lies = ['A001 mentira.csv','C006_mentira.csv','E002_mentira.csv','E004_mentira.csv','E006 mentira.csv','E007_mentira.csv','E011 mentira.csv','E013_mentira.csv','E014 mentira.csv','E016 mentira.csv','E018 mentira.csv','E027 mentira.csv','E030_mentira.csv','E039_mentira.csv','E041 mentira.csv','E052 mentira.csv','E056 mentira.csv','E060 mentira.csv','E061_mentira.csv']
#'E021 mentira.csv'

def generateFiles(truths, lies):
    print("Processing truth files")
    for i in range(len(truths)):
        with open(r"Training_data_raw/{}".format(truths[i]), 'r') as infile, \
             open(r"Training_data/{}".format(truths[i]), 'w') as outfile:
            print("Attempting to create truth file {}".format(i))
            for line in infile:
                try:
                    #print("Trying to split data")
                    d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16 = line.split("\t")
                except:
                    pass
                else:
                    #print("Attempting regex substitution")
                    d2 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d2)
                    d3 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d3)
                    d4 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d4)
                    d5 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d5)
                    d6 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d6)
                    d7 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d7)
                    d8 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d8)
                    d9 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d9)
                    d10 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d10)
                    d11 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d11)
                    d12 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d12)
                    d13 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d13)
                    d14 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d14)
                    d15 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d15)
                    d16 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d16)
                    

                    #outfile.write(d1+"\t"+d2+"\t"+d3+"\t"+d4+"\t"+d5+"\t"+d6+"\t"+d7+"\t"+d8+"\t"+d9+"\t"+d10+"\t"+d11+"\t"+d12+"\t"+d13+"\t"+d14+"\t"+d15+"\t"+d16)
                    #print("Writing output file")
                    outfile.write(d2+"\t"+d3+"\t"+d4+"\t"+d5+"\t"+d6+"\t"+d7+"\t"+d8+"\t"+d9+"\t"+"0"+"\n")
                    #outfile.write(d2+"\t"+d3+"\t"+d5+"\t"+d6+"\t"+"0"+"\n")
            print("Truth file {} completed :]".format(i))

    print("Processing lie files")
    for j in range(len(lies)):
        with open(r'Training_data_raw/{}'.format(lies[j]), 'r') as infile, \
             open(r'Training_data/{}'.format(lies[j]), 'w') as outfile:
            print("Attempting to create lie file {}".format(j))
            for line in infile:
                try:
                    #print("Trying to split data")
                    d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16 = line.split("\t")
                except:
                    #print ("Something bad happened with the data :[")
                    pass
                else:
                    #print("Attempting regex substitution")
                    d2 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d2)
                    d3 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d3)
                    d4 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d4)
                    d5 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d5)
                    d6 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d6)
                    d7 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d7)
                    d8 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d8)
                    d9 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d9)
                    d10 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d10)
                    d11 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d11)
                    d12 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d12)
                    d13 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d13)
                    d14 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d14)
                    d15 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d15)
                    d16 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d16)
                    
                    #print("Writing output file")
                    #outfile.write(d1+"\t"+d2+"\t"+d3+"\t"+d4+"\t"+d5+"\t"+d6+"\t"+d7+"\t"+d8+"\t"+d9+"\t"+d10+"\t"+d11+"\t"+d12+"\t"+d13+"\t"+d14+"\t"+d15+"\t"+d16)
                    #outfile.write(d2+"\t"+d3+"\t"+d4+"\t"+d5+"\t"+d6+"\t"+d7+"\t"+d8+"\t"+d9+"\t"+d16.rstrip()+"\t"+"1"+"\n")
                    outfile.write(d2+"\t"+d3+"\t"+d4+"\t"+d5+"\t"+d6+"\t"+d7+"\t"+d8+"\t"+d9+"\t"+"1"+"\n")
                    #outfile.write(d2+"\t"+d3+"\t"+d5+"\t"+d6+"\t"+"1"+"\n")
            print("Lie file {} completed :]".format(j))


def generateCrossFiles(truths, lies):
    print("Processing truth files")
    for i in range(len(truths)):
        with open(r"Training_data_raw/{}".format(truths[i]), 'r') as infile, \
             open(r"Cross_validation/{}".format(truths[i]), 'w') as outfile:
            print("Attempting to create truth file {}".format(i))
            for line in infile:
                try:
                    #print("Trying to split data")
                    d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16 = line.split("\t")
                except:
                    pass
                else:
                    #print("Attempting regex substitution")
                    d2 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d2)
                    d3 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d3)
                    d4 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d4)
                    d5 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d5)
                    d6 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d6)
                    d7 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d7)
                    d8 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d8)
                    d9 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d9)
                    d10 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d10)
                    d11 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d11)
                    d12 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d12)
                    d13 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d13)
                    d14 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d14)
                    d15 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d15)
                    d16 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d16)
                    

                    #outfile.write(d1+"\t"+d2+"\t"+d3+"\t"+d4+"\t"+d5+"\t"+d6+"\t"+d7+"\t"+d8+"\t"+d9+"\t"+d10+"\t"+d11+"\t"+d12+"\t"+d13+"\t"+d14+"\t"+d15+"\t"+d16)
                    #print("Writing output file")
                    outfile.write(d2+"\t"+d3+"\t"+d4+"\t"+d5+"\t"+d6+"\t"+d7+"\t"+d8+"\t"+d9+"\n")
                    #outfile.write(d2+"\t"+d3+"\t"+d5+"\t"+d6+"\t"+"0"+"\n")
            print("Truth file {} completed :]".format(i))

    print("Processing lie files")
    for j in range(len(lies)):
        with open(r'Training_data_raw/{}'.format(lies[j]), 'r') as infile, \
             open(r'Cross_validation/{}'.format(lies[j]), 'w') as outfile:
            print("Attempting to create lie file {}".format(j))
            for line in infile:
                try:
                    #print("Trying to split data")
                    d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16 = line.split("\t")
                except:
                    #print ("Something bad happened with the data :[")
                    pass
                else:
                    #print("Attempting regex substitution")
                    d2 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d2)
                    d3 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d3)
                    d4 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d4)
                    d5 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d5)
                    d6 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d6)
                    d7 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d7)
                    d8 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d8)
                    d9 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d9)
                    d10 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d10)
                    d11 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d11)
                    d12 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d12)
                    d13 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d13)
                    d14 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d14)
                    d15 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d15)
                    d16 = re.sub(r'[A-Z]*[0-9]*:[0-9]*,', '', d16)
                    
                    #print("Writing output file")
                    #outfile.write(d1+"\t"+d2+"\t"+d3+"\t"+d4+"\t"+d5+"\t"+d6+"\t"+d7+"\t"+d8+"\t"+d9+"\t"+d10+"\t"+d11+"\t"+d12+"\t"+d13+"\t"+d14+"\t"+d15+"\t"+d16)
                    #outfile.write(d2+"\t"+d3+"\t"+d4+"\t"+d5+"\t"+d6+"\t"+d7+"\t"+d8+"\t"+d9+"\t"+d16.rstrip()+"\t"+"1"+"\n")
                    outfile.write(d2+"\t"+d3+"\t"+d4+"\t"+d5+"\t"+d6+"\t"+d7+"\t"+d8+"\t"+d9+"\n")
                    #outfile.write(d2+"\t"+d3+"\t"+d5+"\t"+d6+"\t"+"1"+"\n")
            print("Lie file {} completed :]".format(j))



def unifyFiles(truths, lies):
    all_files = truths + lies
    with open('Training_data/db.csv', 'w') as outfile:
        for fname in all_files:
            with open('Training_data/{}'.format(fname)) as infile:
                for line in infile:
                    outfile.write(line)




#generateFiles(truths, lies)
#unifyFiles(truths, lies)
generateCrossFiles(truths, lies)