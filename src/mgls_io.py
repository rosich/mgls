"""input/output functions
"""
import numpy as np

def write_file_onecol(out_file, data, separator, header):
    line = ""
    with open(out_file, 'w') as outfile:
        #outfile.write(str('#') + str(header) + "\n")
        for row in range(len(data)):
            line = line + str(data[row]) + str(separator)
            outfile.write( line + "\n" )
            line = ""

def write_file(out_file, data, separator, header):
    line = ""
    with open(out_file, 'w') as outfile:
        #outfile.write(str('#') + str(header) + "\n")
        for row in range(len(data)):
            for col in range(len(data[row])):
                if col != len(data[row]) - 1:
                    line = line + str(data[row][col]) + str(separator)
                elif col == len(data[row]) - 1:
                    line = line + str(data[row][col])
            outfile.write( line + "\n" )
            line = ""

def read_file(in_file, separator):
    with open(in_file, 'r') as infile:
        if separator == 'single_line':
            g_list = []
            row = []
            while(True):
                line = infile.readline()
                if not line: break
                if line[0] != '#'and line[0] != '\\' and line[0] != '|':
                    g_list.append(float(line))
        else:
            g_list = []
            row = []
            while(True):
                line = infile.readline()
                if not line: break
                if line[0] != '#' and line[0] != '\\' and line[0] != '|':
                    if separator == ' ':
                        string = line.split()
                    else:
                        string = line.split(separator)
                    for col in range(0,len(string)):
                        row.append(string[col])
                    g_list.append(row)
                    row = []
    return g_list

def get_data_vectors(data_in, col):
    """returns three lists: time, rv, rv uncertainty
    """
    #data_in = read_file(in_file, ' ') #read input file
    time = list()
    rv = list()
    rv_err = list()
    for line in range(len(data_in)):
        time.append(float(data_in[line][0]))
        rv.append(float(data_in[line][col]))
        rv_err.append(float(data_in[line][col+1]))
                
    return np.array(time), np.array(rv), np.array(rv_err)
    

    