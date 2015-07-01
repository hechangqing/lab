#!/usr/bin/python
import sys

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: %s <text> <phone-map> <text-phone-remap-filename>" % sys.argv[0]
        exit(1)
    text = sys.argv[1]
    phn_map = sys.argv[2]
    text_phn_remap = sys.argv[3]

    phn_map_dict = {}
    with open(phn_map) as fin:
        lines = fin.readlines()
        for line in lines:
            splited_line = line.strip().split()
            if (len(splited_line) == 3):
                phn_map_dict[splited_line[1]] = splited_line[2]

    phn_set = set(phn_map_dict.iterkeys())
    print "phone set size: ", len(phn_set)
    phn_remap_set = set(phn_map_dict.itervalues())
    print "remaped phone set size: ", len(phn_remap_set)
    
    with open(text, "r") as fin:
        with open(text_phn_remap, "w") as fout:
            lines = fin.readlines()
            for line in lines:
                splited_line = line.strip().split()
                if (len(splited_line) >= 2):
                    fout.write(splited_line[0])
                    for i in range(1, len(splited_line)):
                        fout.write(" " + phn_map_dict[splited_line[i]])
                    fout.write('\n')

