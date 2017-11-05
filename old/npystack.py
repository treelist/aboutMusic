
import numpy as np

NPYPATH = "./result/npy/"

LISTNAME = "npylist.txt"

with open(LISTNAME, "r") as f:
	wavlist = f.readlines()

globalCounter = 0
npystack = np.array([])

f = open("Chords.txt", 'w')

def switch1(x):
    return {
        'A': 0,
        'B': 1,
		'C': 2,
		'D': 3,
		'E': 4,
		'F': 5,
		'G': 6,
    }.get(x, -1)
	
for i in wavlist:
	temp = np.load(NPYPATH + i.strip())
	if(globalCounter == 0):
		npystack = temp
	else:
		npystack = np.vstack((npystack, temp))
	
	if(str.isalpha(i.strip()[0])):
		chord = i.strip()[0]
	else:
		chord = i.strip()[4]
	print("%d %c" % (globalCounter, chord))
	num = switch1(chord)
	
	f.write("%d %d\n" % (globalCounter, num))
	globalCounter += 1

f.close()
np.save("npystack", npystack)