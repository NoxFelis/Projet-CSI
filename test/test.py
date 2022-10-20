import obja 
import numpy as np

model_origine = obja.parse_file("sphere_input.obj")
faces = model_origine.faces
vertices = model_origine.vertices

patch_sommets = [124,121,118]
barycentre_patch = (1/3) * (vertices[patch_sommets[0]] + vertices[patch_sommets[1]] + vertices[patch_sommets[2]])
bord = [0,140,15,139,5,113,21,112,4,137,19,138,0]

A = []
P = []
O = []

for i in range(len(bord)-1) :
    vi = bord[i]
    viplus1 = bord[i+1]
    for k in range(len(faces)) :
        if faces[k].isIn(vi) and faces[k].isIn(viplus1) :
            if faces[k].orientation(vi,viplus1,'h') :
                if faces[k] not in A : 
                    A.append(faces[k])
                    P.append(faces[k])
            else :
                O.append(faces[k])

while A != [] :
    f = A[0]
    A.remove(f)
    for k in range(len(faces)) :
        if (faces[k] not in P) and f.adjacent(faces[k]) and (faces[k] not in O):
                P.append(faces[k])
                A.append(faces[k])



    

