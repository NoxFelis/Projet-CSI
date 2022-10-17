from gettext import npgettext
from openpyxl import NUMPY as np

from math import *
import numpy as np
import obja

#def construction maillage de base
#def identification des patch 
#def projection 
# Projection : 
# model_base :  model de base 
# model_origine : model d'origine
# patch : matrice contenant les vertices appartenant aux différents patch 
# retour : matrice retournant les coordonnées des vertices du mesh d'origine dans le mesh de base 
def projection(model_base,model_origine,patch):

    
    faces_base = model_base.face
    vertices_base = model_base.vertices
    faces_origine = model_origine.face
    vertices_origine = model_origine.vertices

    vertices_origine_base = np.ones((np.size(vertices_origine)))
    distance_origine_base = np.ones((len(vertices_origine)))

    # Parcours de face du maillage de base 
    for i in range(len(faces_base)) :

        f = faces_base[i]
        A = vertices_base[f.a]
        B = vertices_base[f.b]
        C = vertices_base[f.c]

        # Création de la "base" de la face sur laquelle on projette
        x1 = C - A
        x2 = C - B

        x1 = (1/np.linalg.norm(x1)) * x1
        x2 = (1/np.linalg.norm(x2)) * x2

        # si on veut une base orthonormée  
        # x1 = (1/np.linalg.norm(x1) * x1)
        # x2 = x2 - (1/np.linalg.norm(x1)) * np.dot(x2,x1) * x1
        # x3 = np.cross(x1,x2) #(theroriquement norm(x3) = 1)
        
        patch_courant = patch[i]

        # PArcous de sommets du modèle d'origine
        # appartenant au patch correspondant à une face du modèle de base
        for i in range(len(patch_courant)) : 
            P = vertices_origine[patch_courant[i]]

            # P_proj = (P.x1) * x1 + (P.x2) * x2 (".": produit scalaire et "*" : produit par un scalaire)
            u = np.dot(P,x1)
            v = np.dot(P,x2)

            P_proj = [u,v,1]

            coord_p_proj = u * x1 + v * x2

            dist_proj = np.linalg.norm(P - coord_p_proj,P - coord_p_proj)

            vertices_origine_base[patch_courant[i]] = P_proj
            distance_origine_base[patch_courant[i]] = dist_proj

    return vertices_origine_base, distance_origine_base
    
def aera_triangle(v1,v2,v3) : 
    return  v2[0] * v3[1] - v2[1] * v3[0] + v3[0] * v1[1] - v1[0] * v3[1] + v1[0] * v2[1] - v2[0] * v1[1]

def makeBarycentricCoordsMatrix (vertices_origine_base, f) :
        
        C = np.zeros((3,3))
        
        v1 = vertices_origine_base[f.a]
        v2 = vertices_origine_base[f.b]
        v3 = vertices_origine_base[f.c]
        
        aera = aera_triangle(v1,v2,v3)
        
        x1 = v1[0]
        y1 = v1[1]
        x2 = v2[0]
        y2 = v2[1]
        x3 = v3[0]
        y3 = v3[1]

        C[0,0] = (x2 * y3 - x3 * y2)
        C[0,1] = (y2 - y3) 
        C[0,2] = (x3 - x2) 
        C[1,0] = (x3 * y1 - x1 * y3) 
        C[1,1] = (y3 - y1) 
        C[1,2] = (x1 - x3) 
        C[2,0] = (x1 * y2 - x2 * y1) 
        C[2,1] = (y1 - y2) 
        C[2,2] = (x2 - x1) 


        return (1/aera) * C 

def determine_patch(bord,model_origine) : 
    faces_origine = model_origine.faces
    vertices_origine = model_origine.vertices
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
    return P

def partition(bords,model_origine) :
    patch = {}
    for i in range(len(bords)) :
        patch[i] = determine_patch(bord[i],model_origine)

    return patch 
        
        
