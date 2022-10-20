from gettext import npgettext

from math import *
import random
import numpy as np
import obja
import patch_limit as pl

#def construction maillage de base
#def identification des patch 
#def projection 
# Projection : 
# model_base :  model de base 
# model_origine : model d'origine
# patch : matrice contenant les vertices appartenant aux différents patch 
# retour : matrice retournant les coordonnées des vertices du mesh d'origine dans le mesh de base 
def projection(model_origine,model_base,patch):

    
    faces_base = model_base.faces
    vertices_base = model_base.vertices
    faces_origine = model_origine.faces
    vertices_origine = model_origine.vertices

    #vertices_origine_base = np.ones((np.size(vertices_origine)))
    #distance_origine_base = np.ones((len(vertices_origine)))
    vertices_origine_base = []

    # Parcours de face du maillage de base 
    for k in range(len(faces_base)) :

        f = faces_base[k]
        A = vertices_base[f.a]
        B = vertices_base[f.b]
        C = vertices_base[f.c]

        # Création de la "base" de la face sur laquelle on projette
        x1 = B - A
        x2 = C - A

##        x1 = (1/np.linalg.norm(x1)) * x1
##        x2 = (1/np.linalg.norm(x2)) * x2

        # si on veut une base orthonormée  
        # x1 = (1/np.linalg.norm(x1) * x1)
        # x2 = x2 - (1/np.linalg.norm(x1)) * np.dot(x2,x1) * x1
        # x3 = np.cross(x1,x2) #(theroriquement norm(x3) = 1)        

        patch_courant = transform_patch(patch.get(k))
        projection_points = dict()
        # PArcous de sommets du modèle d'origine
        # appartenant au patch correspondant à une face du modèle de base
        for i in range(len(patch_courant)) : 
            P = vertices_origine[patch_courant[i]]

            # P_proj = (P.x1) * x1 + (P.x2) * x2 (".": produit scalaire et "*" : produit par un scalaire)
            u = np.dot(P,x1)
            v = np.dot(P,x2)

            P_proj = [u,v,0]

            coord_p_proj = u * x1 + v * x2

            dist_proj = np.linalg.norm(P - coord_p_proj)

            projection_points[patch_courant[i]] = P_proj
        vertices_origine_base.append(projection_points)#.append(dist_proj)
    return vertices_origine_base #, distance_origine_base
    
    
def aera_triangle(v1,v2,v3) : 
    return  v2[0] * v3[1] - v2[1] * v3[0] + v3[0] * v1[1] - v1[0] * v3[1] + v1[0] * v2[1] - v2[0] * v1[1]


#vertices_origine_base : dict(int,[int:4]) (indice du vertex dans le input_mesh, coordonnées projection + distance)
# f : Face
# renvoie la matrice de calcul des poids barycentriques 
def makeBarycentricCoordsMatrix (vertices_origine_base, f) :
        
        C = np.zeros((3,3))
        v1 = vertices_origine_base.get(f.a)
        v2 = vertices_origine_base.get(f.b)
        v3 = vertices_origine_base.get(f.c)
        
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

def determine_patch(bord,model_origine,color,colors,faces_restantes) :
    bord = bord[0] + bord[1][1:] + bord[2][1:]
    B = bord

    A = []
    P = []
    O = []
    K = []
    
    for i in range(len(bord)-1) :
        vi = bord[i]
        viplus1 = bord[i+1]
        for k in faces_restantes.keys() :
            f = faces_restantes.get(k)
            if f.isIn(vi) and f.isIn(viplus1) :
                if f.orientation(vi,viplus1,'h') :
                    if f not in A :
                        A.append(f)
                        P.append(f)
                        K.append(k)
                        colors[f.a] = color
                        colors[f.b] = color
                        colors[f.c] = color
                        
                else :
                    O.append(f)
    while A != [] :
        f = A[0]
        A.remove(f)
        for k in faces_restantes.keys() :
            face = faces_restantes.get(k)
            if (face not in P) and f.adjacent(face) and (face not in O):
                P.append(face)
                A.append(face)
                K.append(k)
                colors[face.a] = color
                colors[face.b] = color
                colors[face.c] = color
    supp_keys(faces_restantes,K)
    return P,colors

def transform_patch(P) :
    P_transforme = []
    for i in range(len(P)) :
        f = P[i]
        if f.a not in P_transforme :
            P_transforme.append(f.a)
        if f.b not in P_transforme :
            P_transforme.append(f.b)
        if f.c not in P_transforme :
            P_transforme.append(f.c)
    return P_transforme 

def partition(bords, model_origine) :
    faces_restantes = convert_dict(model_origine.faces)
    colors = {}
    patch = {}
    p = 0
    for i in range(0,len(bords)-2,3) :
        patch[p],colors = determine_patch(test_bord([bords[i],bords[i+1],bords[i+2]]),model_origine,[i*0.05, i* 0.05, i * 0.05],colors,faces_restantes)
        p+=1
    return patch, colors,faces_restantes

def convert_dict(faces) :
    d = {}
    for k in range(len(faces)) :
        d[k] = faces[k]
    return d

def supp_keys(faces,K) :
    for i in range(len(K)) :
        d = faces.pop(K[i])

def test_bord(bord) :
    if bord[0][1] == bord[2][-2] :
        bord[0] = bord[0][1:]
        bord[2] = bord[2][:-1]
    if bord[0][-2] == bord[1][1] :
        bord[0] = bord[0][:-1]
        bord[1] = bord[1][1:]
    if bord[1][-2] == bord[2][1] :
        bord[1] = bord[1][:-1]
        bord[2] = bord[2][1:]
##    print(bord)
    return bord 

##p,r, input_mesh = pl.get_limit("sphere_input.obj")
##patch,colors,faces_restantes = partition(p,input_mesh)
##
####with open('test_bunny.obja','w') as output :
####    for v in model_origine.vertices : 
####        output.write(f'v {v[0]} {v[1]} {v[2]}\n')
####    for i in range(len(model_origine.faces)) :
####        face = model_origine.faces[i]
####        output.write(f'f {face.a + 1} {face.b + 1} {face.c + 1}\n')
####    for k, c in colors.items() :
####        output.write(f'fc {k + 1} {c[0]} {c[1]} {c[2]}\n')
####        
##
##with open('test_sphere.obj','w') as output :
##    for k in range(len(input_mesh.vertices)) :
##        v = input_mesh.vertices[k]
##        if  k in colors :
##            c = colors.get(k)
##            output.write(f'v {v[0]} {v[1]} {v[2]} {c[0]} {c[1]} {c[2]}\n')
##        else :
##            output.write(f'v {v[0]} {v[1]} {v[2]} 1.0 1.0 1.0\n')
##    for i in range(len(input_mesh.faces)) :
##        face = input_mesh.faces[i]
##        output.write(f'f {face.a + 1} {face.b + 1} {face.c + 1}\n')
##
