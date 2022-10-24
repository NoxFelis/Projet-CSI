from gettext import npgettext

from math import *
import random
import numpy as np
import obja
import partition as pt

'''
PARTIE PROJECTION : ensemble des fonctions qui permettent de réaliser la projection
entre l'input mesh et le base mesh

'''
# calcul la base d'une face
# entrees : a1,a2,a3 = sommets de la face
# retour : x1,x2,n = vecteur de la base orthonormée
def calcul_base(a1,a2,a3) :
    # Création de la base de la face sur laquelle on projette
    x1 = a2[0:3] - a1[0:3]
    x2 = a3[0:3] - a1[0:3]

    
    x1 = (1/np.linalg.norm(x1)) * x1
    x2 = (1/np.linalg.norm(x2)) * x2
    
    #On orthonormalise la base
    x2 = x2 - (1/np.linalg.norm(x1)) * np.dot(x2,x1) * x1
    x2 = (1/np.linalg.norm(x2)) * x2

    n = np.cross(x1,x2)
    n = (1/np.linalg.norm(n)) * n
    
    return x1,x2,n

## def projection_point
## projection d'un point sur un plan
## entrees : x1, x2 = vecteur orthonormée d'une base du plan ; P = point à projeter
## retour : P_proj = [u,v,0] où u et v sont les coordonnées 2D du point;
##          coord3D_P_proj = coordonnees 3D du point projetée 
def projection_point(x1,x2,P) :
    u = np.dot(P,x1)
    v = np.dot(P,x2)

    P_proj = np.array([u,v,0])
    coord3D_P_proj = u * x1 + v * x2

    return P_proj,coord3D_P_proj
    

## def projection 
## projection des points de chaque patch sur la face du base mesh correspondant
## entrees : - model_base :  model de base 
##           - model_origine : model d'origine
##           - patch : matrice contenant les vertices appartenant aux différents patch 
## retour : matrice retournant les coordonnées des vertices du mesh d'origine dans le mesh de base 
def projection(model_origine,model_base,patch):
   
    faces_base = model_base.faces
    vertices_base = model_base.vertices
    faces_origine = model_origine.faces
    vertices_origine = model_origine.vertices

    vertices_origine_base = []
    distance = []
    
    # Parcours de face du maillage de base 
    for k in range(len(faces_base)) :

        f = faces_base[k]

        #calcul de la base de la face
        x1,x2,n = calcul_base(vertices_base[f.a],vertices_base[f.b],vertices_base[f.c])

        #calcul barycentre la face
        barycentre_face = 1/3 * (vertices_base[f.a]+ vertices_base[f.b] + vertices_base[f.c])
        
        patch_courant = pt.transform_patch(patch.get(k))
        projection_points = dict()
        distance_proj = dict()
        
        # parcous de sommets du modèle d'origine
        # appartenant au patch correspondant à une face du modèle de base
        """ for i in range(len(patch_courant)) :
            
            P = vertices_origine[patch_courant[i]]

            P_proj, coord_p_proj = projection_point(x1,x2,P)
            
            dist_proj = np.linalg.norm(P - (coord_p_proj + barrycentre_face))

            projection_points[patch_courant[i]] = P_proj
            distance_proj[patch_courant[i]] = dist_proj """
        
        vertices_origine_base.append(projection_points)
        distance.append(distance_proj)

    return vertices_origine_base, distance

'''
PARTIE COORDONNEES BARYCENTRIQUE : permet le calcul des coordonnees barycentriques
'''
# def aera_triangle
# calcul l'aire d'un triangle
# entrees : v1,v2,v3 = coordonnees 2D ([u,v,0]) des sommets du triangle
# retour : air du triangle 
def aera_triangle(v1,v2,v3) : 
    return  (v2[0] * v3[1]) - (v2[1] * v3[0]) + (v3[0] * v1[1]) - (v1[0] * v3[1]) + (v1[0] * v2[1]) - (v2[0] * v1[1])


# def makeBarycentricCoordsMatrix
# calcul la matrice qui permet d'obtenir les coordonnées barycentriques d'un point
# dans une face
# entrees: - vertices_origine_base : dict(int,[int:3]) (indice du vertex dans le input_mesh, coordonnées projection + distance)
#          - f : Face
# retour : C = la matrice de calcul des poids barycentriques 
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

        C[0,0] = ((x2 * y3) - (x3 * y2))
        C[0,1] = (y2 - y3) 
        C[0,2] = (x3 - x2) 
        C[1,0] = ((x3 * y1) - (x1 * y3)) 
        C[1,1] = (y3 - y1) 
        C[1,2] = (x1 - x3) 
        C[2,0] = ((x1 * y2) - (x2 * y1)) 
        C[2,1] = (y1 - y2) 
        C[2,2] = (x2 - x1) 


        return (1/aera) * C 
