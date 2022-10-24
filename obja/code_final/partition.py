from gettext import npgettext

from math import *
import random
import numpy as np
import obja


'''
FONCTION DE CONVERTION 

'''

# def convert_dict
# converti une liste en dictionnaire
# entree : faces = liste de face
# sortie : d = dictionnaire telque d = {position de la face dans la liste : face}
def convert_dict(faces) :
    d = {}
    for k in range(len(faces)) :
        d[k] = faces[k]
    return d

# def supp_keys
# supprime les cles d'un dictionnaire
# entrees : dico = dictionnaire  ; K  = liste de clees à supprimer 
def supp_keys(dico,K) :
    for i in range(len(K)) :
        d = dico.pop(K[i])
# def transform_patch
# transforme un patch de face en une liste d'indices de sommets
# entree : P = patch
# retour : P_transforme = liste d'indices de sommets
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

# def test_bord
# verifie si les arètes d'un bord ne supperpose pas aux extrémitées
# entree : bord = liste (3, x)  
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
    return bord

# def determine_patch
# determine les faces de l'input mesh qui appartiennent au patch
# entrees : - bord = bord du patch
#           - model_origine = input mesh
#           - faces_restantes = dictionnaire des faces qui n'ont pas encore été associées à un patch
#           - orientation = str ('h' ou 'a') qui indique l'orientation des faces d'input mesh
# sortie : P = patch (liste de faces)
def determine_patch(bord,model_origine,faces_restantes,orientation) :
    bord = bord[0] + bord[1][1:] + bord[2][1:]
    A = []
    P = []
    O = []
    K = []

    # Etape 1 : on trouve les faces sur le bord du patch
    for i in range(len(bord)-1) :
        vi = bord[i]
        viplus1 = bord[i+1]
        for k in faces_restantes.keys() :
            f = faces_restantes.get(k)
            if f.isIn(vi) and f.isIn(viplus1) :
                if f.orientation(vi,viplus1,orientation) :
                    if f not in A :
                        A.append(f)
                        P.append(f)
                        K.append(k)
                else :
                    O.append(f)

    # Etape 2 : on trouve les faces à l'intérieur du patch                 
    while A != [] :
        f = A[0]
        A.remove(f)
        for k in faces_restantes.keys() :
            face = faces_restantes.get(k)
            if (face not in P) and f.adjacent(face) and (face not in O):
                P.append(face)
                A.append(face)
                K.append(k)
                
    supp_keys(faces_restantes,K)
    
    return P

# def partition
# associe à chaque patch identifié par un bord l'ensemble des faces à l'intérieur
# entrees : - bords = list de list (nb_patch,x) avec les indices des bords des patchs sur l'input mesh
#           - model_origine = input mesh
#           - model_base = base mesh
#           - r : indice des sommets du base mesh dans l'input mesh
#           - orientation : 'o' ou 'h' qui correspond à l'orientation des faces dans l'input mesh
# retour : - patch : dict(int : list(faces)) (nb_patch, x)
#          - faces_restantes : dictionnaires des faces qui n'ont été identifiée à aucun patch
#          - r : indice des sommets du base mesh (éventuellement modifié)
def partition(bords, model_origine,model_base,r,orientation) :
    faces_restantes = convert_dict(model_origine.faces)
    patch = {}
    p = 0
    for i in range(0,len(bords)-2,3) :
        patch[p] = determine_patch(test_bord([bords[i],bords[i+1],bords[i+2]]),model_origine,faces_restantes,orientation)
        p+=1
    return patch, faces_restantes,r



##def verif_bord(bord1,bord2,bord3) :
##    index13 = -1
##    index12 = -1
##    index23 = -1
##    while bord1[1] == bord3[-2] :
##        bord1 = bord1[1:]
##        bord3 = bord3[:-1]
##        index13 = bord3[-1]
##    while bord1[-2] == bord2[1] :
##        bord1 = bord1[:-1]
##        bord2 = bord2[1:]
##        index12 = bord1[-1]
##    while bord2[-2] == bord3[1] :
##        bord2 = bord2[:-1]
##        bord3 = bord3[1:]
##        index23 = bord2[-1]
##    return index13,index12,index23,[bord1,bord2,bord3]
##
##def modification_patch(bord,model_base,model_origine,r) :
##    r = list(r)
##    old_13 = bord[0][0]
##    old_12 = bord[1][0]
##    old_23 = bord[2][0]
##    index13,index12,index23,bord = verif_bord(bord[0],bord[1],bord[2])
##
##    if index13 != -1 :
##        model_base.vertices[r.index(old_13)] = model_origine.vertices[index13]
##        r[r.index(old_13)] = index13
##    if index12 != -1 :
##        model_base.vertices[r.index(old_12)] = model_origine.vertices[index12]
##        r[r.index(old_12)] = index12
##    if index23 != -1 :
##        model_base.vertices[r.index(old_23)] = model_origine.vertices[index23]
##        r[r.index(old_23)] = index23
##    return bord,model_base,model_origine,np.array(r)
