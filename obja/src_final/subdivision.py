import math
import obja
import numpy as np
import projection as pp
import matplotlib.pyplot as plt

# renvoie le sommet le plus proche du point a
# sommet : np.array([3]) : sommet dont on cherche le plus proche voisin
# correspondance_i : dict(int,[int:3]) : associe sur la base_face i un vertice de l'input_mesh et sa projection
# input_mesh : Model 
def plus_proche(sommet,correspondance_i,input_mesh):
        point = -1
        dist_old = float('inf')
        for k in correspondance_i.keys():    # pour chaque face dans la liste de faces
                dist = np.linalg.norm(sommet-correspondance_i.get(k))
                if  dist < dist_old:
                        point = k
                        dist_old = dist
        #plot_correspondance(correspondance_i,correspondance_i.get(point),patch_i,sommet)
        return input_mesh.vertices[point]

# renvoie la face du input mesh auquel appartient le sommet et les poids barycentriques
# sommet : array(3) : coordonnées relatives dans la face du base_mesh d'un point
# patch_i : list(int) :	liste des faces du input_mesh qui sont projetées dans la face du base mesh que l'on utilise	
# correspondance: dict(int, [int:3]) : 	associe a l'id d'un point du input_mesh, les coordoonées relatives 2D du 
# 										base_mesh ET la distance entre ces 2 points
# poids : [int:3] : poids de calcul pour le barycentre
# face : Face :  face dans le input_mesh (projeté) auquel appartient le point
def recherche_face(sommet,patch_i,correspondance):
        poids = [None,None,None]
        face = -1
        epsilon = - math.pow(10,-2)
        #plot_correspondance(correspondance, sommet,patch_i)
        for k in range(len(patch_i)): # pour chaque face du input
                #on a la face avec les points projetés
                f = patch_i[k]
                C = pp.makeBarycentricCoordsMatrix(correspondance, f)
                s = np.array([1,sommet[0],sommet[1]])
                r = np.dot(C,s) # r représente les poids barycentriques
                #print('r = ',r)
                #print(r[0]>= 0,r[1]>=0 ,r[2]>=0.0)
##                p_recal = r[0] * correspondance.get(f.a) + r[1] * correspondance.get(f.b) + r[2] * correspondance.get(f.c) 
##                if np.linalg.norm(sommet-p_recal) < 0.0001 :
##                        print('ok')
                if r[0] >= epsilon and r[1]>= epsilon and r[2]>=epsilon:
                        face = f
                        poids = r
                        break
        print('Face retournée: ' + str(face))
        return poids,face

# renvoie le determinant d'une matrice 2x2 telle qu'elle est de la forme [u v]
def det_v(u,v) :
    return u[0] * v[1] - u[1] * v[0]

# affichahe des correspondances
# correspondance : list(dict(int,[int:3])) : pour chaque face du base_mesh, associe l'id d'un point et son projeté
# sommet : [int:3] : un point
# patch : dict(int,list(int)) : associe l'indice d'une face du base_mesh aux indices de faces du input_mesh qui sont incluses
def plot_correspondance(correspondance,sommet,patch) :
        plt.plot(sommet[0],sommet[1],'ro')
        X = []
        Y = []
        X_pts = []
        Y_pts = []
        for k in range(len(patch)) :
                f = patch[k]
                X_pts.append(correspondance.get(f.a)[0])
                X_pts.append(correspondance.get(f.b)[0])
                X_pts.append(correspondance.get(f.c)[0])

                Y_pts.append(correspondance.get(f.a)[1])
                Y_pts.append(correspondance.get(f.b)[1])
                Y_pts.append(correspondance.get(f.c)[1])
        for k in range(len(patch)) :
                f = patch[k]
                X.append(correspondance.get(f.a)[0])
                X.append(correspondance.get(f.b)[0])
                X.append(correspondance.get(f.c)[0])
                X.append(correspondance.get(f.a)[0])
                Y.append(correspondance.get(f.a)[1])
                Y.append(correspondance.get(f.b)[1])
                Y.append(correspondance.get(f.c)[1])
                Y.append(correspondance.get(f.a)[1])
                plt.plot(X,Y)
                X = []
                Y = []
        plt.plot(X_pts,Y_pts,'bo')
        plt.show()

# calcule la distance d'un point à un plan
# point : np[int:3] : point
# face : Face : la face qui nous permet de définir le plan
def distance_plan(point,face):
        # pour calculer la distance d'un point à un plan il faut d'abord projeter ce point dans le plan
        # soit les vecteurs x1 et x2
        x1 = face.a - face.c
        x2 = face.b - face.c
        point_proj = np.dot(point,x1)*x1 + np.dot(point,x2)*x2

        # et ensuite on calcule la distance de point à point_proj
        return np.linalg.norm(point-point_proj)

# soit on a déjà les points juste pour la face qu'on analyse (comment on met à jour)
# soit on doit récuperer uniquement les points dans la face en utilisant les coordonnées barycentriques
# f : int : index  de la face inter_mesh (ou final_mesh c'est pareil)
# correspondance : dict(int,[int:4]) : associe des points du input_mesh  et leurs projections dans la face du base_mesh correspondants (et distance)
# input_mesh : Model : input_mesh
# final_mesh : Output 
# inter_mesh : Output
# patch : dict(int,list(int)) : associe l'indice d'une face du base_mesh aux indices de faces du input_mesh qui sont incluses
# f_inter : int : index de la face dans le inter et final_mesh
# f_base : int : index de la face du base_mesh correspondant
def erreur(f_inter,f_base,patch,input_mesh,final_mesh,inter_mesh,correspondance):
        max = float('-inf')
        face_inter = inter_mesh.faces[f_inter]
        face_final = final_mesh.faces[f_inter]
        # pour toutes les faces projetés dans la face du base_mesh
        for f in range(len(patch.get(f_base))):
                face = input_mesh.faces[f] # Face
                # pour chaque sommet de cette face
                # on récupère les coordonnées projetées
                a_projete = correspondance[face.a][0:3]
                # on voit si ce point est dans la face avec les coordonées du inter_mesh
                if face_inter.isInside(correspondance,a_projete):
                        #on calcule donc la distance de ce point à la face face_final
                        dist = distance_plan(face.a,face_final)
                        max = max if dist>max else max

                b_projete = correspondance[face.b][0:3]
                if face_inter.isInside(correspondance,b_projete):
                        #on calcule donc la distance de ce point à la face face_final
                        dist = distance_plan(face.b,face_final)
                        max = max if dist>max else max

                c_projete = correspondance[face.c][0:3]
                if face_inter.isInside(correspondance,c_projete):
                        #on calcule donc la distance de ce point à la face face_final
                        dist = distance_plan(face.c,face_final)
                        max = max if dist>max else max

        return max

# permet de trouver la bounding box du input_mesh en cherchant les deux coins les plus éloignés
# on cherche donc toutes les valeurs min et toutes les valeurs max
# input_mesh : Model : le mesh d'entrée
def bound_box(input_mesh):
        minx = float('inf')
        miny = float('inf')
        minz = float('inf')
        maxx = float('-inf')
        maxy = float('-inf')
        maxz = float('-inf')
        for vertex in input_mesh.vertices:
                x = vertex[0]
                y = vertex[1]
                z = vertex[2]
                minx = x if x < minx else minx
                miny = y if y < miny else miny
                minz = z if z < minz else minz
                maxx = x if x > maxx else maxx
                maxy = y if y > maxy else maxy
                maxz = z if z > maxz else maxz
        return [minx,miny,minz],[maxx,maxy,maxz]

# travaille la subdivision jusqu'à obtenir un mesh semi_régulier
# input_mesh : 	Model :		le mesh d'origine (coordonées des sommets dans repère monde)
# base_mesh : 	Model : 	le base_mesh (coordonnées des sommets dans repère monde)
# patch :		dict(int,list(int))	:		associe l'indice d'une face du base_mesh aux indices de faces du input_mesh qui sont incluses 
# correspondance : 	dict(int,[int,4]) : 	associe l'indice d'un sommet du input_mesh aux coordoonées RELATIVES (base: (s3:s1,s2))
#									 		de sa projection dans la bonne face et la DISTANCE entre ces points
# TODO FAIRE DES FACTORISATIONS DE CODE
def subdivision(input_mesh, base_mesh,patch, correspondance,r,kmax):
        # initialisation : liste L avec toutes les faces du mesh
        # on commence déjà par créer le modèle sur lequel on va travailler, qui est une copie du base_mesh
        output = open("sphere_final_mesh.obja",'w') 
        final_mesh = obja.Output(output,True)
        inter = open("sphere_inter_mesh.obja",'w')
        inter_mesh = obja.Output(inter) #le inter_mesh représente le final mesh, mais strictement dans les repères relatifs du base_mesh

        # on va aussi initialiser la liste sur laquelle on va travailler qui va avoir les indices correspondant aux faces
        # en fait ce sera un dictionnaire avec en clé l'indice de face du final_mesh, et en valeurs l'indice de la face du base_mesh auquel elle appartient
        L = dict()
        borders = dict()
        # on va tout copier
        for i in range(len(base_mesh.faces)): # pour chaque face du base mesh
                face = base_mesh.faces[i]	# face : Face
                # on ajoute les sommets au final_mesh
                # rappel : une face est l'association de 3 indices des sommets le formant
                [ida,idb,idc] = [face.a,face.b,face.c]

                # on ajoute dans final_mesh les sommets (attention il faut les copier)
                final_mesh.add_vertex(ida,base_mesh.vertices[ida][0:3])
                final_mesh.add_vertex(idb,base_mesh.vertices[idb][0:3])
                final_mesh.add_vertex(idc,base_mesh.vertices[idc][0:3])

                # il faut récupérer les coordonnées des points dans l'input mesh projeté pour le patch correspondant

                x1,x2,n = pp.calcul_base(base_mesh.vertices[ida][0:3],base_mesh.vertices[idb][0:3],base_mesh.vertices[idc][0:3] )

                if r[i][1] in correspondance[i] :        
                        inter_mesh.add_vertex(ida,correspondance[i].get(r[i][1]),i)
                else :
                        p_projete, coord_p = pp.projection_point(x1,x2,base_mesh.vertices[ida][0:3])
                        inter_mesh.add_vertex(ida,p_projete,i)

                if r[i][2] in correspondance[i] :        
                        inter_mesh.add_vertex(idb,correspondance[i].get(r[i][2]),i)
                else :
                        p_projete, coord_p = pp.projection_point(x1,x2,base_mesh.vertices[idb][0:3])
                        inter_mesh.add_vertex(idb,p_projete,i)
                        
                if r[i][3] in correspondance[i] :        
                        inter_mesh.add_vertex(idc,correspondance[i].get(r[i][3]),i)
                else :
                        p_projete, coord_p = pp.projection_point(x1,x2,base_mesh.vertices[idc][0:3])
                        inter_mesh.add_vertex(idc,p_projete,i)

                #maintenant on peut ajouter la face dans final_mesh
                final_mesh.add_face(i,obja.Face(ida,idb,idc))	# vu que normalement c'est les mêmes indices
                inter_mesh.add_face(i,obja.Face(ida,idb,idc))

                # on peut aussi ajouter les indices des faces directement dans la liste L
                L[i] = i
                borders[frozenset((ida,idb))] = [False,-1] #le frozenset va permettre des egalités sans se soucier de l'ordre
                borders[frozenset((idb,idc))] = [False,-1]
                borders[frozenset((ida,idc))] = [False,-1] # on associe un chiffre qui sera l'index de l'autre élément à la frontière


        indice_f = len(L)
        indice_v = 3*(indice_f+1)
        # on calcul aussi la longueur de la diagonale de la bounding_box du input_mesh
        min_val,max_val = bound_box(input_mesh)
        diag_bound = math.dist(min_val,max_val)
        Seuil = 1
        
        iter = 0
        
        # une fois l'initialisation faite nous pouvons procéder à la subdivision à proprement parler
        # on va boucler sur les étapes suivantes tant qu'il reste une face à traiter
        while len(L) > 0 and iter < kmax:
                iter+=1
                E = dict()
                to_add = []
                to_del = []
                borders = dict.fromkeys(borders, [False,-1])
                for index_face_final in L.keys():	# pour chaque face (du final_mesh) restante 
                        # on calcul l'erreur E(f)
                        #E[index_face_final] = erreur(index_face_final,L.get(index_face_final),patch,input_mesh,final_mesh,inter_mesh,correspondance[index_face_final]/diag_bound)

##                        # si cette erreur est inférieure à un seuil, on considère la face suffisamment proche du input_mesh
##                        if E[index_face_final] < Seuil :
##                                to_del.append(index_face_final)	
##                                #del L[index_face_final]
##                                continue
##                        else :
                                # on travail avec les coordonnées 2D relatives
                                # on divise chaque face f en 4 triangles 
                                # rappel; base(s3:s1,s2)
                        f = inter_mesh.faces.get(index_face_final)
                        [i1,i2,i3] = [f.a,f.b,f.c]
                        s1 = inter_mesh.vertices.get((i1,L[index_face_final]))
                        s2 = inter_mesh.vertices.get((i2,L[index_face_final]))
                        s3 = inter_mesh.vertices.get((i3,L[index_face_final]))

                        a = 0.5*s3 + 0.5*s1	#barycentre s3 s1
                        b = 0.5*s3 + 0.5*s2	#barycentre s3 s2
                        c = 0.5*s1 + 0.5*s2	# barycentre s1 s2

                        #pour chacune de ces nouveaux sommets on cherche à quel face de l'input_mesh ils appartiennent


                        (poids_a,fa) = recherche_face(a,patch.get(L.get(index_face_final)),correspondance[L.get(index_face_final)])
                        (poids_b,fb) = recherche_face(b,patch.get(L.get(index_face_final)),correspondance[L.get(index_face_final)])
                        (poids_c,fc) = recherche_face(c,patch.get(L.get(index_face_final)),correspondance[L.get(index_face_final)])
                        # on a donc les coordonnées dans le plan relatif à la face du base_mesh 

                        #il faut donc mtn stocker ces valeurs dans le final_mesh
                        #on commence on crée les 3 nouveaux vertex dans le final_mesh
                        [idv1,idv2,idv3] = [indice_v+1,indice_v+2,indice_v+3] #nroamelemnt les valeurs cont changer
                        if fa != -1 :
                                point = poids_a[0]*input_mesh.vertices[fa.a] + poids_a[1]*input_mesh.vertices[fa.b] + poids_a[2]*input_mesh.vertices[fa.c]
                                match borders.get(frozenset((i1,i3))):
                                        case None:
                                                indice_v +=1
                                                final_mesh.add_vertex(indice_v,point)
                                                idv1 = indice_v
                                        case [True ,id] : 
                                                idv1 = id
                                                final_mesh.edit_vertex(idv1,0.5*final_mesh.vertices[(idv1,None)]+0.5*point)
                                                del borders[frozenset((i1,i3))]
                                        case [False ,id] :
                                                indice_v +=1
                                                final_mesh.add_vertex(indice_v,point)
                                                borders[frozenset((i1,indice_v))] = [False,-1]
                                                borders[frozenset((i3,indice_v))] = [False,-1]
                                                borders[frozenset((i1,i3))] = [True,indice_v]
                                                idv1 = indice_v
                                                
                        else :
                                point = plus_proche(a,patch.get(L.get(index_face_final)),correspondance[L.get(index_face_final)],input_mesh)
                                #point = 0.5*final_mesh.vertices[i3] + 0.5*final_mesh.vertices[i1]
                                match borders.get(frozenset((i1,i3))):
                                        case None:
                                                indice_v +=1
                                                idv1 = indice_v
                                                final_mesh.add_vertex(indice_v,point)
                                        case [True ,id] : 
                                                idv1 = id
                                                final_mesh.edit_vertex(idv1,0.5*final_mesh.vertices[(idv1,None)]+0.5*point)
                                                del borders[frozenset((i1,i3))]
                                        case [False ,id] :
                                                indice_v +=1
                                                idv1 = indice_v
                                                final_mesh.add_vertex(indice_v,point)
                                                borders[frozenset((i1,indice_v))] = [False,-1]
                                                borders[frozenset((i3,indice_v))] = [False,-1]
                                                borders[frozenset((i1,i3))] = [True,indice_v]
                        if fb != -1 :
                                point = poids_b[0]*input_mesh.vertices[fb.a] + poids_b[1]*input_mesh.vertices[fb.b] + poids_b[2]*input_mesh.vertices[fb.c]
                                match borders.get(frozenset((i2,i3))):
                                        case None:
                                                indice_v +=1
                                                idv2 = indice_v
                                                final_mesh.add_vertex(indice_v,point)
                                        case [True ,id]: 
                                                idv2 = id
                                                final_mesh.edit_vertex(idv2,0.5*final_mesh.vertices[(idv2,None)]+0.5*point)
                                                del borders[frozenset((i2,i3))]
                                        case [False,id] :
                                                indice_v +=1
                                                idv2 = indice_v
                                                final_mesh.add_vertex(indice_v,point)
                                                borders[frozenset((i2,indice_v))] = [False,-1]
                                                borders[frozenset((i3,indice_v))] = [False,-1]
                                                borders[frozenset((i2,i3))] = [True,indice_v]
                                                
                        else :
                                #point = 0.5*final_mesh.vertices[i3] + 0.5*final_mesh.vertices[i2]
                                point = plus_proche(b,patch.get(L.get(index_face_final)),correspondance[L.get(index_face_final)],input_mesh)
                                match borders.get(frozenset((i2,i3))):
                                        case None:
                                                indice_v +=1
                                                idv2 = indice_v
                                                final_mesh.add_vertex(indice_v,point)
                                        case [True ,id] : 
        
                                                idv2 = id
                                                final_mesh.edit_vertex(idv2,0.5*final_mesh.vertices[(idv2,None)]+0.5*point)
                                                del borders[frozenset((i2,i3))]
                                        case [False ,id] :
                                                indice_v +=1
                                                idv2 = indice_v
                                                final_mesh.add_vertex(indice_v,point)
                                                borders[frozenset((i2,indice_v))] = [False,-1]
                                                borders[frozenset((i3,indice_v))] = [False,-1]
                                                borders[frozenset((i2,i3))] = [True,indice_v]
                        if fc != -1 :
                                point = poids_c[0]*input_mesh.vertices[fc.a] + poids_c[1]*input_mesh.vertices[fc.b] + poids_c[2]*input_mesh.vertices[fc.c]
                                match borders.get(frozenset((i1,i2))):
                                        case None:
                                                indice_v +=1
                                                idv3 = indice_v
                                                final_mesh.add_vertex(indice_v,point)
                                        case [True ,id] : 
                                                idv3 = id
                                                final_mesh.edit_vertex(idv3,0.5*final_mesh.vertices[(idv3,None)]+0.5*point)
                                                del borders[frozenset((i1,i2))]
                                        case [False ,id] :
                                                indice_v +=1
                                                idv3 = indice_v
                                                final_mesh.add_vertex(indice_v,point)
                                                borders[frozenset((i1,indice_v))] = [False,-1]
                                                borders[frozenset((i2,indice_v))] = [False,-1]
                                                borders[frozenset((i1,i2))] = [True,indice_v]
                                                
                        else :
                                #point = 0.5*final_mesh.vertices[i2] + 0.5*final_mesh.vertices[i1]
                                point = plus_proche(c,patch.get(L.get(index_face_final)),correspondance[L.get(index_face_final)],input_mesh)
                                match borders.get(frozenset((i1,i2))):
                                        case None:
                                                indice_v +=1
                                                idv3 = indice_v
                                                final_mesh.add_vertex(indice_v,point)
                                        case [True ,id] : 
                                                idv3 = id
                                                final_mesh.edit_vertex(idv3,0.5*final_mesh.vertices[(idv3,None)]+0.5*point)
                                                del borders[frozenset((i1,i2))]
                                        case [False ,id] :
                                                indice_v +=1
                                                idv3 = indice_v
                                                final_mesh.add_vertex(indice_v,point)
                                                borders[frozenset((i1,indice_v))] = [False,-1]
                                                borders[frozenset((i2,indice_v))] = [False,-1]
                                                borders[frozenset((i1,i2))] = [True,indice_v]
                        

                        #on peut donc mtn créer les 4 nouvelles faces du final_mesh
                        [idf1,idf2,idf3,idf4] = [indice_f+1,indice_f+2,indice_f+3,indice_f+4]
                        final_mesh.add_face(idf1,obja.Face(i1,idv3,idv1))
                        final_mesh.add_face(idf2,obja.Face(idv3,i2,idv2))
                        final_mesh.add_face(idf3,obja.Face(idv3,idv2,idv1))
                        final_mesh.add_face(idf4,obja.Face(idv1,idv2,i3))
                        indice_f+=4

                        #finalement on retire la face sur laquelle on vient de travailler
                        face_base = L.get(index_face_final)
                        final_mesh.delete_face(index_face_final)


                        #avec les mêmes indices on va le faire pour le intermesh
                        inter_mesh.delete_face(index_face_final)
                        inter_mesh.add_vertex(idv1,a,L[index_face_final])
                        inter_mesh.add_vertex(idv2,b,L[index_face_final])
                        inter_mesh.add_vertex(idv3,c,L[index_face_final])

                        inter_mesh.add_face(idf1,obja.Face(i1,idv3,idv1))
                        inter_mesh.add_face(idf2,obja.Face(idv3,i2,idv2))
                        inter_mesh.add_face(idf3,obja.Face(idv3,idv2,idv1))
                        inter_mesh.add_face(idf4,obja.Face(idv1,idv2,i3))

                        to_del.append(index_face_final)

                        #pour ajouter les 4 nouvelles
                        to_add.extend([[idf1,face_base],[idf2,face_base],[idf3,face_base],[idf4,face_base]])


                        
                for delete in to_del:
                        del L[delete]
                for k in range(len(to_add)):
                        L[to_add[k][0]] = to_add[k][1]
                
        return final_mesh
