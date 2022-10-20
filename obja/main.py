import math
import obja
import numpy as np
import os
from os import getcwd
import csi_code as cc

DATA_DIR = getcwd()

# path : string : path of entry mesh
def division_4(path):
	input_mesh = obja.parse_file(path)

	mesh = obja.Output()
	# on va tout copier
	for i in len(input_mesh.faces): # pour chaque face du base mesh
		face = input_mesh.faces[i]	# face : Face

		# on ajoute les sommets au final_mesh
		# rappel : une face est l'association de 3 indices des sommets le formant
		[ida,idb,idc] = [face.a,face.b,face.c]

		# on ajoute dans final_mesh les sommets (attention il faut les copier)
		mesh.add_vertex(ida,input_mesh.vertices[ida])
		mesh.add_vertex(idb,input_mesh.vertices[idb])
		mesh.add_vertex(idc,input_mesh.vertices[idc])

		#maintenant on peut ajouter la face dans final_mesh
		mesh.add_face(i,face)	# vu que normalement c'est les mêmes indices


	indice_f = len(mesh.faces)
	indice_v = len(mesh.vertices)
	for f in mesh.faces.keys():
		[s1,s2,s3] = mesh.faces[f]
		a = np.sum(0.5*s3,0.5*s1)	#barycentre s3 s1
		b = np.sum(0.5*s3,0.5*s2)	#barycentre s3 s2
		c = np.sum(0.5*s1,0.5*s2)	# barycentre s1 s2

		# on ajoute les 3 points au mesh
		[idv1,idv2,idv3] = [indice_v+1,indice_v+2,indice_v+3]
		mesh.add_vertex(idv1,a)
		mesh.add_vertex(idv2,b)
		mesh.add_vertex(idv3,c)
		indice_v += 3

		#on peut donc mtn créer les 4 nouvelles faces du mesh
		[idf1,idf2,idf3,idf4] = [indice_f+1,indice_f+2,indice_f+3,indice_f+4]
		mesh.add_face(idf1,[s1,idv3,idv1])
		mesh.add_face(idf2,[idv3,s2,idv2])
		mesh.add_face(idf3,[idv3,idv2,idv1])
		mesh.add_face(idf4,[idv1,idv2,s3])
		indice_f+=4

		# et il faut mtn enlever la première face du mesh
		mesh.delete_face(f)
	return mesh

# renvoie la face du input mesh auquel appartient le sommet et les poids barycentriques
# sommet : array(3) : coordonnées relatives dans la face du base_mesh d'un point
# patch_i : list(int) :	liste des faces du input_mesh qui sont projetées dans la face du base mesh que l'on utilise	
# correspondance: dict(int, [int:4]) : 	associe a l'id d'un point du input_mesh, les coordoonées relatives 2D du 
# 										base_mesh ET la distance entre ces 2 points
# poids : [int:3] : poids de calcul pour le barycentre
# face : int : id de la face dans le input_mesh (projeté) auquel appartient le point
def recherche_face(sommet,patch_i,correspondance):
	poids = [None,None,None]
	face = -1
	for f in patch_i:	# pour chaque face du input
		#on a la face avec les points projetés
		projected_f = obja.Face(correspondance[f.a][0:2],correspondance[f.b][0:2],correspondance[f.c][0:2])
		C = cc.makeBarycentricCoordsMatrix(sommet, projected_f) 
		r = C.dot(sommet) # r représente les poids barycentriques
		if r[0] >= 0 and r[1]>=0 and r[2]>=0:
			face = f
			poids = r
			break
	return poids,face

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
	face_inter = inter_mesh.face[f_inter]
	face_final = final_mesh.face[f_inter]
	# pour toutes les faces projetés dans la face du base_mesh
	for f in patch[f_base]:
		face = input_mesh.face[f] # Face
		# pour chaque sommet de cette face
		# on récupère les coordonnées projetées
		a_projete = correspondance[face.a][0:2]
		# on voit si ce point est dans la face avec les coordonées du inter_mesh
		if face_inter.isInside(a_projete):
			#on calcule donc la distance de ce point à la face face_final
			dist = distance_plan(face.a,face_final)
			max = max if dist>max else max

		b_projete = correspondance[face.b][0:2]
		if face_inter.isInside(b_projete):
			#on calcule donc la distance de ce point à la face face_final
			dist = distance_plan(face.b,face_final)
			max = max if dist>max else max

		c_projete = correspondance[face.c][0:2]
		if face_inter.isInside(c_projete):
			#on calcule donc la distance de ce point à la face face_final
			dist = distance_plan(face.c,face_final)
			max = max if dist>max else max
	
	return max

""" def projection(input_mesh, base_mesh, patch):
	# une fois qu'on à la répartition pour les patchs on peut projeter les points sur les faces du base mesh
	# ATTENTION : les coordonnées ici seront toutes relatives à la face sur laquelle elles sont projetées
	correspondance = dict()
	# pour chaque face du base_mesh
	for face in patch.keys(): # face = indice de la face sur le base_mesh
		# pour chaque face du input_mesh à projeter sur le base_mesh
		for face_input in patch[face]: # face_input : indice de la face sur l'input mesh
			[ida,idb,idc] = base_mesh.face[face] # on récupère les indices de sommets de la face

			# on va à partir de ces 3 sommets, créer deux vecteur x1 et x2
			# ATTENTION, ces vecteurs sont désormais notre base tq:
			# x1 = a-c, x2 = b-c et base(c,x1,x2)
			x1 = np.substract(base_mesh.vertices[ida],base_mesh.vertices[idc])
			x2 = np.substract(base_mesh.vertices[idb],base_mesh.vertices[idc])

			for idp in input_mesh.face[face_input]: # pour chaque point de la face dans le input_mesh
				# on projette le point sur la face dur base_mesh
				p = input_mesh.vertices[idp]
				x = np.dot(p,x1)
				y = np.dot(p,x2)
				p_prim = np.array(x,y,0)
				correspondance[idp] = p_prim

	return correspondance """

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
def subdivision(input_mesh, base_mesh,patch, correspondance):
	# initialisation : liste L avec toutes les faces du mesh
	# on commence déjà par créer le modèle sur lequel on va travailler, qui est une copie du base_mesh
	final_mesh = obja.Output()
	inter_mesh = obja.Output() #le inter_mesh représente le final mesh, mais strictement dans les repères relatifs du base_mesh

	# on va aussi initialiser la liste sur laquelle on va travailler qui va avoir les indices correspondant aux faces
	# en fait ce sera un dictionnaire avec en clé l'indice de face du final_mesh, et en valeurs l'indice de la face du base_mesh auquel elle appartient
	L = dict();

	# on va tout copier
	for i in len(base_mesh.faces): # pour chaque face du base mesh
		face = base_mesh.faces[i]	# face : Face

		# on ajoute les sommets au final_mesh
		# rappel : une face est l'association de 3 indices des sommets le formant
		[ida,idb,idc] = [face.a,face.b,face.c]

		# on ajoute dans final_mesh les sommets (attention il faut les copier)
		final_mesh.add_vertex(ida,base_mesh.vertices[ida])
		final_mesh.add_vertex(idb,base_mesh.vertices[idb])
		final_mesh.add_vertex(idc,base_mesh.vertices[idc])

		inter_mesh.add_vertex(ida,[1,0,0])
		inter_mesh.add_vertex(idb,[0,1,0])
		inter_mesh.add_vertex(idc,[0,0,0])

		#maintenant on peut ajouter la face dans final_mesh
		final_mesh.add_face(i,face)	# vu que normalement c'est les mêmes indices
		inter_mesh.add_face(i,face)

		# on peut aussi ajouter les indices des faces directement dans la liste L
		L[i] = i
		
	
	indice_f = len(L)
	indice_v = 3*(indice_f+1)
	# on calcul aussi la longueur de la diagonale de la bounding_box du input_mesh
	min_val,max_val = bound_box(input_mesh)
	diag_bound = math.dist(min_val,max_val)
	Seuil = 1

	# une fois l'initialisation faite nous pouvons procéder à la subdivision à proprement parler
	# on va boucler sur les étapes suivantes tant qu'il reste une face à traiter
	while len(L) > 0:
		
		E = dict()
		for index_face_final in L.keys():	# pour chaque face (du final_mesh) restante 
			
			# on calcul l'erreur E(f)
			E[index_face_final] = erreur(index_face_final,L[index_face_final],patch,input_mesh,final_mesh,inter_mesh,correspondance)/diag_bound
			
			# si cette erreur est inférieure à un seuil, on considère la face suffisamment proche du input_mesh
			if E[index_face_final] < Seuil :	
				del L[index_face_final]
				continue
			
			# on travail avec les coordonnées 2D relatives
			# on divise chaque face f en 4 triangles 
			# rappel; base(s3:s1,s2)
			[s1,s2,s3] = inter_mesh.faces[index_face_final]
			a = np.sum(0.5*s3,0.5*s1)	#barycentre s3 s1
			b = np.sum(0.5*s3,0.5*s2)	#barycentre s3 s2
			c = np.sum(0.5*s1,0.5*s2)	# barycentre s1 s2

			#pour chacune de ces nouveaux sommets on cherche à quel face de l'input_mesh ils appartiennent
			(poids_a,fa) = recherche_face(a,patch[L[index_face_final]],correspondance)
			(poids_b,fb) = recherche_face(b,patch[L[index_face_final]],correspondance)
			(poids_c,fc) = recherche_face(c,patch[L[index_face_final]],correspondance)
			# on a donc les coordonnées dans le plan relatif à la face du base_mesh 

			#il faut donc mtn stocker ces valeurs dans le final_mesh
			#on commence on crée les 3 nouveaux vertex dans le final_mesh
			[idv1,idv2,idv3] = [indice_v+1,indice_v+2,indice_v+3]
			final_mesh.add_vertex(idv1,np.sum(np.sum(poids_a[0]*input_mesh.vertices[fa.a],poids_a[1]*input_mesh.vertices[fa.b]),poids_a[2]*input_mesh.vertices[fa.c]))
			final_mesh.add_vertex(idv2,np.sum(np.sum(poids_b[0]*input_mesh.vertices[fb.a],poids_b[1]*input_mesh.vertices[fb.b]),poids_b[2]*input_mesh.vertices[fb.c]))
			final_mesh.add_vertex(idv3,np.sum(np.sum(poids_c[0]*input_mesh.vertices[fc.a],poids_c[1]*input_mesh.vertices[fc.b]),poids_c[2]*input_mesh.vertices[fc.c]))
			indice_v += 3

			#on peut donc mtn créer les 4 nouvelles faces du final_mesh
			[idf1,idf2,idf3,idf4] = [indice_f+1,indice_f+2,indice_f+3,indice_f+4]
			final_mesh.add_face(idf1,[s1,idv3,idv1])
			final_mesh.add_face(idf2,[idv3,s2,idv2])
			final_mesh.add_face(idf3,[idv3,idv2,idv1])
			final_mesh.add_face(idf4,[idv1,idv2,s3])
			indice_f+=4

			#finalement on retire la face sur laquelle on vient de travailler
			face_base = L[index_face_final]
			final_mesh.delete_face(index_face_final)
			

			#avec les mêmes indices on va le faire pour le intermesh
			inter_mesh.delete_face(index_face_final)
			inter_mesh.add_vertex(idv1,a)
			inter_mesh.add_vertex(idv2,b)
			inter_mesh.add_vertex(idv3,c)

			inter_mesh.add_face(idf1,[s1,idv3,idv1])
			inter_mesh.add_face(idf2,[idv3,s2,idv2])
			inter_mesh.add_face(idf3,[idv3,idv2,idv1])
			inter_mesh.add_face(idf4,[idv1,idv2,s3])

			del L[index_face_final]
			#pour ajouter les 4 nouvelles
			L[idf1] = face_base
			L[idf2] = face_base
			L[idf3] = face_base
			L[idf4] = face_base

	return final_mesh

def main(args=None):
	# 1 - On récupère l'objet dans son maillage irrégulier 
	if args is None:
		path = DATA_DIR + "/example/suzanne.obj"
	else :
		path = DATA_DIR + "/" + args
	# partie faite par Yang
	# model de type obja.Model
	input_mesh = obja.parse_file(path)	# input_mesh : model

	# 2 - On crée le maillage de base 
	# pour l'instant cette étape n'est pas gérée, on doit le faire à la main
	# et on utilisera un code fourni
	base_mesh = simplify_mesh(input_mesh) #base_mesh : Output

	# 3 - On partitionne le maillage d'origine
	# sur le maillage de base (patchs)
	# partie faite par Jade
	patch = patcher(input_mesh,base_mesh) # patch : dict(face,list(faces))

	# 4 - On projète les patchs sur le maillage de base
	correspondance = cc.projection(input_mesh, base_mesh, patch) # correspondance: dict(id vertex 3D, vertex2D)
	
	# 5 - On travaille la subdivision
	final_mesh = subdivision(input_mesh, base_mesh,patch, correspondance) # final_mesh : Output

	# on retourne le maillage semi-régulier
	print(args)
	return final_mesh

if __name__ == '__main__':
    main()
