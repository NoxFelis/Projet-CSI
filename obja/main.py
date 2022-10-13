import obja
import numpy as np
import os
from os import getcwd
import csi_code

DATA_DIR = getcwd()

def patcher(input_mesh,base_mesh):
	return patch

# renvoie la face du input mesh auquel appartient le sommet
# sommet : array(3) : coordonnées relatives dans la face du base_mesh d'un point
# patch_i : list(int) :	liste des faces du input_mesh qui sont projetées dans la face du base mesh que l'on utilise	
def recherche_face(sommet,patch_i,correspondance):
	coord_barycentre_sommet = [-1,-1,-1]
	for face in patch_i:	# pour chaque face du input 
		C = makeBarycentricCoordsMatrix (sommet, face)
	return coord_barycentre_sommet

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

# travaille la subdivision jusqu'à obtenir un mesh semi_régulier
# input_mesh : 	Model :		le mesh d'origine (coordonées des sommets dans repère monde)
# base_mesh : 	Model : 	le base_mesh (coordonnées des sommets dans repère monde)
# patch :		dict(int,list(int))	:		associe l'indice d'une face du base_mesh aux indices de faces du input_mesh qui sont incluses 
# correspondance : 	dict(int,[int,4]) : 	associe l'indice d'un sommet du input_mesh aux coordoonées RELATIVES (base: (s3:s1,s2))
#									 		de sa projection dans la bonne face et la DISTANCE entre ces points
def subdivision(input_mesh, base_mesh,patch, correspondance):
	# initialisation : liste L avec toutes les faces du mesh
	# on commence déjà par créer le modèle sur lequel on va travailler, qui est une copie du base_mesh
	final_mesh = Output()
	inter_mesh = Output() #le inter_mesh représente le final mesh, mais strictement dans les repères relatifs du base_mesh

	# on va aussi initialiser la liste sur laquelle on va travailler qui va avoir les indices correspondant aux faces
	# en fait ce sera un dictionnaire avec en clé l'indice de face du final_mesh, et en clé l'indice de la face du base_mesh auquel elle appartient
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
		
	
	indice = len(L)
	# on calcul aussi la longueur de la diagonale de la bounding_box du input_mesh
	# TODO
	bound_box = 1
	Seuil = 1

	# une fois l'initialisation faite nous pouvons procéder à la subdivision à proprement parler
	# on va boucler sur les étapes suivantes tant qu'il reste une face à traiter
	while len(L) > 0:
		
		E = dict()
		for index_face_final in L.keys():	# pour chaque face (du final_mesh) restante 
			
			# on calcul l'erreur E(f)
			# TODO
			E[index_face_final] = erreur()/bound_box
			
			# si cette erreur est inférieure à un seuil, on considère la face suffisamment proche du input_mesh
			if E[index_face_final] < Seuil :	
				del L[index_face_final]
				continue
			
			# TODO
			# on travail avec les coordonnées 2D relatives
			# on divise chaque face f en 4 triangles 
			# rappel; base(s3:s1,s2)
			[s1,s2,s3] = inter_mesh.face_mapping[index_face_final]
			a = np.sum(0.5*s3,0.5*s1)	#barycentre s3 s1
			b = np.sum(0.5*s3,0.5*s2)	#barycentre s3 s2
			c = np.sum(0.5*s1,0.5*s2)	# barycentre s1 s2

			#pour chacune de ces nouveaux sommets on cherche à quel face de l'input_mesh ils appartiennent
			coord_barycentre_a = recherche_face(a,patch[L[index_face_final]],correspondance)
			coord_barycentre_b = recherche_face(b,patch,correspondance,L[index_face_final])
			coord_barycentre_c = recherche_face(c,patch,correspondance,L[index_face_final])
			 

		# TODO

	return final_mesh

def main(args=None):
	# 1 - On récupère l'objet dans son maillage irrégulier 
	if args is None:
		path = DATA_DIR + "/example/suzanne.obj"
	else :
		path = DATA_DIR + "/" + args
	# partie faite par Yang
	# model de type obja.Model
	input_mesh = parse_file(path)	# input_mesh : model


	# 2 - On crée le maillage de base 
	# pour l'instant cette étape n'est pas gérée, on doit le faire à la main
	# et on utilisera un code fourni
	base_mesh = simplify_mesh(input_mesh) #base_mesh : Output

	# 3 - On partitionne le maillage d'origine
	# sur le maillage de base (patchs)
	# partie faite par Jade
	patch = patcher(input_mesh,base_mesh) # patch : dict(face,list(faces))

	# 4 - On projète les patchs sur le maillage de base
	correspondance = projection(input_mesh, base_mesh, patch) # correspondance: dict(id vertex 3D, vertex2D)
	
	# 5 - On travaille la subdivision
	final_mesh = subdivision(input_mesh, base_mesh,patch, correspondance) # final_mesh : Output

	# on retourne le maillage semi-régulier

	print(args)
	return final_mesh

if __name__ == '__main__':
    main()
