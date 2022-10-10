import obja
import numpy as np
import os
from os import getcwd

DATA_DIR = getcwd()

def patcher(input_mesh,base_mesh):
	return patch

def projection(input_mesh, base_mesh, patch):
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

	return correspondance

def subdivision(input_mesh, base_mesh,patch, correspondance):
	# initialisation : liste L avec toutes les faces du mesh
	# on commence déjà par créer le modèle sur lequel on va travailler, qui est une copie du base_mesh
	final_mesh = Output()
	
	# on va aussi initialiser la liste sur laquelle on va travailler qui va avoir les indices correspondant aux faces
	L = dict();

	# on va tout copier
	for i in len(base_mesh.faces): # pour chaque face du base mesh
		# face : Face
		face = base_mesh.faces[i]

		# on ajoute les sommets au final_mesh
		# rappel : une face est l'association de 3 indices des sommets le formant
		[ida,idb,idc] = [face.a,face.b,face.c]

		# on ajoute dans final_mesh les sommets (attention il faut les copier)
		final_mesh.add_vertex(ida,base_mesh.vertices[ida])
		final_mesh.add_vertex(idb,base_mesh.vertices[idb])
		final_mesh.add_vertex(idc,base_mesh.vertices[idc])

		#maintenant on peut ajouter la face dans final_mesh
		final_mesh.add_face(i,face.clone())	# vu que normalement c'est les mêmes indices
		
		# on peut aussi ajouter les indices des faces directement dans la liste L
		L.append(i)
		
	
	indice = len(L)
	# on calcul aussi la longueur de la diagonale de la bounding_box du input_mesh
	# TODO
	bound_box = 1
	Seuil = 1

	# une fois l'initialisation faite nous pouvons procéder à la subdivision à proprement parler
	# on va boucler sur les étapes suivantes tant qu'il reste une face à traiter
	while len(L) > 0:
		# pour chaque face restante
		E = dict()
		for index_face in final_mesh.face_mapping.keys():
			# on calcul l'erreur E(f)
			# TODO
			E[index_face] = erreur()/bound_box
			if E[index_face] < Seuil :
				del L[index_face]
				continue
			
			# TODO
			# on divise chaque face f en 4 triangles
			[a,b,c] = barycentres()
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
