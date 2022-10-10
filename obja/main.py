import obja
import numpy as np
import os
from os import getcwd

DATA_DIR = getcwd()

def patch = patcher(input_mesh,base_mesh)
	return patch

def correspondance = projection(input_mesh, base_mesh, patch)
	return correspondance

def final_mesh = subdivision(input_mesh, base_mesh,patch, correspondance)
	# initialisation : liste L avec toutes les faces du mesh
	# on commence déjà par créer le modèle sur lequel on va travailler, qui est une copie du base_mesh
	final_mesh = Output()
	
	# on va aussi initialiser la liste sur laquelle on va travailler qui va avoir les indices correspondant aux faces
	L = dict();

	# on va tout copier
	for i in len(base_mesh.faces):
		final_mesh.add_face(i,clone(base_mesh.faces[i]))
		L.append(i)
	
	indice = len(L)
	# on calcul aussi la longueur de la diagonale de la bounding_box du input_mesh
	bound_box = 1
	Seuil = 1

	# une fois l'initialisation faite nous pouvons procéder à la subdivision à proprement parler
	# on va boucler sur les étapes suivantes tant qu'il reste une face à traiter
	while len(L) > 0:
		# pour chaque face restante
		E = dict()
		for index_face in final_mesh.face_mapping.keys()
			# on calcul l'erreur E(f)
			E[index_face] = erreur()/bound_box
			if E[index_face] < Seuil :
				del L[index_face]
				continue
			
			# on divise chaque face f en 4 triangles
			[a,b,c] = barycentres()

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
	patch = patcher(input_mesh,base_mesh) # patch : dict(face,list(vertices))

	# 4 - On projète les patchs sur le maillage de base
	correspondance = projection(input_mesh, base_mesh, patch) # correspondance: dict(vertex 3D, vertex2D)
	
	# 5 - On travaille la subdivision
	final_mesh = subdivision(input_mesh, base_mesh,patch, correspondance) # final_mesh : Output

	# on retourne le maillage semi-régulier

	print(args)
	return final_mesh

if __name__ == '__main__':
    main()
