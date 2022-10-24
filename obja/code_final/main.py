import math
from re import A
import obja
import numpy as np
import os
from os import getcwd
import matplotlib.pyplot as plt
import logging as lg
import boundary_curves as bc
import subdivision as sb
import partition as pt
import projection as pp


DATA_DIR = getcwd()

DEFAULT_INPUT = "../sphere4.obj"
DEFAULT_BASE = "../sphere5.obj"
DEFAULT_NEW_INPUT = "../sphere_new_input.obj"


def main(args=None):

        
        # 1 - On récupère l'objet dans son maillage irrégulier 
        if args is None:
                DEFAULT_INPUT = DATA_DIR +"/" + "../sphere4.obj"
                DEFAULT_BASE = DATA_DIR + "/" + "../sphere5.obj"
                DEFAULT_NEW_INPUT = DATA_DIR + "/" + "../sphere_new_input.obj"
        else :
                path = DATA_DIR + "/" + args

        # model de type obja.Model

        bords, DEFAULT_NEW_INPUT, r = bc.main(DEFAULT_INPUT, DEFAULT_BASE, DEFAULT_NEW_INPUT)
        input_mesh = obja.parse_file(DEFAULT_NEW_INPUT)
        
        print("calculs de bords : done")

        # 2 - On crée le maillage de base 
        # pour l'instant cette étape n'est pas gérée, on doit le faire à la main
        # et on utilisera un code fourni
        base_mesh = obja.parse_file(DEFAULT_BASE)


        # 3 - On partitionne le maillage d'origine
        # sur le maillage de base (patchs)
        patch , faces_restantes,r = pt.partition(bords, input_mesh,base_mesh,r,str('h')) # patch : dict(face,list(faces))
        print("partionnage : done")

        # 4 - On projète les patchs sur le maillage de base
        correspondance, distances = pp.projection(input_mesh,base_mesh,patch) # correspondance: dict(id vertex 3D, vertex2D)   
        print("projections : done")

        # 5 - On travaille la subdivision
        final_mesh = sb.subdivision(input_mesh, base_mesh,patch, correspondance,r) # final_mesh : Output
        print("subdivisions : done")

        with open('test_sphere.obj','w') as output :
                for k in final_mesh.vertices.keys() :
                        v = final_mesh.vertices.get(k)
                        output.write(f'v {v[0]} {v[1]} {v[2]}\n')
                for i in final_mesh.faces.keys() :
                        face = final_mesh.faces.get(i)
                        output.write(f'f {face.a + 1} {face.b + 1} {face.c + 1}\n')
        # on retourne le maillage semi-régulier
        #print(args)
        return final_mesh

if __name__ == '__main__':
    main()

    patchs_limits, new_input_mesh_path, reel_indexes = bc.main(input_mesh_path=DEFAULT_INPUT, base_mesh_path=DEFAULT_BASE, new_input_mesh_path=DEFAULT_NEW_INPUT)
