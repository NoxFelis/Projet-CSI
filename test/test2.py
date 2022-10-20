from obja import *
from numpy import *
import numpy as np

np.set_printoptions(threshold=np.inf)
model = parse_file("./example/suzanne.obj")
vertices = model.vertices
faces = model.faces

num_vertrices = np.size(vertices)
num_face = np.size(faces)

# la distance entre 2 points
def distance2point(a,b):
    squares = np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)
    return squares
# creer un graph de distance des points
network = np.zeros((num_vertrices,num_vertrices))
print(np.shape(network))
for i in range(num_face):

    network[faces[i].a][faces[i].b] = distance2point(vertices[(faces[i].a)],vertices[(faces[i].b)])
    network[faces[i].a][faces[i].c] = distance2point(vertices[(faces[i].a)],vertices[(faces[i].c)])
    network[faces[i].b][faces[i].c] = distance2point(vertices[(faces[i].b)],vertices[(faces[i].c)])
print (network)

#utilise algorithme de floyed puor calculer la plus court chemin
def Floyd(network):
    FMAX = 999
    n = len(network)
    d = [[network[i][j] if network[i][j] !=0 \
    else FMAX for i in range(n)]for j in range(n)]
    for k in range(n):# Nœud Broker
        for i in range(n):# Le nœud de départ
            for j in range(n):# Nœud de fin
                # Le nœud de départ→Nœud de fin
                # Le nœud de départ→Nœud Broker→Nœud de fin
                # valeur minimum
                d[i][j] = min(d[i][j], d[i][k]+d[k][j])
    return d

res = Floyd(network)
for item in res:
    print(item)