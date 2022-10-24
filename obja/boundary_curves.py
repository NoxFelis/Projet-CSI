import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import time
import obja as obja

EPSILON = 0.00001
MAX = float('inf')

DEFAULT_INPUT = "./sphere_input.obj"
DEFAULT_BASE = "./sphere_base.obj"
DEFAULT_NEW_INPUT = "./sphere_new_input.obj"

DEFAULT_INPUT2 = "./bunny_input.obj"
DEFAULT_BASE2 = "./bunny6.obj"
DEFAULT_NEW_INPUT2 = "./bunny_new_input.obj"


def read_obj(file_path):
  """To read an .obj file
      Only for faces
        - with 3 vertices with /
      return array of 
        - vertices n_v x 7
          index, x, y, z, r, g, b
        - faces n_f x 4
          index, v1, v2, v3
  """
  nb_vertices = 0
  nb_faces = 0

  # reading the file
  with open(file_path, "r") as f:
    lines = f.readlines()

  
  # get the number of vertices and faces to create tensors
  for line in lines:

    split_line = line.split(" ")

    # the line is a vertex
    if split_line[0] == "v" :
      nb_vertices += 1

    # the line is a face
      # face = [index, v1, v2, v3] (index in [1,max]) (v in [1, max])
    elif split_line[0] == "f":

      nb_faces += 1

  
  # putting data in tensors
  vertices = np.ones((nb_vertices, 7))
  faces = np.ones((nb_faces, 4), dtype = int)

  current_vertex = 0
  current_face = 0

  for line in lines:

    split_line = line.split(" ")

    # the line is a vertex
      # vertices = [index, x, y, z, r, g, b, a] (index in [1,max])
    if split_line[0] == "v" :



      vertices[current_vertex] = np.array([current_vertex +1, float(split_line[1]), float(split_line[2]), float(split_line[3]), 0, 0, 0])
      current_vertex += 1
      
    elif split_line[0] == "f":

      v1 = split_line[1].split("/")[0]
      v2 = split_line[2].split("/")[0]
      v3 = split_line[3].split("/")[0]


      faces[current_face] = np.array([current_face +1, int(v1), int(v2), int(v3)], dtype = int)
      current_face += 1
      

  return faces, vertices

def write_obj(faces, vertices, file_path):
  """Write an .obj file from faces, vertices
      faces : array n_f * [index, v1, v2, v3]
                      (v in 1:max)
                    written in the order given
      vertices : array n_v * [index, x, y, z, r, g, b]
                    written in the order given
      file_path : path of the file to write
          """

  with open(file_path, "w+") as f:
      
    [f.write(f"v {float(_[1])} {float(_[2])} {float(_[3])} {float(_[4])} {float(_[5])} {float(_[6])} 0.7\n") for _ in vertices]
      
    [f.write(f"f {int(_[1])} {int(_[2])} {int(_[3])}\n") for _ in faces]
    
  return None




def find_shortest_path(predecessors, graph, start, end):
  """Return the shortest path form the vertex start to end using dijkstra from scipy.
      predecessors = output of dijkstra
      graph = the graph representing the mesh (scipy)
      start = int vertex index
      end = int vertex index
      """

  


  current = start
  short_path = [start]

  while current != end :
    short_path = short_path + [predecessors[end,current]]
    current = predecessors[end,current]
  

  return short_path
  
def get_output_base_mesh(input_path="./bunny_input.obj", base_path="./bunny_base.obj"):
  """Get data from the input mesh and the base mesh, return also the indexes of the vertices in the base mesh from the input mesh
      return reel_indexes, base_faces, base_vertices, faces, vertices
      """

  # base mesh
  base_faces, base_vertices = read_obj(base_path)

  # input mesh
  faces, vertices = read_obj(input_path)

  # indexes of the vertices in the base mesh from the input mesh 
  reel_indexes = np.zeros(base_faces.shape, dtype = int)
  
  # creating a new "base_faces"
  for f in range(base_faces.shape[0]):

    v1 = base_vertices[base_faces[f,1] -1, 1:4]
    v2 = base_vertices[base_faces[f,2] -1, 1:4]
    v3 = base_vertices[base_faces[f,3] -1, 1:4]

    
    v1_b, v2_b, v3_b = 0, 0, 0
    min_1, min_2, min_3 = MAX, MAX, MAX

    
    # find the vertex in the input mesh the closest to a vertex from the base mesh
    for v in range(vertices.shape[0]):

      if np.linalg.norm(v1 - vertices[v, 1:4]) < min_1:
        v1_b = v
        min_1 = np.linalg.norm(v1 - vertices[v, 1:4])
      elif np.linalg.norm(v2 - vertices[v, 1:4]) < min_2:
        v2_b = v
        min_2 = np.linalg.norm(v2 - vertices[v, 1:4])
      elif np.linalg.norm(v3 - vertices[v, 1:4]) < min_3:
        v3_b = v
        min_3 = np.linalg.norm(v3 - vertices[v, 1:4])

    reel_indexes[f, :] = [f, v1_b +1, v2_b +1, v3_b +1]

  return reel_indexes, base_faces, base_vertices, faces, vertices


def create_augmented_graph(vertices, faces):
  """Create a scipy graph from the mesh (vertices, faces)
      return graph, np_graph, nb_new_vertices, new_vertices_coor, edges"""

    
    # to create each new vertex only once
  check = np.zeros((vertices.shape[0], vertices.shape[0]), dtype=int)

    # indexes of the new vertices per face
  new_vertices = np.zeros((faces.shape[0], 3))

    

    # number of vertices added
  nb_new_vertices = 0


    
    # indexing of the new vertices
  for f in range(0, faces.shape[0]):
        
        # vertices of the face
    v1 = faces[f, 1] -1
    v2 = faces[f, 2] -1
    v3 = faces[f, 3] -1

        # if the vertex has not been added yet
    if check[v1, v2] == 0:
          
          # new vertex added
      nb_new_vertices += 1
          
          #
      check[v1, v2] = nb_new_vertices
      check[v2, v1] = nb_new_vertices

          
      new_vertices[f, 0] = nb_new_vertices

    else:

      new_vertices[f, 0] = check[v1, v2]

    if check[v1, v3] == 0:

          
      nb_new_vertices += 1

      check[v1, v3] = nb_new_vertices
      check[v3, v1] = nb_new_vertices

      new_vertices[f, 1] = nb_new_vertices

    else:

      new_vertices[f, 1] = check[v1, v3]


    if check[v2, v3] == 0:

          
      nb_new_vertices += 1

      check[v2, v3] = nb_new_vertices
      check[v3, v2] = nb_new_vertices

      new_vertices[f, 2] = nb_new_vertices

    else:

      new_vertices[f, 2] = check[v2, v3]

  # adjacency matrix
  np_graph = np.zeros((vertices.shape[0] + nb_new_vertices, vertices.shape[0] + nb_new_vertices))
  # indexes of the new vertices per face
  edges = np.zeros((nb_new_vertices, 3), int)

  # coordinates of the new vertices
  new_vertices_coor = np.ones((nb_new_vertices, 7)) 


  
  for f in range(0, faces.shape[0]):
    
    v1 = faces[f, 1] -1
    v2 = faces[f, 2] -1
    v3 = faces[f, 3] -1

    v1p = check[v1, v2] -1 
    v2p = check[v1, v3] -1 
    v3p = check[v2, v3] -1

    edges[v1p, 0] = v1p
    edges[v1p, 1] = v1
    edges[v1p, 2] = v2
    edges[v2p, 0] = v2p
    edges[v2p, 1] = v1
    edges[v2p, 2] = v3
    edges[v3p, 0] = v3p
    edges[v3p, 1] = v2
    edges[v3p, 2] = v3




    new_vertices_coor[v1p, 1:4] = 0.5 * ( vertices[v1, 1:4] + vertices[v2, 1:4] )
    new_vertices_coor[v2p, 1:4] = 0.5 * ( vertices[v1, 1:4] + vertices[v3, 1:4] )
    new_vertices_coor[v3p, 1:4] = 0.5 * ( vertices[v2, 1:4] + vertices[v3, 1:4] )
    

    distance = np.linalg.norm( vertices[v1, 1:4] - vertices[v2, 1:4] )
    np_graph[v1, v2] = distance
    np_graph[v2, v1] = distance

    distance = np.linalg.norm( vertices[v1, 1:4] - vertices[v3, 1:4] )
    np_graph[v1, v3] = distance
    np_graph[v3, v1] = distance

    distance = np.linalg.norm( vertices[v2, 1:4] - vertices[v3, 1:4] )
    np_graph[v2, v3] = distance
    np_graph[v3, v2] = distance
    

    distance = np.linalg.norm( new_vertices_coor[v1p, 1:4] - vertices[v3, 1:4] )
    np_graph[vertices.shape[0] + v1p, v3] = distance
    np_graph[v3, vertices.shape[0] + v1p] = distance

    distance = np.linalg.norm( new_vertices_coor[v2p, 1:4] - vertices[v2, 1:4] )
    np_graph[vertices.shape[0] + v2p, v2] = distance
    np_graph[v2, vertices.shape[0] + v2p] = distance

    distance = np.linalg.norm( new_vertices_coor[v3p, 1:4] - vertices[v1, 1:4] )
    np_graph[vertices.shape[0] + v3p, v1] = distance
    np_graph[v1, vertices.shape[0] + v3p] = distance
    

    distance = np.linalg.norm( new_vertices_coor[v1p, 1:4] - new_vertices_coor[v2p, 1:4] )
    np_graph[vertices.shape[0] + v1p, vertices.shape[0] + v2p] = distance
    np_graph[vertices.shape[0] + v2p, vertices.shape[0] + v1p] = distance

    distance = np.linalg.norm( new_vertices_coor[v1p, 1:4] - new_vertices_coor[v3p, 1:4] )
    np_graph[vertices.shape[0] + v1p, vertices.shape[0] + v3p] = distance
    np_graph[vertices.shape[0] + v3p, vertices.shape[0] + v1p] = distance

    distance = np.linalg.norm( new_vertices_coor[v2p, 1:4] - new_vertices_coor[v3p, 1:4] )
    np_graph[vertices.shape[0] + v2p, vertices.shape[0] + v3p] = distance
    np_graph[vertices.shape[0] + v3p, vertices.shape[0] + v2p] = distance



    


  graph = csr_matrix(np_graph)

  return graph, np_graph, nb_new_vertices, new_vertices_coor, edges



def get_base_edges(reel_indexes):
  """To get the edges of the base mesh
      return
        edges_f : reel_indexes.shape, indexes of the edges
        edges_dic : (start, end) : edge indexe
        nb_edges
  """

  edges_f = np.zeros((reel_indexes.shape[0], 3))
  edges_dic = {}
  nb_edges = 0

  # for each face of the base mesh
  for f in range(reel_indexes.shape[0]):

    
    # get the index of the vertex in the input mesh
    v1 = reel_indexes[f, 1]
    v2 = reel_indexes[f, 2]
    v3 = reel_indexes[f, 3]

    # first edge (v1, v2)
    a = min(v1, v2)
    b = max(v1, v2)
      # edge does not exist yet
    if not (a, b) in edges_dic  : 

      # add edge to the dictionary
      edges_dic[(a, b)] = nb_edges

      # keep in memory the index of the edge according to the face
      edges_f[f, 0] = nb_edges
      
      # incr the number of edges
      nb_edges += 1

      #edge already exist
    else :
      # keep in memory the index of the edge according to the face
      edges_f[f, 0] = edges_dic[(a, b)]
      

    # 2nd edge (v2, v3)
    a = min(v3, v2)
    b = max(v3, v2)
      # edge does not exist yet
    if not (a, b) in edges_dic :



      # add edge to the dictionary
      edges_dic[(a, b)] = nb_edges

      # keep in memory the index of the edge according to the face
      edges_f[f, 1] = nb_edges
      
      # incr the number of edges
      nb_edges += 1

      #edge already exist
    else :
      # keep in memory the index of the edge according to the face
      edges_f[f, 1] = edges_dic[(a, b)]

    # 3rd edge (v3, v1)
    a = min(v1, v3)
    b = max(v1, v3)
        # edge does not exist yet
    if not (a, b) in edges_dic:

      

      # add edge to the dictionary
      edges_dic[(a, b)] = nb_edges

      # keep in memory the index of the edge according to the face
      edges_f[f, 2] = nb_edges
      
      # incr the number of edges
      nb_edges += 1

      #edge already exist
    else :
      
      # keep in memory the index of the edge according to the face
      edges_f[f, 2] = edges_dic[(a, b)]

  return edges_f, edges_dic, nb_edges

##def get_limit_old(input_path="./sphere_input.obj", base_path="./sphere_base.obj"):
##  """"Get the shortest path in the input mesh for each edge of the base mesh"""
##
##
##  #faces, vertices = read_obj("./sphere_input.obj")
##  reel_indexes, base_faces, base_vertices, faces, vertices = get_output_base_mesh(input_path, base_path)
##  edges_f, edges_dic, nb_edges = get_base_edges(reel_indexes)
##  #print(reel_indexes)
##
##  graph, np_graph, nb_new_vertices, new_vertices_coor, edges = create_augmented_graph(vertices, faces)
##
##  all_vertices = np.zeros((new_vertices_coor.shape[0] + vertices.shape[0], 7))
##
##  all_vertices[:vertices.shape[0], :] = vertices
##  all_vertices[vertices.shape[0]:, 1:4] = new_vertices_coor[:, 1:4]
##
##  S_P = []
##
##  dist_matrix, predecessors = dijkstra(csgraph=graph, directed=False, return_predecessors=True)
##
##
##  for e in edges_dic:
##    #print(e)
##
##    start = e[0] -1
##    end = e[1] -1
##
##    s_p = find_shortest_path(predecessors, graph, start, end)
##    S_P.append(s_p)
##    
##
####  for p in S_P:
####    for i in p:
####
####      all_vertices[ i, 5] = 1
####  write_results([], all_vertices, faces, "./sphere_test_e.obj")
##          
##  return np_graph, S_P, reel_indexes, edges, base_faces, vertices.shape[0], faces, vertices, all_vertices, edges_dic, edges_f

def get_all_limits(reel_indexes, edges_f, S_P):
  """"Get the shortest path in the input mesh for each face of the base mesh
        edges_f : array of edge indexes nb_f * 3
        S_P : list of lists of shortest path for each edge
      return list of lists
          [ [path 1 face 1] [path 2 face 1] [path 3 face 1] ... [path 3 face f]"""

  S_P_F = []



  for f in range(edges_f.shape[0]):

    v1_r = reel_indexes[f, 1]
    v2_r = reel_indexes[f, 2]
    v3_r = reel_indexes[f, 3]

    
    p1 = S_P[int(edges_f[f, 0])].copy()

    if v1_r -1 != p1[0]:
      
      p1.reverse()
   

      
    S_P_F.append(p1)
    
    ##print(v1_r, p1[0], p1[-1])

    p2 = S_P[int(edges_f[f, 1])].copy()
 
    if v2_r -1 != p2[0]:
      p2.reverse()

    S_P_F.append(p2)
    #print(v2_r, p2[0], p2[-1])


    p3 = S_P[int(edges_f[f, 2])].copy()

    if v3_r -1 != p3[0]:
      p3.reverse()

   
    S_P_F.append(p3)
    #print(v3_r, p3[0], p3[-1])
      
    
  return S_P_F

    

def get_new_mesh(S_P, edges, nb_vertices, faces, vertices):
  """Create and download the new mesh with the new patch limits
        S_P : shortest paths by edges, list of list
        nb_vertices : nb of input vertices
        edges : correspondance between new and original points
      retunr
        all_faces, all_vertices, S_P"""

  # creation of faces
  new_faces = []

  # adding vertices
  nb_new_vertices = 0

  # deleting faces
  to_delete = []

  # correspondance between the indexes of the added vertices
  correspondance = {}

  # for each patch limit
  for f in range(len(S_P)):

    s_p = S_P[f]

    # going through each vertex to know which one must be added
    for v in range(1, len(s_p)):

      v1 = s_p[v]
      v2 = s_p[v -1]

      # un point ajouté fait parti du plus court chemin
      # il faut ajouter deux ou trois nouvelles faces

      # we had a new vertex
      if v1 >= nb_vertices:
        
        # si le point suivant est aussi un point ajouté
        # we had two new vertices
        if v2 >= nb_vertices:
          
          
          # find the face to split
          a = edges[v1 - nb_vertices, 1]
          b = edges[v1 - nb_vertices, 2]
          c = edges[v2 -  nb_vertices, 1]
          d = edges[v2 - nb_vertices, 2]
          
            # common point between the two edges
          if c == a or c == b:
            common = c
            c = d
            
          else:
            common = d
          
          i1 = np.argwhere(faces[:, 1:] == a+1)
          f_i1 = faces[ i1[:, 0] ]
          
          i2 = np.argwhere(f_i1[:, 1:] == b+1)
          f_i2 = f_i1[ i2[:, 0] ]

          i3 = np.argwhere(f_i2[:, 1:] == c+1)
          f_i3 = f_i2[ i3[:, 0] ]

          # we found the index of the face we want to split
          to_split_face = f_i3[0, 0] -1
          to_delete.append(to_split_face)


            # to keep the right order
          if faces[to_split_face, 1] == common +1:

            A = faces[to_split_face, 2]
            C = faces[to_split_face, 1]
            E = faces[to_split_face, 3]


          elif faces[to_split_face, 2] == common +1:

            A = faces[to_split_face, 3]
            C = faces[to_split_face, 2]
            E = faces[to_split_face, 1]
            

          else:

            A = faces[to_split_face, 2]
            C = faces[to_split_face, 3]
            E = faces[to_split_face, 1]
            

            # to keep the right order
          if E != a+1  and E != b +1:
            B = v1 +1
            D = v2 +1

          else:
            B = v2 +1
            D = v1 +1
          

          #new faces = ABE, BCD, BDE
          new_faces.append(np.array([A, B, E]))
          new_faces.append(np.array([B, C, D]))
          new_faces.append(np.array([B, D, E]))

          # when only a part of the vertices is going to be added          
          if not v1 in correspondance:
            correspondance[v1] = nb_new_vertices
            nb_new_vertices += 1

          if not v2 in correspondance:
            correspondance[v2] = nb_new_vertices
            nb_new_vertices += 1



        # one new vertex and one original
        else:

          # find the face

          a = edges[v1 - nb_vertices, 1]
          b = edges[v1 - nb_vertices, 2]
          c = v2

          i1 = np.argwhere(faces[:, 1:] == a+1)
          f_i1 = faces[ i1[:, 0] ]
          
          i2 = np.argwhere(f_i1[:, 1:] == b+1)
          f_i2 = f_i1[ i2[:, 0] ]

          i3 = np.argwhere(f_i2[:, 1:] == c+1)
          f_i3 = f_i2[ i3[:, 0] ]

          # we found the index of the face we want to split
          to_split_face = f_i3[0, 0] -1
          to_delete.append(to_split_face)

            # to keep the right order
          if faces[to_split_face, 1] == c +1:
            A = faces[to_split_face, 2]
            B = faces[to_split_face, 3]
            C = faces[to_split_face, 1]

          elif faces[to_split_face, 2] == c +1:

            A = faces[to_split_face, 3]
            B = faces[to_split_face, 1]
            C = faces[to_split_face, 2]

          else:

            A = faces[to_split_face, 1]
            B = faces[to_split_face, 2]
            C = faces[to_split_face, 3]

          D = v1 +1
          
          #new faces = ADC, BCD
          new_faces.append(np.array([A, D, C]))
          new_faces.append(np.array([B, C, D]))

          # when only a part of the vertices is going to be added   
          if not v1 in correspondance:
            correspondance[v1] = nb_new_vertices
            nb_new_vertices += 1

      # the other vertex is a new one
      elif v2 >= nb_vertices:

        # find the face
        a = edges[v2 - nb_vertices, 1]
        b = edges[v2 - nb_vertices, 2]
        c = v1
        

        i1 = np.argwhere(faces[:, 1:] == a+1)
        f_i1 = faces[ i1[:, 0] ]
        
        i2 = np.argwhere(f_i1[:, 1:] == b+1)
        f_i2 = f_i1[ i2[:, 0] ]

        i3 = np.argwhere(f_i2[:, 1:] == c+1)
        f_i3 = f_i2[ i3[:, 0] ]

        # we found the index of the face we want to split
        to_split_face = f_i3[0, 0]  -1
        to_delete.append(to_split_face)

        
          # to keep the right order
        if faces[to_split_face, 1] == c+1:

          A = faces[to_split_face, 2]
          B = faces[to_split_face, 3]
          C = faces[to_split_face, 1]

        elif faces[to_split_face, 2] == c+1:

          A = faces[to_split_face, 3]
          B = faces[to_split_face, 1]
          C = faces[to_split_face, 2]


        else:

          A = faces[to_split_face, 1]
          B = faces[to_split_face, 2]
          C = faces[to_split_face, 3]


        D = v2 +1
        
        #new faces = ADC, BCD
        new_faces.append(np.array([A, D, C]))
        new_faces.append(np.array([B, C, D]))
        
        # when only a part of the vertices is going to be added  
        if not v2 in correspondance:
          correspondance[v2] = nb_new_vertices
          nb_new_vertices += 1
        


        

  # create the new set of vertices and faces
  all_vertices = np.zeros((nb_vertices + nb_new_vertices, 7))
  all_vertices[0:nb_vertices, :] = vertices[0:nb_vertices, :]
  all_faces = np.zeros((faces.shape[0] + len(new_faces) - len(to_delete), 4), int)

  # to delete some faces
  current_face = 0
  for f in range(faces.shape[0]):

    if not f in to_delete:
      all_faces[current_face, :] = faces[f, :]
      current_face +=1

  # to add the new faces
  for f in range(len(new_faces)):
    v1 = new_faces[f][0]
    v2 = new_faces[f][1]
    v3 = new_faces[f][2]

    if v1-1 >= nb_vertices:
      v1 = correspondance[v1-1] + nb_vertices +1
    if v2-1 >= nb_vertices:
      v2 = correspondance[v2-1] + nb_vertices +1
    if v3-1 >= nb_vertices:
      v3 = correspondance[v3-1] + nb_vertices +1
      
    all_faces[faces.shape[0] - len(to_delete) + f, 1:] = [v1, v2, v3]

  for v in correspondance:
    all_vertices[correspondance[v] + nb_vertices, 1:4] = vertices[v, 1:4]


  # update the shortest paths with the new indexes
  for p in range(len(S_P)):
  
    s_p = S_P[p]
    for i in range(len(s_p)):

      if s_p[i] >= nb_vertices:
        
        all_vertices[ correspondance[s_p[i]] + nb_vertices, 5] = 1
        s_p[i] = correspondance[s_p[i]] + nb_vertices
        
      else:
        all_vertices[ s_p[i], 5] = 1

    S_P[p] = s_p
    

  
  return all_faces, all_vertices
    
        
    


def main(input_mesh_path=DEFAULT_INPUT, base_mesh_path=DEFAULT_BASE, new_input_mesh_path=DEFAULT_NEW_INPUT):
  """"Get the shortest path in the input mesh for each edge of the base mesh"""

  # get the data of the input and the base mesh
  reel_indexes, base_faces, base_vertices, faces, vertices = get_output_base_mesh(input_mesh_path, base_mesh_path)

  # get set of edges of the base mesh
  edges_f, edges_dic, nb_edges = get_base_edges(reel_indexes)

  # create a graph
  graph, np_graph, nb_new_vertices, new_vertices_coor, edges = create_augmented_graph(vertices, faces)

  # input vertices + new vertices
  all_vertices = np.zeros((new_vertices_coor.shape[0] + vertices.shape[0], 7))
  all_vertices[:vertices.shape[0], :] = vertices
  all_vertices[vertices.shape[0]:, 1:4] = new_vertices_coor[:, 1:4]

  # dijkstra
  dist_matrix, predecessors = dijkstra(csgraph=graph, directed=False, return_predecessors=True)

  # find the patch limits per edges of the base mesh
  S_P = []

  for e in edges_dic:
    #print(e)

    start = e[0] -1
    end = e[1] -1

    s_p = find_shortest_path(predecessors, graph, start, end)
    S_P.append(s_p)

  
  # create a new mesh with the path limits    
  all_new_faces, all_new_vertices = get_new_mesh(S_P, edges, vertices.shape[0], faces, all_vertices)
  write_obj(all_new_faces, all_new_vertices, new_input_mesh_path)
  
  # get the patch limits per faces of the base mesh
  S_P_all = get_all_limits(reel_indexes, edges_f, S_P)

  return S_P_all, new_input_mesh_path, reel_indexes

  
  



