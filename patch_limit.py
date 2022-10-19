import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import time
from obja import *

EPSILON = 0.00001
START = 0
END = 10

def read_obj(file_path):
  """To read an .obj file
      Only with two different type of faces
        - with 3 vertices
        - with 4 vertices with /
      return tensor of 
        - vertices n_v x 8
          index, x, y, z, r, g, b, alpha
        - faces n_f x 4
          index, v1, v2, v3
      and a list of strings (vt, vn, etc)
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
  """Write an .obj file from faces, vertices, and others
      faces_lists : python list of tensors of faces [index, v1, v2, v3]
                      (v in 1:max)
                    written in the order given
      vertices_lists : python list of tensors of faces [index, x, y, z, r, g, b, alpha]
                    written in the order given
      leftovers : other data to write between vertices and points (list of strings)
      file_path : path of the file to write
          """

  with open(file_path, "w+") as f:
      
    [f.write(f"v {float(_[1])} {float(_[2])} {float(_[3])} {float(_[4])} {float(_[5])} {float(_[6])} 0.7\n") for _ in vertices]
      
    [f.write(f"f {int(_[1])} {int(_[2])} {int(_[3])}\n") for _ in faces]
    
  return None


def main(input_path="./bunny.obj", base_path="./bunny40.obj", output_path="./test_bunny.obj", start=START, end=END):
  faces, vertices = read_obj(input_path)
  base_faces, base_vertices = read_obj(base_path)
  reel_indexes, base_faces, base_vertices = get_reel_indexes_base_mesh()
  start = reel_indexes[0, 1] -1
  end = reel_indexes[0, 2] -1

  #short_path = get_limit(vertices, faces, start, end)
  short_path, short_path_2, new_vertices_coor = get_limit_2(vertices, faces, start, end)

  all_vertices = np.zeros((new_vertices_coor.shape[0] + vertices.shape[0], 7))

  all_vertices[:vertices.shape[0], :] = vertices
  all_vertices[vertices.shape[0]:, 1:4] = new_vertices_coor[:, 1:4]
  #all_vertices[vertices.shape[0]:, 5] = np.ones((new_vertices_coor.shape[0],))

  write_results(short_path_2, all_vertices, faces, output_path)
  

  return faces, vertices, short_path, short_path_2

def main_2(input_path="./bunny_input.obj", base_path="./bunny_base.obj", output_path="./test_bunny.obj", start=START, end=END):
  faces, vertices = read_obj(input_path)
  base_faces, base_vertices = read_obj(base_path)
  
  all_short_path = []
  all_short_path_2 = []

  reel_indexes = np.zeros(base_faces.shape, dtype = int)

  #get the indexes of the faces of the base mesh
  for f in range(base_faces.shape[0]):

    v1 = base_vertices[base_faces[f,1], 1:4]
    v2 = base_vertices[base_faces[f,2], 1:4]
    v3 = base_vertices[base_faces[f,3], 1:4]
    v1_b, v2_b, v3_b = 0, 0, 0
    min_1, min_2, min_3 = 10, 10, 10

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

    reel_indexes[f, :] = [f, v1_b, v2_b, v3_b]

  start = base_faces[0, 1] -1
  end = base_faces[0, 2] -1
    
  for k in range(base_faces.shape[0]):
    print("face numero")
    print(k)

    start1 = reel_indexes[k, 0] -1
    end1 = reel_indexes[k, 1] -1

    start2 = reel_indexes[k, 1] -1
    end2 = reel_indexes[k, 2] -1

    start3 = reel_indexes[k, 2] -1
    end3 = reel_indexes[k, 0] -1
    

    #short_path = get_limit(vertices, faces, start, end)
    short_path, short_path_2, new_vertices_coor = get_limit_2(vertices, faces, start1, end1)
    all_short_path.append(short_path)
    all_short_path_2.append(short_path_2)

    all_vertices = np.zeros((new_vertices_coor.shape[0] + vertices.shape[0], 7))

    all_vertices[:vertices.shape[0], :] = vertices
    if k == 0:
      all_vertices[vertices.shape[0]:, 1:4] = new_vertices_coor[:, 1:4]
      #all_vertices[vertices.shape[0]:, 5] = np.ones((new_vertices_coor.shape[0],))


    #short_path = get_limit(vertices, faces, start, end)
    short_path, short_path_2, new_vertices_coor = get_limit_2(vertices, faces, start2, end2)
    all_short_path.append(short_path)
    all_short_path_2.append(short_path_2)

    all_vertices = np.zeros((new_vertices_coor.shape[0] + vertices.shape[0], 7))

    all_vertices[:vertices.shape[0], :] = vertices
    if k == 0:
      all_vertices[vertices.shape[0]:, 1:4] = new_vertices_coor[:, 1:4]
      #all_vertices[vertices.shape[0]:, 5] = np.ones((new_vertices_coor.shape[0],))


    #short_path = get_limit(vertices, faces, start, end)
    short_path, short_path_2, new_vertices_coor = get_limit_2(vertices, faces, start3, end3)
    all_short_path.append(short_path)
    all_short_path_2.append(short_path_2)

    all_vertices = np.zeros((new_vertices_coor.shape[0] + vertices.shape[0], 7))

    all_vertices[:vertices.shape[0], :] = vertices
    if k == 0:
      all_vertices[vertices.shape[0]:, 1:4] = new_vertices_coor[:, 1:4]
      #all_vertices[vertices.shape[0]:, 5] = np.ones((new_vertices_coor.shape[0],))

  write_results(short_path_2, all_vertices, faces, output_path)
  

  return faces, vertices, all_short_path, all_short_path_2

def main_3(base_path="./bunny_base.obj", input_path="./bunny_input.obj", output_path="./test_bunny.obj"):

  faces, vertices = read_obj(input_path)
  reel_indexes, base_faces, base_vertices = get_reel_indexes_base_mesh(input_path="./bunny_input.obj", base_path="./bunny_base.obj")
  
  all_short_path = []
  all_short_path_2 = []

 

    
  for k in range(base_faces.shape[0]):#range(0,1):#

    print("face numero")
    print(k)

    start1 = reel_indexes[k, 0] -1
    end1 = reel_indexes[k, 1] -1

    start2 = reel_indexes[k, 1] -1
    end2 = reel_indexes[k, 2] -1

    start3 = reel_indexes[k, 2] -1
    end3 = reel_indexes[k, 0] -1
    

    print("limit1")
    #short_path = get_limit(vertices, faces, start, end)
    short_path, short_path_2, new_vertices_coor = get_limit_2(vertices, faces, start1, end1)
    all_short_path.append(short_path)
    all_short_path_2.append(short_path_2)

    
    if k == 0:
      all_vertices = np.zeros((new_vertices_coor.shape[0] + vertices.shape[0], 7))

      all_vertices[:vertices.shape[0], :] = vertices
      all_vertices[vertices.shape[0]:, 1:4] = new_vertices_coor[:, 1:4]
      #all_vertices[vertices.shape[0]:, 5] = np.ones((new_vertices_coor.shape[0],))

    for v in short_path:
      all_vertices[v, 4] = 1
      all_vertices[v, 5] = 1
      all_vertices[v, 6] = 1
      

    print("limit2")
    #short_path = get_limit(vertices, faces, start, end)
    short_path, short_path_2, new_vertices_coor = get_limit_2(vertices, faces, start2, end2)
    all_short_path.append(short_path)
    all_short_path_2.append(short_path_2)

    for v in short_path:
      all_vertices[v, 4] = 1
      all_vertices[v, 5] = 1
      all_vertices[v, 6] = 1

    print("limit3")
    #short_path = get_limit(vertices, faces, start, end)
    short_path, short_path_2, new_vertices_coor = get_limit_2(vertices, faces, start3, end3)
    all_short_path.append(short_path)
    all_short_path_2.append(short_path_2)

    for v in short_path:
      all_vertices[v, 4] = 1
      all_vertices[v, 5] = 1
      all_vertices[v, 6] = 1

    
  write_results([], all_vertices, faces, output_path)
  

  return faces, vertices, all_short_path, all_short_path_2

def iteration_2(short_path, vertices, np_graph, nb_originals):
  #utiliser les edges du graph ?

  new_graph = short_path

  for k in range(1, length(new_graph) -1):
    v1 = new_graph[k]
    v2 = new_graph[k +1]

    #if v1, v2 originals
    if v1 < original and v2 < original:

      # find the face
      # find the edge
      # find the other vertices on the edge and add them on the edge
      new_graph = new_graph

    else:

      pass
      # find the face
      # find the two edges
      # find the other vertices on the edge and add them on the edge
    
    

  return None

"""def get_limit(vertices, faces, start, end):

  np_graph = np.zeros((vertices.shape[0], vertices.shape[0]))

  for f in range(0, faces.shape[0]):

    v1 = faces[f, 1] -1
    v2 = faces[f, 2] -1
    v3 = faces[f, 3] -1

    #if abs(np_graph[v1, v2] ) < EPSILON:
    distance = np.linalg.norm( vertices[v1, 1:] - vertices[v2, 1:] )
    #print(distance)
    np_graph[v1, v2] = distance
    np_graph[v2, v1] = distance

    #if abs(np_graph[v1, v3] ) < EPSILON
    distance = np.linalg.norm( vertices[v1, 1:] - vertices[v3, 1:] )
    np_graph[v1, v3] = distance
    np_graph[v3, v1] = distance

    #if abs(np_graph[v2, v3] ) < EPSILON:
    distance = np.linalg.norm( vertices[v2, 1:] - vertices[v3, 1:] )
    np_graph[v2, v3] = distance
    np_graph[v3, v2] = distance


  graph = csr_matrix(np_graph)
  dist_matrix, predecessors = dijkstra(csgraph=graph, directed=False, return_predecessors=True)


  current = start
  short_path = [start]
  while current != end:

    short_path = short_path + [predecessors[end,current]]
    current = predecessors[end,current]

  return short_path"""

def write_results(short_path, vertices, faces, output_path):

  for k in short_path:
      vertices[k, 6] = 1

      
  write_obj(faces, vertices, output_path)

  return None


def get_limit_2(vertices, faces, start, end):

  check = np.zeros((vertices.shape[0], vertices.shape[0]), dtype=int)
  new_vertices = np.zeros((faces.shape[0], 3))

  nb_new_vertices = 0


  for f in range(0, faces.shape[0]):

    v1 = faces[f, 1] -1
    v2 = faces[f, 2] -1
    v3 = faces[f, 3] -1

    if check[v1, v2] == 0:

      

      nb_new_vertices += 1

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

  np_graph = np.zeros((vertices.shape[0] + nb_new_vertices, vertices.shape[0] + nb_new_vertices))
  new_vertices_coor = np.ones((nb_new_vertices, 7)) 


  
  for f in range(0, faces.shape[0]):
    
    v1 = faces[f, 1] -1
    v2 = faces[f, 2] -1
    v3 = faces[f, 3] -1

    v1p = check[v1, v2] -1 
    v2p = check[v1, v3] -1 
    v3p = check[v2, v3] -1




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

    distance = np.linalg.norm( new_vertices_coor[v2p, 1:4] - vertices[v1, 1:4] )
    np_graph[vertices.shape[0] + v2p, v1] = distance
    np_graph[v1, vertices.shape[0] + v2p] = distance

    distance = np.linalg.norm( new_vertices_coor[v3p, 1:4] - vertices[v2, 1:4] )
    np_graph[vertices.shape[0] + v3p, v2] = distance
    np_graph[v2, vertices.shape[0] + v3p] = distance
    

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
  dist_matrix, predecessors = dijkstra(csgraph=graph, directed=False, return_predecessors=True)


  current = start
  short_path = [start]
  short_path_2 = [start]
  while current != end:

    if predecessors[end,current] < vertices.shape[0]:
      short_path = short_path + [predecessors[end,current]]

    short_path_2 = short_path_2 + [predecessors[end,current]]
    current = predecessors[end,current]

  return short_path, short_path_2, new_vertices_coor

def find_shortest_path_old(dist_matrix, predecessors, graph, start, end, nb_input_vertices):

  


  current = start
  short_path_input = [start]
  short_path_all = [start]
  
  while current != end:

    if predecessors[end,current] < nb_input_vertices:
      short_path_input = short_path_input + [predecessors[end,current]]

    short_path_all = short_path_all + [predecessors[end,current]]
    current = predecessors[end,current]

  return short_path_input, short_path_all

def find_shortest_path(dist_matrix, predecessors, graph, start, end, nb_input_vertices):

  


  current = start
  short_path = [start]
  
  
  while current != end:

    

    short_path = short_path + [predecessors[end,current]]
    current = predecessors[end,current]

  return short_path
  
def get_reel_indexes_base_mesh_old(input_path="./bunny_input.obj", base_path="./bunny_base.obj"):
  base_faces, base_vertices = read_obj(base_path)
  faces, vertices = read_obj(input_path)
  reel_indexes = np.zeros(base_faces.shape, dtype = int)
  #get the indexes of the faces of the base mesh
  for f in range(base_faces.shape[0]):

    v1 = base_vertices[base_faces[f,1] -1, 1:4]
    v2 = base_vertices[base_faces[f,2] -1, 1:4]
    v3 = base_vertices[base_faces[f,3] -1, 1:4]
    v1_b, v2_b, v3_b = 0, 0, 0
    min_1, min_2, min_3 = 10, 10, 10

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

  return reel_indexes, base_faces, base_vertices

def get_reel_indexes_base_mesh(input_path="./bunny_input.obj", base_path="./bunny_base.obj"):
  base_faces, base_vertices = read_obj(base_path)
  faces, vertices = read_obj(input_path)
  reel_indexes = np.zeros(base_vertices.shape[0], dtype = int)
  #get the indexes of the faces of the base mesh
  for i in range(base_vertices.shape[0]):

    c = base_vertices[i, 1:4]
    mini = 10 #/!\
    r = c
    for v in range(vertices.shape[0]):

      if np.linalg.norm(c - vertices[v, 1:4]) < mini:
        r = v
        mini = np.linalg.norm(c - vertices[v, 1:4])

    reel_indexes[i] = r

  return reel_indexes, base_faces, base_vertices

def get_reel_indexes_base_mesh_new(model, base_path="./bunny_base.obj"):
  base_faces, base_vertices = read_obj(base_path)
  #faces, vertices = read_obj(input_path)
  reel_indexes = np.zeros(base_vertices.shape[0], dtype = int)
  #get the indexes of the faces of the base mesh
  for i in range(base_vertices.shape[0]):

    c = base_vertices[i, 1:4]
    mini = 10 #/!\
    r = c
    for v in range(len(model.vertices)):

      if np.linalg.norm(c - model.vertices[v]) < mini:
        r = v
        mini = np.linalg.norm(c - model.vertices[v])

    reel_indexes[i] = r

  return reel_indexes, base_faces, base_vertices
  

def create_augmented_graph(vertices, faces):

    
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
  edges = np.zeros((nb_new_vertices, 3))

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
    edges[v1p, 0] = v1
    edges[v1p, 0] = v2
    edges[v2p, 0] = v2p
    edges[v2p, 0] = v1
    edges[v2p, 0] = v3
    edges[v3p, 0] = v3p
    edges[v3p, 0] = v2
    edges[v3p, 0] = v3




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

    distance = np.linalg.norm( new_vertices_coor[v2p, 1:4] - vertices[v1, 1:4] )
    np_graph[vertices.shape[0] + v2p, v1] = distance
    np_graph[v1, vertices.shape[0] + v2p] = distance

    distance = np.linalg.norm( new_vertices_coor[v3p, 1:4] - vertices[v2, 1:4] )
    np_graph[vertices.shape[0] + v3p, v2] = distance
    np_graph[v2, vertices.shape[0] + v3p] = distance
    

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


def create_graph(model):

    


    

  # adjacency matrix
  np_graph = np.zeros((len(model.vertices), len(model.vertices)))


  
  for f in range(len(model.faces)):
    
    v1 = model.faces[f].a
    v2 = model.faces[f].b
    v3 = model.faces[f].c
    

    distance = np.linalg.norm( model.vertices[v1] - model.vertices[v2] )
    np_graph[v1, v2] = distance
    np_graph[v2, v1] = distance

    distance = np.linalg.norm( model.vertices[v1] - model.vertices[v3] )
    np_graph[v1, v3] = distance
    np_graph[v3, v1] = distance

    distance = np.linalg.norm( model.vertices[v2] - model.vertices[v3] )
    np_graph[v2, v3] = distance
    np_graph[v3, v2] = distance



    


  graph = csr_matrix(np_graph)

  return graph, np_graph

def get_limit_old():

  tic = time.time()

  faces, vertices = read_obj("./bunny_input.obj")
  reel_indexes, base_faces, base_vertices = get_reel_indexes_base_mesh(input_path="./bunny_input.obj", base_path="./bunny_base.obj")

  graph, np_graph, nb_new_vertices, new_vertices_coor, edges = create_augmented_graph(vertices, faces)

  all_vertices = np.zeros((new_vertices_coor.shape[0] + vertices.shape[0], 7))

  all_vertices[:vertices.shape[0], :] = vertices
  all_vertices[vertices.shape[0]:, 1:4] = new_vertices_coor[:, 1:4]

  S_P_I = []
  S_P_A = []

  dist_matrix, predecessors = dijkstra(csgraph=graph, directed=False, return_predecessors=True)
  
  for f in range(base_faces.shape[0]): #range(1):#
    print(f)

    start = base_faces[f, 1] -1
    start = reel_indexes[start]
    end = base_faces[f, 2] -1
    end = reel_indexes[end]

    s_p_input, s_p_all = find_shortest_path(dist_matrix, predecessors, graph, start, end, vertices.shape[0])
    S_P_I.append(s_p_input)
    S_P_A.append(s_p_all)

    start = base_faces[f, 2] -1
    start = reel_indexes[start]
    end = base_faces[f, 3] -1
    end = reel_indexes[end]

    s_p_input, s_p_all = find_shortest_path(dist_matrix, predecessors, graph, start, end, vertices.shape[0])
    S_P_I.append(s_p_input)
    S_P_A.append(s_p_all)

    start = base_faces[f, 3] -1
    start = reel_indexes[start]
    end = base_faces[f, 1] -1
    end = reel_indexes[end]

    s_p_input, s_p_all = find_shortest_path(dist_matrix, predecessors, graph, start, end, vertices.shape[0])
    S_P_I.append(s_p_input)
    S_P_A.append(s_p_all)

  #print(S_P_A)
  for p in S_P_A:
    for i in p:

      all_vertices[ i, 5] = 1
  write_results([], all_vertices, faces, "./bunny_test.obj")

  toc = time.time()
          
  return S_P_I, S_P_A, reel_indexes, toc -tic


def get_limit(input_path):

  #tic = time.time()

  #faces, vertices = read_obj("./bunny_input.obj")
  model = parse_file(input_path)
  #model = division_4(input_path)
  reel_indexes, base_faces, base_vertices = get_reel_indexes_base_mesh_new(model, base_path="./bunny_base.obj")

  graph, np_graph = create_graph(model)

  #all_vertices = np.zeros((new_vertices_coor.shape[0] + vertices.shape[0], 7))

  #all_vertices[:vertices.shape[0], :] = vertices
  #all_vertices[vertices.shape[0]:, 1:4] = new_vertices_coor[:, 1:4]

  S_P = []
  #S_P_A = []

  n_v = len(model.vertices)

  dist_matrix, predecessors = dijkstra(csgraph=graph, directed=False, return_predecessors=True)
  
  for f in range(base_faces.shape[0]): #range(1):#
    print(f)

    start = base_faces[f, 1] -1
    start = reel_indexes[start]
    end = base_faces[f, 2] -1
    end = reel_indexes[end]

    s_p = find_shortest_path(dist_matrix, predecessors, graph, start, end, n_v)
    S_P.append(s_p)
    

    start = base_faces[f, 2] -1
    start = reel_indexes[start]
    end = base_faces[f, 3] -1
    end = reel_indexes[end]

    s_p = find_shortest_path(dist_matrix, predecessors, graph, start, end, n_v)
    S_P.append(s_p)
    

    start = base_faces[f, 3] -1
    start = reel_indexes[start]
    end = base_faces[f, 1] -1
    end = reel_indexes[end]

    s_p = find_shortest_path(dist_matrix, predecessors, graph, start, end, n_v)
    S_P.append(s_p)
    

  #print(S_P_A)
  #for p in S_P:
  #  for i in p:

  #    all_vertices[ i, 5] = 1
  #write_results([], all_vertices, faces, "./bunny_test.obj")

  #toc = time.time()
          
  return S_P, reel_indexes#, toc -tic

# path : string : path of entry mesh
def division_4(path):
    input_mesh = parse_file(path)

    mesh = obja.Output()
    # on va tout copier
    for i in len(input_mesh.faces): # pour chaque face du base mesh
        face = input_mesh.faces[i]    # face : Face

        # on ajoute les sommets au final_mesh
        # rappel : une face est l'association de 3 indices des sommets le formant
        [ida,idb,idc] = [face.a,face.b,face.c]

        # on ajoute dans final_mesh les sommets (attention il faut les copier)
        mesh.add_vertex(ida,input_mesh.vertices[ida])
        mesh.add_vertex(idb,input_mesh.vertices[idb])
        mesh.add_vertex(idc,input_mesh.vertices[idc])

        #maintenant on peut ajouter la face dans final_mesh
        mesh.add_face(i,face)    # vu que normalement c'est les mêmes indices


    indice_f = len(mesh.face_mapping)
    indice_v = len(mesh.vertex_mapping)
    for f in mesh.face_mapping.keys():
        [s1,s2,s3] = mesh.face_mapping[f]
        a = np.sum(0.5*s3,0.5*s1)    #barycentre s3 s1
        b = np.sum(0.5*s3,0.5*s2)    #barycentre s3 s2
        c = np.sum(0.5*s1,0.5*s2)    # barycentre s1 s2

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
        #del mesh.face_mapping(f)
    return mesh



