import csi_code as cc
import obja
import patch_limit as pl

p,r, input_mesh = pl.get_limit("sphere_input.obj")
##base_mesh = obja.parse_file("sphere_base.obj")
patch,colors,faces_restantes = cc.partition(p,input_mesh)
##correspondance = cc.projection(input_mesh,base_mesh,patch)
##
####
##with open('test_bunny.obja','w') as output :
##    for v in input_mesh.vertices : 
##        output.write(f'v {v[0]} {v[1]} {v[2]}\n')
##    for i in range(len(input_mesh.faces)) :
##        face = input_mesh.faces[i]
##        output.write(f'f {face.a + 1} {face.b + 1} {face.c + 1}\n')
##    for k, c in colors.items() :
##        output.write(f'fc {k + 1} {c[0]} {c[1]} {c[2]}\n')
        
##
with open('test_bunny_v3.obj','w') as output :
    for k in range(len(input_mesh.vertices)) :
        v = input_mesh.vertices[k]
        if  k in colors :
            c = colors.get(k)
            output.write(f'v {v[0]} {v[1]} {v[2]} {c[0]} {c[1]} {c[2]}\n')
        else :
            output.write(f'v {v[0]} {v[1]} {v[2]} 1.0 1.0 1.0\n')
    for i in range(len(input_mesh.faces)) :
        face = input_mesh.faces[i]
        output.write(f'f {face.a + 1} {face.b + 1} {face.c + 1}\n')



##with open('bunny_base_main.obj','w') as output :
##    for k in range(len(input_mesh.vertices)) :
##        v = input_mesh.vertices[k] 
##        output.write(f'v {v[0]} {v[1]} {v[2]} 1.0 1.0 1.0\n')
##    for i in range(len(input_mesh.faces)) :
##        face = input_mesh.faces[i]
##        output.write(f'f {face.a + 1} {face.b + 1} {face.c + 1}\n')
