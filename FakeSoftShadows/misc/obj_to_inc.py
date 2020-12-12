import sys

if len(sys.argv) < 5:
    raise Exception('Missing arguments')

teapot_obj = sys.argv[1]
teapot_vertices_out = sys.argv[2]
teapot_normals_out = sys.argv[3]
teapot_indices_out = sys.argv[4]

with open(teapot_obj, 'r') as f:
    buf = filter(lambda line: len(line) > 0, f.read().split('\n'))
    verts = filter(lambda line: line.split(' ')[0] == 'v', buf)
    norms = filter(lambda line: line.split(' ')[0] == 'vn', buf)
    faces = filter(lambda line: line.split(' ')[0] == 'f', buf)
    if len(verts) == 0:
        raise Exception('Missing vertices in obj')
    if len(norms) == 0:
        raise Exception('Missing normals in obj')
    if len(faces) == 0:
        raise Exception('Missing faces in obj')
    raw_vertices = []
    raw_normals = []
    for vert in verts:
        nums = vert[2:].split(' ')
        raw_vertices.append(tuple(map(float, nums)))
    for norm in norms:
        raw_normals.append(tuple(map(float, norm[3:].split(' '))))
    counter = 0
    vinis = {}
    indices = []
    
    for face in faces:
        parts = face[2:].split(' ')
        if len(parts) != 3:
            raise Exception('Only objs with triangular faces are supported')
        for s in parts:
            vini = tuple(map(int, s.split('//')))
            if vini in vinis:
                index = vinis[vini]
            else:
                vinis[vini] = counter
                index = counter
                counter += 1
            indices.append(index)

    vinis_list = map(lambda e: e[0], sorted(vinis.items(), lambda a, b: a[1]-b[1]))
    vertices = map(lambda vini: raw_vertices[vini[0]-1], vinis_list)
    normals = map(lambda vini: raw_normals[vini[1]-1], vinis_list)
    with open(teapot_vertices_out, 'w') as g:
        g.write('{\n')
        for x, y, z in vertices[:-1]:
            g.write('\t{}f, {}f, {}f, \n'.format(x, y, z)) 
        x, y, z =  vertices[-1]
        g.write('\t{}f, {}f, {}f \n'.format(x, y, z))
        g.write('}')
    with open(teapot_normals_out, 'w') as g:
        g.write('{\n')
        for x, y, z in normals[:-1]:
            g.write('\t{}f, {}f, {}f, \n'.format(x, y, z)) 
        x, y, z = normals[-1]
        g.write('\t{}f, {}f, {}f \n'.format(x, y, z))
        g.write('}')
    with open(teapot_indices_out, 'w') as g:
        g.write('{')
        for i in indices[:-1]:
            g.write('{},'.format(i))
        g.write('{}'.format(indices[-1]))
        g.write('}')
