#!/bin/bash
if [[ -z $1 ]]; then
    echo 'Missing argument' 1>&2
    exit 1
fi
root=$(dirname $(realpath $0))
python $root/obj_to_inc.py $root/$1.obj $root/../models/$1_vertices.inc $root/../models/$1_normals.inc $root/../models/$1_indices.inc
