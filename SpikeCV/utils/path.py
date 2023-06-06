# -*- encoding: utf-8 -*-

# here put the import lib
import os


def split_path_into_pieces(path: str):
    pieces = []
    
    while True:
        splits = os.path.split(path)
        if splits[0] == '':
            pieces.insert(0, splits[-1])
            break
        pieces.insert(0, splits[-1])
        path = splits[0]
    
    return pieces

def replace_identifier(path: list, src: str, dst: str):
    new_path = []
    for piece in path:
        added_piece = piece
        if piece == src:
            added_piece = dst
        new_path.append(added_piece)
    
    return new_path