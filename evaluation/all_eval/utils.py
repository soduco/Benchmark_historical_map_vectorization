import os
import sys
import json
import re
import os
import glob
import pickle
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import time

__all__ = ["json_read", "json_write", "pickle_read", "pickle_write", 
           "mkdir", "sort_nicely", "find_files", "render_segments", "interpolate_new_nodes",
           "plot_graph", "load_graph_txt", "save_graph_txt", "oversampling_graph",
           "shift_graph", "crop_graph", "length_path", "find_closest", 
           "uniform_node_sampling", "node_degree", "is_intersection", "is_end_point",
           "is_control_nodes", "is_intersection", "relabel_nodes", "undersampling_graph",
           "simplify_graph_ramer_douglas_peucker", "f1_score", "edges_count", "is_empty"]

def json_read(filename):
    try:
        with open(os.path.abspath(filename)) as f:    
            data = json.load(f)
        return data
    except:
        raise ValueError("Unable to read JSON {}".format(filename))
        
def json_write(filename, data):
    try:
        directory = os.path.dirname(os.path.abspath(filename))
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(os.path.abspath(filename), 'w') as f:
            json.dump(data, f, indent=2)
    except:
        raise ValueError("Unable to write JSON {}".format(filename))   
        
def pickle_read(filename):
    with open(filename, "rb") as f:    
        data = pickle.load(f)
    return data

def pickle_write(filename, data):
    directory = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)        

def mkdir(directory):
    directory = os.path.abspath(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key=alphanum_key)

def find_files(file_or_folder, hint=None, recursive=False):
    # make sure to use ** in file_or_folder when using recusive
    # ie find_files("folder/**", "*.json", recursive=True)
    import os
    import glob
    if hint is not None:
        file_or_folder = os.path.join(file_or_folder, hint)
    filenames = [f for f in glob.glob(file_or_folder, recursive=recursive)]
    filenames = sort_nicely(filenames)    
    filename_files = []
    for filename in filenames:
        if os.path.isfile(filename):
            filename_files.append(filename)                 
    return filename_files
        
def render_segments(segments, filename=None, height=3072, width=3072, thickness=4):
    
    if isinstance(segments, np.ndarray):
        segments = segments.tolist()
    
    from PIL import Image, ImageDraw

    im = Image.new('RGB', (int(width), int(height)), (0, 0, 0)) 
    draw = ImageDraw.Draw(im) 
    for p1,p2 in segments:
        xy = [round(x) for x in p1]+[round(x) for x in p2]
        draw.line(xy, fill=(255,255,255), width=thickness)
    if filename is not None:
        mkdir(os.path.dirname(filename))
        im.save(filename) 
    return np.array(im)

def plot_graph(graph, node_size=20, font_size=-1, 
               node_color='y', edge_color='y', 
               linewidths=2, offset=np.array([0,0]), **kwargs):
  
    pos = dict({n:np.reshape(graph.nodes[n]['pos'], (2,))+offset for n in graph.nodes()})
    nx.draw_networkx(graph, pos=pos, node_size=node_size, node_color=node_color,
                     edge_color=edge_color, font_size=font_size, **kwargs)
    #plt.gca().invert_yaxis()
    plt.legend()     
    
def load_graph_txt(filename):
    G = nx.Graph()

    nodes = []
    edges = []
    i = 0
    switch = True
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if len(line)==0 and switch:
                switch = False
                continue
            if switch:
                x,y = line.split(' ')
                G.add_node(i, pos=(float(x),float(y)))
                i+=1
            else:
                idx_node1, idx_node2 = line.split(' ')
                G.add_edge(int(idx_node1),int(idx_node2))
    
    return G

def save_graph_txt(G, filename):
    
    mkdir(os.path.dirname(filename))
    
    nodes = list(G.nodes())
    
    file = open(filename, "w+")
    for n in nodes:
        file.write("{:.6f} {:.6f}\r\n".format(G.nodes[n]['pos'][0], G.nodes[n]['pos'][1]))
    file.write("\r\n")
    for s,t in G.edges():
        file.write("{} {}\r\n".format(nodes.index(s), nodes.index(t)))
    file.close()  
    
def edges_count(G):
    return len(G.edges())

def is_empty(G):
    return len(G.edges())==0
    
def interpolate_new_nodes(p1, p2, spacing=2):
    
    _p1 = np.reshape(p1, (2,))
    _p2 = np.reshape(p2, (2,))
    
    diff = _p1-_p2
    segment_length = np.linalg.norm(diff)

    new_node_pos = _p1 -diff*np.linspace(0,1,int(np.round(segment_length/spacing)+1))[1:-1,None]

    return new_node_pos      
    
def oversampling_graph(G, spacing=20):
    """
    Add new regularly spaced nodes in each edge.
    The distance between nodes conncted by an edge will
    approximately equal to the param 'spacing'
    """
    G_ = G.copy()
    
    edges = list(G_.edges())
    for s,t in edges:

        new_nodes_pos = interpolate_new_nodes(G_.nodes[s]['pos'], G_.nodes[t]['pos'], spacing)

        if len(new_nodes_pos)>0:
            G_.remove_edge(s,t)
            n = max(G_.nodes())+1

            for i,n_pos in enumerate(new_nodes_pos):
                G_.add_node(n+i, pos=tuple(n_pos))

            G_.add_edge(s,n)
            for _ in range(len(new_nodes_pos)-1):
                G_.add_edge(n,n+1)
                n+=1
            G_.add_edge(n,t)
    return G_

def undersampling_graph(G, spacing=10, inplace=False):

    if inplace:
        _G = G
    else:
        _G = G.copy()

    def distance(g, n1, n2):
        return np.sqrt((g.nodes[n1]['pos'][0]-g.nodes[n2]['pos'][0])**2+\
                       (g.nodes[n1]['pos'][1]-g.nodes[n2]['pos'][1])**2)
    
    _spacing = spacing/2
    
    # shuffling the nodes is necessary to avoid 
    # making a long sequence of segments a single long straight one
    nodes = list(_G.nodes())
    random.shuffle(nodes)

    for n in nodes:

        # chnage only the nodes that have two adjacent edges
        if len(_G.edges(n))==2:
            ajacent_nodes = list(nx.neighbors(_G, n))

            d1 = distance(_G, n, ajacent_nodes[0])
            d2 = distance(_G, n, ajacent_nodes[1])

            if d1<_spacing or d2<_spacing:

                _G.add_edge(ajacent_nodes[0], ajacent_nodes[1])
                _G.remove_node(n)
 
    return _G

def simplify_graph_ramer_douglas_peucker(G, epsilon=5, verbose=True, inplace=False):
    import rdp
    
    if inplace:
        _G = G
    else:
        _G = G.copy()
    
    start = time.time()
    def f():   
        start = time.time()
        
        nodes = list(_G.nodes())
        random.shuffle(nodes)        
        
        changed = False
        for n in nodes:
            
            if verbose:
                delta = time.time()-start
                if delta>5:
                    start = time.time()
                    if verbose:
                        print("Ramer-Douglas-Peucker remaining nodes:", len(_G.nodes()))

            ajacent_nodes = list(nx.neighbors(_G, n))
            if n in ajacent_nodes:
                ajacent_nodes.remove(n)
            if len(ajacent_nodes)==2:
                node_triplet = [_G.nodes[ajacent_nodes[0]]['pos'], 
                                _G.nodes[n]['pos'], 
                                _G.nodes[ajacent_nodes[1]]['pos']]
                if len(rdp.rdp(node_triplet, epsilon=epsilon))==2:
                    _G.add_edge(*ajacent_nodes)
                    _G.remove_node(n)
                    changed = True
        return changed

    while True:
        if not f():
            break
            
    if verbose:
        print("Ramer-Douglas-Peucker remaining nodes:", len(_G.nodes())) 
        
    return _G

def shift_graph(G, shift_x, shift_y):
    G_ = G.copy()
    for _,data in G_.nodes(data=True):
        x,y = data['pos']
        x,y = x+shift_x,y+shift_y
        if isinstance(data['pos'], np.ndarray):
            data['pos'] = np.array([x,y])
        else:
            data['pos'] = (x,y)
    return G_

def crop_graph_naive(G, xmin=None, ymin=None, xmax=None, ymax=None):
    G_ = G.copy()
    for n in list(G_.nodes()):
        p = G_.nodes[n]['pos']
        if p[0]>=xmin and p[0]<xmax and p[1]>=ymin and p[1]<ymax:
            pass
        else:
            G_.remove_node(n)
    return G_

def segments_intersection_point(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def segment_intersection_point_to_box(segment, xmin, ymin, xmax, ymax): 

    bs = [((xmin, ymin),(xmin, ymax)), 
                       ((xmin, ymin),(xmax, ymin)), 
                       ((xmin, ymax),(xmax, ymax)),
                       ((xmax, ymin),(xmax, ymax))]
    P = np.array([b[0] for b in bs])
    Q = np.array([b[1] for b in bs])    

    p1, p2 = segment

    p1_out = p1[0]<xmin or p1[0]>=xmax or p1[1]<ymin or p1[1]>=ymax
    p2_out = p2[0]<xmin or p2[0]>=xmax or p2[1]<ymin or p2[1]>=ymax
    
    if not p1_out and not p2_out:
        return None

    if p1_out and not p2_out:
        
        X = np.reshape(p1, (1,2))
        S, D, id = closest_points_on_segments(X, P, Q)
        idx_closer_segment = np.argmin(D[0])        
        
        new_p1 = segments_intersection_point(bs[idx_closer_segment], segment)
        return (new_p1, p2)
    
    elif p2_out and not p1_out:
        
        X = np.reshape(p2, (1,2))
        S, D, id = closest_points_on_segments(X, P, Q)
        idx_closer_segment = np.argmin(D[0])        
        
        new_p2 = segments_intersection_point(bs[idx_closer_segment], segment)
        return (p1, new_p2)
    
def crop_graph(G, xmin=None, ymin=None, xmax=None, ymax=None):
    G_ = G.copy()
    for s,t in list(G_.edges()):
        p1 = G_.nodes[s]['pos']
        p2 = G_.nodes[t]['pos']
        
        p1_out = p1[0]<xmin or p1[0]>=xmax or p1[1]<ymin or p1[1]>=ymax
        p2_out = p2[0]<xmin or p2[0]>=xmax or p2[1]<ymin or p2[1]>=ymax
        
        if p1_out and p2_out:
            G_.remove_edge(s,t)
        elif not p1_out and not p2_out:
            pass
        elif p1_out:

            new_seg = segment_intersection_point_to_box((p1,p2), xmin, ymin, xmax, ymax)
            new_node = max(G_.nodes())+1
            G_.add_node(new_node, pos=new_seg[0])
            G_.add_edge(new_node, t)
            G_.remove_edge(s, t)
                        
        elif p2_out:
            new_seg = segment_intersection_point_to_box((p1,p2), xmin, ymin, xmax, ymax)
            new_node = max(G_.nodes())+1
            G_.add_node(new_node, pos=new_seg[1])
            G_.add_edge(s, new_node) 
            G_.remove_edge(s, t)
            
    # remove nodes that are not attached to any edge
    for n in list(G_.nodes()):
        if len(G_.edges(n))==0:
            G_.remove_node(n)
            
    return G_    

def length_path(G, path):
    length = 0
    for i in range(len(path)-1):
        p1 = np.array(G.nodes[path[i]]['pos'])
        p2 = np.array(G.nodes[path[i+1]]['pos'])
        length += np.linalg.norm(p1-p2)
    return length

def find_closest(point, points): 
    dists = np.linalg.norm(points-point[None], axis=1)
    idx_min = np.argmin(dists)
    dist_min = dists[idx_min]
    return dist_min, idx_min

def node_degree(G, node):
    return len(G.edges(node))

def is_intersection(G, node):
    return node_degree(G, node)>2

def is_end_point(G, node):
    return node_degree(G, node)==1

def is_control_nodes(G, node):
    return is_intersection(G, node) or is_end_point(G, node)   

def is_intersection(G, node):
    if len(G.edges(node))>2:
        return True
    else:
        return False

def uniform_node_sampling(G, dist_matching=25, max_node_probe=10000):
    start = time.time()
    
    nodes = list(G.nodes())
    
    # limit on the number of nodes, it makes this function slow otherwise
    random.shuffle(nodes)
    nodes = nodes[:max_node_probe]   
    
    nodes_pos = np.vstack([G.nodes[n]['pos'] for n in nodes])
    
    xmin, ymin = nodes_pos.min(0)
    xmax, ymax = nodes_pos.max(0)    
    
    random_node = None
    for _ in range(10000):
        x = np.random.uniform(low=xmin, high=xmax)
        y = np.random.uniform(low=ymin, high=ymax)
        random_position = np.array([x,y])

        dists = np.linalg.norm(nodes_pos-random_position[None], axis=1)
        idx_min = np.argmin(dists)
        if dists[idx_min]>dist_matching:
            random_node = nodes[idx_min]
            break

    if random_node is None:
        random_node = np.random.choice(G.nodes())
        print("uniform_node_sampling: node picked from the set of nodes of the graph!")
        
    return random_node     
    
def uniform_node_sampling_with_snapping(G, dist_matching=25):    
    
    nodes_pos_gt = np.vstack([G.nodes[n]['pos'] for n in G.nodes()])
    xmin, ymin = nodes_pos_gt.min(0)
    xmax, ymax = nodes_pos_gt.max(0)
    
    edges = list(G.edges())
    P = np.array([G.nodes[s]['pos'] for s,t in edges])
    Q = np.array([G.nodes[t]['pos'] for s,t in edges])    

    for _ in range(100):
        xs = np.random.uniform(low=xmin, high=xmax, size=100)
        ys = np.random.uniform(low=ymin, high=ymax, size=100) 
        random_positions = np.vstack([xs, ys]).T

        S, D, id = closest_points_on_segments(random_positions, P, Q)

        random_node = None
        for idx_point, point in enumerate(random_positions):

            idx_closest_edge = D[idx_point].argmin()
            dist = D[idx_point, idx_closest_edge]

            if dist<dist_matching:
                
                if id[idx_point, idx_closest_edge]==0:
                    random_node = edges[idx_closest_edge][0]
                elif id[idx_point, idx_closest_edge]==1:
                    random_node = edges[idx_closest_edge][1]
                else:
                    s,t = edges[idx_closest_edge]   
                    new_nodes = [max(G.nodes())+1]
                    new_nodes_pos = [S[idx_point, idx_closest_edge]]
                    G = insert_nodes_in_edge(G, s, t, new_nodes, new_nodes_pos) 

                    random_node = new_nodes[0]
                break

        if random_node is not None:
            break

    if random_node is None:
        random_node = np.random.choice(G.nodes())
        print("uniform_node_sampling: node picked from the set of nodes of the graph!")

    return G, random_node    

def closest_point_on_segment(X, P, Q):
    """
    Computes the closest point on a segment to a point
    
    Parameters
    ----------
    X : np.array (M,)
        point in the space
    P and Q : np.array (M,)
        points defining the start and end of the segment
    
    Return
    ------
    S : np.array (M,)
        the closest point to X on the segment
    D : float
        distance from the point X to the closest on the segment        
    id : int
        0 if S=P, 1 if S=Q and None if on the segment
    """
    Q_P = Q-P
    lambd = np.dot(X-P,Q_P)/(np.dot(Q_P,Q_P)+1e-12)
        
    if lambd<=0:
        S = P    
        id = 0
    elif lambd>=1:
        S = Q
        id = 1
    else:
        S = P + lambd*Q_P
        id = None 
        
    D = np.linalg.norm(S-X)
        
    return S, D, id

def closest_points_on_segments(X, P, Q):
    """
    Computes the closest point on a segment to a point
    for all points and all segments
    
    Parameters
    ----------
    X : numpy.ndarray (N,M)
        points in the space
    P and Q : numpy.ndarray (O,M)
        points defining the start and end of the segments
    
    Return
    ------
    S : numpy.ndarray (N,O,M)
        the closest points to X on the segments
    D : numpy.ndarray (N,O)
        distance from the points X to the closests on the segments
    id : numpy.int (N,O)
        0 if S=P, 1 if S=Q and None if on the segment
    """
    assert len(X)!=0    
    assert len(P)!=0
    assert len(Q)!=0    
    
    N,M = X.shape
    
    Q_P = (Q-P)[None]
    X_P = X[:,None]-P[None]
    lambdas = np.sum(X_P*Q_P, axis=2)/(np.sum(Q_P*Q_P, axis=2)+1e-12) # [N,O]
    
    id = np.array([[None]*len(P)]*len(X)) # [N,O]
    id[lambdas<=0] = 0
    id[lambdas>=1] = 1    
    
    lambdas = np.repeat(lambdas[:,:,None], M, axis=2) # [N,O,M]
    S = P[None] + lambdas*Q_P                         # [N,O,M]
    np.putmask(S, lambdas<=0, np.repeat(P[None], N, axis=0))
    np.putmask(S, lambdas>=1, np.repeat(Q[None], N, axis=0)) 
    
    D = np.linalg.norm(S-X[:,None], axis=2)
    
    return S, D, id

def insert_nodes_in_edge(G, s, t, nodes, nodes_pos):
    
    G_ = G#.copy()
    
    # reorder nodes positions
    def distance(idx):
        return (G_.nodes[s]['pos'][0]-nodes_pos[idx][0])**2+\
                  (G_.nodes[s]['pos'][1]-nodes_pos[idx][1])**2  
    idxs = list(range(len(nodes)))
    idxs.sort(key=lambda idx: distance(idx))
    
    G_.remove_edge(s,t)
    G_.add_node(nodes[idxs[0]], pos=nodes_pos[idxs[0]], snapped=True) 
    G_.add_edge(s, nodes[idxs[0]])
    for i_1, i in zip(idxs[:-1], idxs[1:]):
        G_.add_node(nodes[i], pos=nodes_pos[i], snapped=True) 
        G_.add_edge(nodes[i_1], nodes[i])
    G_.add_edge(nodes[idxs[-1]], t)  
    
    return G_

def snap_points_to_graph(G, points, th_existing=10, th_snap=25, inplace=False):
    
    name_new_node = lambda i: str(i)+"_snapped"
    
    if inplace:
        G_ = G
    else:
        G_ = G.copy()
    
    points_ = np.reshape(points, (-1,2))
    edges = list(G.edges())
    s_nodes = np.array([G.nodes[s]['pos'] for s,t in edges])
    t_nodes = np.array([G.nodes[t]['pos'] for s,t in edges])

    S, D, id = closest_points_on_segments(points_, s_nodes, t_nodes)

    # find the edges where to snap the new points
    to_snap = {}
    correspondences = []
    for idx_point, point in enumerate(points_):
        
        idx_closest_edge = D[idx_point].argmin()
        dist = D[idx_point, idx_closest_edge]
        
        s,t = edges[idx_closest_edge] 
        
        # do not snap the point if it is too far form any edge
        if dist<th_snap:
            
            # do not create an additional node if the closest point is the 
            # starting or ending nodes of the edge
            if id[idx_point, idx_closest_edge]==0:
                correspondences.append(s)
            elif id[idx_point, idx_closest_edge]==1:
                correspondences.append(t)
            else:
                # If one between the starting or ending nodes is very close to the point
                # do not create an additional node in the graph.
                if np.linalg.norm(s_nodes[idx_closest_edge]-point)<th_existing:
                    correspondences.append(s)
                elif np.linalg.norm(t_nodes[idx_closest_edge]-point)<th_existing:
                    correspondences.append(t)
                else:
                    if idx_closest_edge not in to_snap:
                        to_snap[idx_closest_edge] = []
                    to_snap[idx_closest_edge].append(idx_point)
                    correspondences.append(name_new_node(idx_point))
           
        else:
            correspondences.append(None)
                           
    # modify the edges
    for idx_closest_edge, idxs_points in to_snap.items():            
                
        s,t = edges[idx_closest_edge]   

        new_nodes = [name_new_node(i) for i in idxs_points]
        new_nodes_pos = [S[idx_point, idx_closest_edge] for idx_point in idxs_points]
        G_ = insert_nodes_in_edge(G_, s, t, new_nodes, new_nodes_pos)       
            
    return G_, correspondences

def relabel_nodes(G, mapping=None):
    G_ = G.copy()
    if mapping is None:
        mapping = dict(zip(G_.nodes(), range(len(G_.nodes()))))
    G_ = nx.relabel_nodes(G_, mapping)
    return G_

def f1_score(precision, recall):
    return 2*(precision*recall)/(precision+recall)