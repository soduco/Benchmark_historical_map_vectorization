import torch
import torch.nn as nn
import numpy as np
import pathloss
import pdb

def Path_loss(inputs, seed, boundary):
    '''
    inputs : prediction image
    seed: seeding pixel (geodesic center of the connected component)
    boundary: boundary of the connected component
    '''
    EPM = inputs.squeeze()

    seed = seed.squeeze()
    boundary = boundary.squeeze()

    _, _ = seed.shape

    EPM_s = EPM

    EPM_s_clone = EPM_s.clone()
    EPM_s_clone = np.array((EPM_s_clone).detach().cpu().numpy()).astype(np.uint8)*255
    EPM_s = torch.sigmoid(EPM_s)

    loss = 0

    # transform tensor to array to find the shortest path in C++
    max_range = torch.max(seed).type(torch.uint8)
    seed = np.array(seed.detach().cpu().numpy()).astype(np.uint8)    
    destination = np.array(seed).astype(np.uint8)

    if max_range>1:
        start = np.array(seed == 1).astype(np.uint8)*255
        max_value_array = torch.zeros(max_range-1).cuda()
        max_value_array_gt = torch.ones(max_range-1).cuda() 

        # find the shortest path using djikstra-like algorithm
        shortest_path = pathloss.geodesic_shortest_all(EPM_s_clone,start,destination)
        shortest_path = torch.from_numpy(np.array([shortest_path])).cuda()
        
        for j in range(2, max_range +1):
            path_tmp = shortest_path == j
            
            # get the value of the intersection between the shortest path 
            # and the boundary of the connected component 
            maximum_value = torch.max(EPM_s * path_tmp * boundary)
            max_value_array[j-2] = maximum_value

        # compute the MSE loss function
        loss = loss + nn.MSELoss()(max_value_array, max_value_array_gt)
    else:
        loss = torch.tensor(0).type(torch.float64).cuda()
    return loss
