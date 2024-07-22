import random
import numpy as np
import KDtree_search as kds
import matplotlib.pyplot as plt
import image_gaussian_noise as ign


# p-Laplacian operator for oriented hypergraphs
def p_laplacian_or(f, p, w_I, w_G, W_I, W_G, vertices, hyperarcs, membership_arcs):

    # parameter for making sure we don't devide by zero
    if p < 2:
        epsilon = 0.05
    else:
        epsilon = 0.0

   # check for isolated vertices
    isolated_vertices = []
    for i in vertices:
        if len(membership_arcs[i]) == 0:
            print("Empty vertex found: " + str(i))
            isolated_vertices.append(i)

    p_lap = np.zeros(len(f))

     # compute p-Laplacian
    for i in range(len(vertices)):

        # skip isolated vertices
        if i in isolated_vertices:
            continue

        # compute influence of arcs in which v_i is input vertex
    
        #### We assume there is currently only one hyperarc in which v_i is an input node, namely the hyperarc a_i
        a_i = hyperarcs[i]
        p1 = - w_G[i] * W_I[i] * W_G[i] ** p

        # compute second term
        p2 = 0
        for j in a_i[1]:
            p2 = p2 - w_I[j] * w_G[j] / len(a_i[1]) * f[j]

        p3 = p2

        p2 = (abs(p2) + epsilon) ** (p - 2)

        p_lap[i] = p1 * p2 * p3
        

        # compute influence of arcs in which v_i is output vertex
        for q in membership_arcs[i][1]:

            # extract output nodes of hyperarc a_q
            a_q = hyperarcs[q][1]

            # compute first term
            p1 = w_G[i] / len(a_q) * W_I[q] * W_G[q] ** p

            # compute second term
            p2 = 0
            for j in a_q:
                p2 = p2 - w_I[j] * w_G[j] / len(a_q) * f[j]
            # don't forget the input node
            p2 = p2 + w_I[i] * w_G[i] * f[i]

            # third term is basically the same
            p3 = p2

            p2 = (abs(p2) + epsilon) ** (p - 2)

            p_lap[i] = p_lap[i] + p1 * p2 * p3

    return p_lap

# fast averaging operator as derived by Ksenia and Ariane
def p_laplacian_fast_new(f, hyperedges, memberships, scaling_factors):

    p_lap = np.zeros(len(f))
    weighted_average = np.zeros(len(hyperedges))
    number_of_vertices = np.zeros(len(hyperedges))

    # compute per hyperedge
    for q, e_q in enumerate(hyperedges):
        weighted_average[q] = sum(scaling_factors[e_q] * f[e_q]) #/ len(e_q)
        number_of_vertices[q] = len(e_q)

    # compute Laplacian
    for i in range(len(vertices)):
        for q in memberships[i]:
            p_lap[i] = p_lap[i] + weighted_average[q] / number_of_vertices[q]

    return p_lap

# fast averaging operator as average of averaged hyperedges
def p_laplacian_fast_old(f, hyperedges, memberships, scaling_factors):

    p_lap = np.zeros(len(f))
    weighted_average = np.zeros(len(hyperedges))
    number_of_vertices = np.zeros(len(hyperedges))

    # compute per hyperedge
    for q, e_q in enumerate(hyperedges):
        #weighted_average[q] = sum(scaling_factors[e_q] * f[e_q]) #/ len(e_q)
        weighted_average[q] = sum(f[e_q]) / len(e_q)
        number_of_vertices[q] = len(e_q)

    # compute Laplacian
    for i in range(len(vertices)):
        for q in memberships[i]:
            #p_lap[i] = p_lap[i] + weighted_average[q] / number_of_vertices[q]
            p_lap[i] = p_lap[i] + weighted_average[q] 
        p_lap[i] = p_lap[i] / len(memberships[i])

    return p_lap

# p-Laplacian operator for non-oriented hypergraphs
def p_laplacian_nr(f, p, w_I, w_G, W_I, W_G, vertices, hyper, membership):

    # parameter for making sure we don't devide by zero
    if p < 2:
        epsilon = 0.05
    else:
        epsilon = 0.0

    # check for isolated vertices
    isolated_vertices = []
    for i in vertices:
        if len(membership[i]) == 0:
            print("Empty vertex found: " + str(i))
            isolated_vertices.append(i)

    p_lap = np.zeros(len(f))

    # compute p-Laplacian
    for i in range(len(vertices)):

        # skip isolated vertices
        if i in isolated_vertices:
            continue
        
        for q in membership[i]:

            # extract hyperedge
            e_q = hyper[q]

            # compute first term
            p1 = w_G[i] * W_I[q] * W_G[q] ** p

            # compute second term
            p2 = 0
            for j in e_q:
                p2 = p2 + w_I[j] * w_G[j] / len(e_q) * f[j]

            # third term is basically the same
            p3 = p2

            p2 = (abs(p2) + epsilon) ** (p - 2)

            p_lap[i] = p_lap[i] + p1 * p2 * p3

        p_lap[i] = p_lap[i] / len(membership[i])

    # print("Number of hyperedges this vertex is in: " + str(len(membership[34])))
    #
    # print("Original pixel value: " + str(f[34]))
    # print("Average value of p-Laplacian: " + str(p_lap[34]))
    #
    # for hyperedge in membership[34]:
    #     print("All values in hyperedge: " + str(f[hyper[hyperedge]]))
    #     print("Average in this hyperedge: " + str(sum(f[hyper[hyperedge]] / len(hyper[hyperedge]))))

    return p_lap


def hypergraph_diffusion(f, p, lambda_reg, delta_t, max_iterations, vertices, hyperarcs, hyperedges, shape, oriented):
    
    # set constant weights for vertices
    w_I = np.ones(len(vertices))
    w_G = np.ones(len(vertices))

     # check if we perform computations on an oriented or non-oriented hypergraph
    if oriented == False:
   
        #scaling_factors = scaling_factor(len(vertices), hyperedges)

        # Set constant weights for hyperedges
        W_I = np.ones(len(hyperedges))
        W_G = np.ones(len(hyperedges))

        # # Set different weights to hyperedges
        # 
        # W_I = np.ones(len(hyperedges))
        # print("Old W_I" + str(W_I))
        #
        # # Check the difference between the main and average values of the found neighbors +
        # # Set the different W_I based on |diff|
        # for i in range(len(W_I)):
        #     W_I[i] =  1 / (sigma[i] + 1)
        #     if W_I[i] > 1:
        #         W_I[i] = 1
        #     
        #
        # # Copy the weights
        # W_G = np.copy(W_I)
        #
        # print("New W_I" + str(W_I))
        # print("New W_G" + str(W_G))



        # initialize hyperedge membership
        membership_edges = []
        for i in vertices:
            membership_edges.append([])

        #print("Vertices: " + str(vertices))
        # compute hyperedge membership
        for q, e_q in enumerate(hyperedges):
            #print("q: " + str(q) + ", edge: " + str(e_q))
            for vertex in e_q:
                membership_edges[vertex].append(q)

        # compute scaling factors
        scaling_factors = np.zeros(len(vertices))
        for i in vertices:
            scaling_factors[i] = 1 / len(membership_edges[i])
        # print some output
        print("Noisy Image with " + str(len(vertices)) + " vertices, "  + str(len(hyperedges)) + " hyperedges.")

    else:
        
		# set constant weights for arcs
        W_I = np.ones(len(hyperarcs))
        W_G = np.ones(len(hyperarcs))

        # initialize hyperarc membership
        membership_arcs = []
        for i in vertices:
            membership_arcs.append([[],[]])

        # compute hyperarc membership
        for q, a_q in enumerate(hyperarcs):
            membership_arcs[q][0] = q # == a_q[0]
            for vertex in a_q[1]:
                membership_arcs[vertex][1].append(q)

        # print some output
        print("Noisy Image with " + str(len(vertices)) + " vertices, "  + str(len(hyperarcs)) + " hyperarcs.")
        
        
    # initialize numpy array for convergence analysis
    distances = np.zeros(max_iterations)

    # initialize figure as non-blocking
    plt.show(block=False)

    # perform iteration scheme for the diffusion equation
    for iteration in range(max_iterations):

        # compute the p-Laplacian either for oriented or non-oriented hypergraphs
        if oriented == False:
            # Non Local ???
            # p_laplacian = p_laplacian_nr(f, p, w_I, w_G, W_I, W_G, vertices, hyperedges, membership_edges)
            # Local
            p_laplacian = p_laplacian_fast_old(f, hyperedges, membership_edges, scaling_factors)
        else:
            p_laplacian = p_laplacian_or(f, p, w_I, w_G, W_I, W_G, vertices, hyperarcs, membership_arcs)

        #print("Min: " + str(min(f)) + " ; Max: " + str(max(f)))
        #print("Min: " + str(min(p_laplacian)) + " ; Max: " + str(max(p_laplacian)))
        
        # print(p_laplacian[21])
        # print(f_new[21])
        
        # Forward Euler time discretization with time step size delta_t
        #f_new = f + delta_t * (f - f0 + lambda_reg * p_laplacian)   # this should be correct but behaves weirdly
        f_new = f + delta_t * ((p_laplacian - f) + lambda_reg*(f0 - f)) # this performs nonlocal denoising without data term 

        #print(np.linalg.norm(f,1))
        #print(np.linalg.norm(p_laplacian,1))
        #f_new = delta_t * (lambda_reg * p_laplacian) # just some testing

        print(sum(f))
        print(sum(p_laplacian))
        # compute Euclidean norm between iterates
        delta_f = np.linalg.norm(f-f_new) / np.linalg.norm(f)

        # save current distance in numpy array
        distances[iteration] = delta_f

        # print("Change between iterations: " + str(delta_f))

        # set reference to updated vertex function for next iteration
        f = f_new
          
        # print("Iteration №: " + str(iteration) + " / " + str(max_iterations))

        # Plot denoised image
        #image_out = np.array(f_new).reshape(shape)
        #plt.imshow(image_out, cmap=plt.cm.gray)
        #plt.title("Denoised image")
        #plt.axis('off')
        #plt.tight_layout()
        #plt.draw()
        #plt.pause(0.001)

    return np.round(f_new)


########## MAIN

# set algorithm parameters
delta_t = 0.03
lambda_reg = 0
p = 2
max_iterations = 100
oriented = False
# sigma = 0.8

# get pixels of noisy image
pixel_values = ign.pixel_values

# get image shape
image_shape = (ign.height, ign.width)

# Access the vertices and hyperedges for epsilon nearest search
vertices = kds.vertices_eps
hyperarcs = kds.hyperedge_eps_all
hyperedges = kds.hyperedge_eps

sigma = kds.sigma_kdt

# initialize images
f = np.copy(pixel_values)
f0 = np.copy(f)

# for delta_t in [ 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.00, 1.30, 1.50, 1.70 ]:#[0.1, 0.25, 0.5, 0.75, 1.00, 1.30, 1.50, 1.70]:
#   for lambda_reg in  [0, 0.001, 0.01, 0.1, 0.5, 1, 2, 5, 20]:

# perform diffusion-based denoising with p-Laplace operator on hypergraphs

f_out = hypergraph_diffusion(f, p, lambda_reg, delta_t, max_iterations, vertices, hyperarcs, hyperedges, image_shape, oriented)

# Plot denoised image
image_out = np.array(f_out).reshape(image_shape)
plt.figure(figsize=(8, 8))
plt.imshow(image_out, cmap=plt.cm.gray)
# plt.title("Denoised image: " + "Lambda " + str(lambda_reg ) + ", Tau " + str(delta_t))
plt.axis('off')
plt.tight_layout()


# plt.savefig('./results/venice_100_eps=' + str(kds.epsilon) + '_t=' + str(delta_t) + '_its=' + str(max_iterations) + '.png', bbox_inches='tight')
plt.savefig('./results/venice_100_lambda=' + str(lambda_reg) + '_t=' + str(delta_t) + '_its=' + str(max_iterations) + '.png', bbox_inches='tight')
plt.show()


# Plot original pic
    #image_out = np.array(f0).reshape(image_shape)
    #plt.figure(figsize=(8, 8))
    #plt.imshow(image_out, cmap=plt.cm.gray)
    #plt.title("Original image")
    #plt.axis('off')
    #plt.tight_layout()
#plt.show()
#  plt.savefig('./results/original_image.png', bbox_inches='tight')
