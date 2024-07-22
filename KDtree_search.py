from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import numpy as np
import datetime
import skimage
import time

import image_to_vertices as itv
from image_to_vertices import vertex_generation



def k_search_all():

    # Iterate through each pixel and find its k nearest neighbors
    for i in range(len(flattened_data)):
        pixel_value = flattened_data[i]

        # Query the KDTree
        _, neighbor_indices = kdtree.query(pixel_value, k + 1)

        # print(i, neighbor_indices)

        tmp = [[i], []]
        max_dist = 0
        for j in range(len(neighbor_indices)):
            if neighbor_indices[j] != i:
                tmp[1].append(neighbor_indices[j])
                if np.linalg.norm(flattened_data[neighbor_indices[j]] - flattened_data[i]) > max_dist:
                    max_dist = np.linalg.norm(flattened_data[neighbor_indices[j]] - flattened_data[i])

        # Max_dist ball neighborhood option
        all_neighbor_indices = kdtree.query_ball_point(pixel_value, max_dist, p=2)
        # print("Max_dist ball:", all_neighbor_indices)
        for j in range(len(all_neighbor_indices)):
            if all_neighbor_indices[j] != i and all_neighbor_indices[j] not in tmp[1]:
                tmp[1].append(all_neighbor_indices[j])

        # Brute force option
        #for j in range(len(flattened_data)):
        #    if not np.array_equal(flattened_data[j], flattened_data[i]) and j not in tmp[1]:
        #        if np.linalg.norm(flattened_data[j] - flattened_data[i]) <= max_dist:
        #            tmp[1].append(flattened_data[j])

        hyperarc_k.append(tmp)

        # print("At least k nearest neighbors:", tmp)
        # for j in range(len(tmp[1])):
        #     print(tmp[1][j], np.linalg.norm(flattened_data[tmp[1][j]] - flattened_data[i]))
        # print()



def epsilon_search_all():
    # Iterate through each pixel and find neighbors within epsilon radius
    for i in range(len(flattened_data)):
        pixel_value = flattened_data[i]

        # Query the KDTree to find neighbors within epsilon radius
        neighbor_indices = kdtree.query_ball_point(pixel_value, epsilon, p=2)
        neighbor_indices.sort()

        #tmp = []
        #for j in range(len(neighbor_indices)):
        #    tmp.append(neighbor_indices[j])
        #tmp.sort()

        # Keep only those vertices that have neighbours
        #if len(neighbor_indices) > 1: #and neighbor_indices not in hyperedge_eps:
        if len(neighbor_indices) > 1 and neighbor_indices not in hyperedge_eps:
            hyperedge_eps.append(neighbor_indices)
            hyperedge_eps_pix_values.append(flattened_data[neighbor_indices].flatten())

            # print(i, tmp)
            # for j in range(len(tmp)):
            #     print(tmp[j], np.linalg.norm(flattened_data[tmp[j]] - flattened_data[i]))
            # print()


          #  Show all vertices include vertices with zero neighbours
        #tmp_arc = [[], []]
        #for j in range(len(neighbor_indices)):

        #    tmp_arc[0] = [i]
            # Exclude the pixel itself
        #    if neighbor_indices[j] != i:
        #        tmp_arc[1].append(neighbor_indices[j])

        # if len(tmp[1]) > 0:
        #hyperedge_eps_all.append(tmp_arc)
        if i % 100 == 0:
            print("Current Step: " + str(i) + "/" + str(len(flattened_data)))
    print("Done")

    means = []
    differences = []
    tmp = []

   
    # Finding the means of the main node and its found neighbors in hyperedge.

    means = []
    differences = []
    tmp = []

    for arr in hyperedge_eps_pix_values:
        mean_for_all = np.mean(arr[1:])
        means.append(mean_for_all)
        # first_element = arr[0]
        # difference = hyperedge_eps_pix_values - mean_for_all
        # hyperedge_eps_all_difference.append(difference)

    for array_a in hyperedge_eps_pix_values:
        diff = array_a - means[len(differences)]
        differences.append(abs(diff))

    sums_of_differences = [np.sum(diff) for diff in differences]

    # print("hyperedge_eps_pix_values:" + str(hyperedge_eps_pix_values))
    # print("means:" + str(means))
    # print("dif:" + str(differences))
    # print("sum of diff:" + str(sums_of_differences))


# Calculate the sum for each array in a
    sums_of_arrays = np.round([np.sum(array) for array in differences], decimals=2)


# Print the sums of each array
#     print("Sums of arrays:", sums_of_arrays)

    for arr in differences:
        array_length = len(arr)
        tmp.append(1/(array_length - 1))

    # Multiply each element in sigma_vector by the corresponding element in sums_of_arrays
    for sigma, sum_array in zip(tmp, sums_of_arrays):
        sigma_kdt.append(sigma * sum_array)

    # print("----------------------")
    # print("Sigma", sigma_kdt)


def localNeighborhood(height, width):

    coordinates = np.unravel_index(list(range(width*height)), (height, width))

    hyperedges_local = []

    for vertex in range(width*height):
        cY = coordinates[0][vertex]
        cX = coordinates[1][vertex]


        hyperedge = [vertex]
        
        if cX > 0 and cX < width-1 and cY > 0 and cY < height-1:
            hyperedge += list(np.ravel_multi_index(np.array([[cY, cY-1, cY+1, cY],[cX-1, cX, cX, cX+1]]),(height, width)))
        else:

            # we assume Neumann boundary conditions
            if cX == 0:
                hyperedge += list(np.ravel_multi_index(np.array([[cY, cY], [cX, cX+1]]), (height, width)))
            elif cX == width-1:
                hyperedge += list(np.ravel_multi_index(np.array([[cY, cY], [cX-1, cX]]), (height, width)))
            else:
                hyperedge += list(np.ravel_multi_index(np.array([[cY, cY],[cX-1, cX+1]]), (height, width)))

            if cY == 0:
                hyperedge += list(np.ravel_multi_index(np.array([[cY, cY+1],[cX, cX]]), (height, width)))
            elif cY == height-1:
                hyperedge += list(np.ravel_multi_index(np.array([[cY-1, cY],[cX, cX]]), (height, width)))
            else:
                hyperedge += list(np.ravel_multi_index(np.array([[cY-1, cY+1], [cX, cX]]), (height, width)))
        #print(hyperedge)
        assert len(hyperedge) == 5  # sanity check
        hyperedges_local.append(hyperedge)

    return hyperedges_local


# Let's measure elapsed time
start = time.time()


batch_size = 1
vertices = vertex_generation(batch_size)
# print()
# print("Vertices: " + str(len(vertices)))
# print(vertices)
# print()

# Flatten the image data for easier processing
flattened_data = np.array(vertices).reshape(-1, batch_size)
# print("Flattened Data: ")
# print(flattened_data)

# Create a KDTree
print("Building a KD tree")
kdtree = cKDTree(flattened_data)
print("KD tree built!")

# Set a k number and an epsilon radius for nearest neighbors
k = 6
epsilon = 10

now = datetime.datetime.now()

# Visualization
#plt.figure(figsize=(8, 9))
#plt.imshow(image, cmap=plt.cm.gray)
# plt.axis('off')
#plt.tight_layout()

hyperarc_k = []
hyperedge_eps = []
hyperedge_eps_all = []

hyperedge_eps_pix_values = []
hyperedge_eps_all_difference = []
hyperedge_eps_difference_element_mean = []

sigma_kdt = []
hyperedge_eps = localNeighborhood(169, 200)


# k_search_all()
# epsilon_search_all()
#
# print("---")
# print("Results: ")
#
# print("---")
# print("K = " + str(k) + " or more nearest neighbors: ")
# print(hyperarc_k)
# #
# print("---")
# print("Nearest neighbors within radius epsilon = " + str(epsilon) + ": ")
# print(hyperedge_eps)
# print("---")
#
# print("k-Vertices: " )
vertices_k = [sublist[0][0] for sublist in hyperarc_k]
# print(vertices_k)
#
# print("eps-Vertices: " )
#vertices_eps = [sublist[0][0] for sublist in hyperedge_eps]
vertices_eps = list(range(0, len(flattened_data)))
# print(vertices_eps)


#plt.legend()
#plt.show()

end = time.time() - start
print("Elapsed time for building neighborhood: " + str(end))
