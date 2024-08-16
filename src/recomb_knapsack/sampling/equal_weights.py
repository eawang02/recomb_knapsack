
import numpy as np
import scipy.stats as stats
from tqdm import tqdm


def equal_weights_step(partitions, i, j, C):
    """
    Take the list of partitions and two indices i and j, and attempt to recombine them into two (closer to) valid partitions.
    Randomly shuffle the two partitions, then split them into left and right partitions based on the split method.

    Equal weights: Split as soon as the left partition exceeds half the total weight of the districts
    """

    # Concatenate + shuffle the two partitions
    new_partition = np.random.permutation(np.concatenate([partitions[i], partitions[j]]))
    cumsums = np.cumsum(new_partition)
    
    # Split the partitions based on the split method
    total_sum = cumsums[-1]
    target = total_sum / 2
    nearest_to_target_idx = np.abs(cumsums - target).argmin()
    partitions[i] = new_partition[:nearest_to_target_idx+1]
    partitions[j] = new_partition[nearest_to_target_idx+1:]
    i_valid = C[0] <= cumsums[nearest_to_target_idx] <= C[1]
    j_valid = C[0] <= (total_sum - cumsums[nearest_to_target_idx]) <= C[1]

    return (i_valid, j_valid)



def equal_weights_recomb(X=None, k=10, tol=0.01, max_iter=10000, verbose=False):
    """
    Sample solutions to the knapsack problem.
    X: Sampled data to knapsack with
    k: Number of districts/partitions
    tol: Tolerance (as a value) for partition sizes, each partition is allowed to be C ± tol
    max_iter: Maximum number of iterations for MCMC sampling before stopping
    verbose: Whether to print out progress updates

    Returns:
        - 
    """

    X = np.random.permutation(X)
    partitions = np.array_split(X, k)
    C = ((sum(X) / k) - tol, (sum(X) / k) + tol)

    if verbose:
        print("Attempting Random Recomb method...")
        print(f"Partitioning into {k} districts in the range [{C[0]}, {C[1]}].")

    
    ## Run burnin steps (10k steps for now)
    for i in range(10000):
        i, j = np.random.choice(k, size=2, replace=False)
        equal_weights_step(partitions, i, j, C)
    

    ## Recomb steps
    solution_counts = {}
    valid_count = 0
    invalid_set = {i for i in range(k) if not (C[0] <= sum(partitions[i]) <= C[1])}
    valid_set = set(range(k)) - invalid_set

    # if len(invalid_set) == 0:
    #     valid_count += 1
        
    #     par_tup = tuple(sorted([tuple(float(x) for x in np.sort(p)) for p in partitions]))
        
    #     solution_counts[par_tup] = solution_counts.get(par_tup, 0) + 1

    for i in range(max_iter):
        i, j = np.random.choice(k, size=2, replace=False)
        invalid_set.add(i)
        invalid_set.add(j)

        # Perform recombination step
        i_bool, j_bool = equal_weights_step(partitions, i, j, C)
        if i_bool:
            invalid_set.discard(i)
            valid_set.add(i)
        if j_bool:
            invalid_set.discard(j)
            valid_set.add(j)

        if len(invalid_set) == 0:
            valid_count += 1
            
            par_tup = tuple(sorted([tuple(float(x) for x in np.sort(p)) for p in partitions]))
            
            solution_counts[par_tup] = solution_counts.get(par_tup, 0) + 1


    if verbose:
        print(f"% of steps with valid states: {valid_count / max_iter}")
        print(f"Raw count: {valid_count}")
        print(f"Unique solutions: {len(solution_counts)}")


    return valid_count, solution_counts


