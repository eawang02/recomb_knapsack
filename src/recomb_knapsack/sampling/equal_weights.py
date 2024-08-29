
import numpy as np
from tqdm import tqdm


def equal_weights_step(partitions, i, j, C):
    """
    Takes the list of partitions and two indices i and j, and attempt to recombine them into two (closer to) valid partitions.
    Randomly shuffles the two partitions, then splits them into left and right partitions based on the split method.
    Using equal weights, this method splits as soon as the left partition exceeds half the total weight of the districts.

    Args:
        partitions (List[List[float]]): List of partitions to recombine
        i (int): Index of the first partition to recombine
        j (int): Index of the second partition to recombine
        C (Tuple[float, float]): Tuple of the lower and upper bounds for the partition weights

    Returns:
        Tuple[bool, bool]: Tuple of booleans indicating whether the partitions i and j are valid after recombination
    """
    new_partition = np.random.permutation(np.concatenate([partitions[i], partitions[j]]))
    cumsums = np.cumsum(new_partition)
    
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
    Sample solutions to the knapsack problem, using the equal weights recombination method.

    Args:
        X (List[float]): Sampled data to knapsack with
        k (int): Number of districts/partitions
        tol (float): Tolerance (as a value) for partition sizes, each partition is allowed to be C ± tol
        max_iter (int): Maximum number of iterations for MCMC sampling before stopping
        verbose (bool): Whether to print out progress updates

    Returns:
        Tuple[int, Dict[Tuple[Tuple[float], ...], int]]: Tuple of the number of valid states and a dictionary that maps unique solutions to frequencies
    """

    X = np.random.permutation(X)
    partitions = np.array_split(X, k)
    C = ((sum(X) / k) - tol, (sum(X) / k) + tol)

    if verbose:
        print("Attempting Random Recomb method...")
        print(f"Partitioning into {k} districts in the range [{C[0]}, {C[1]}].")

    # Run burnin steps
    for i in range(10000):
        i, j = np.random.choice(k, size=2, replace=False)
        equal_weights_step(partitions, i, j, C)
    
    solution_counts = {}
    valid_count = 0
    invalid_set = {i for i in range(k) if not (C[0] <= sum(partitions[i]) <= C[1])}
    valid_set = set(range(k)) - invalid_set

    for i in tqdm(range(max_iter), disable=not verbose):
        i, j = np.random.choice(k, size=2, replace=False)
        invalid_set.add(i)
        invalid_set.add(j)

        # Perform recomb step
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


