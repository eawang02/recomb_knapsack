
import numpy as np
import scipy.stats as stats
from tqdm import tqdm


def first_valid_step(partitions, i, j, C):
    """
    Takes the list of partitions and two indices i and j, and attempt to recombine them into two (closer to) valid partitions.
    Randomly shuffles the two partitions, then splits them into left and right partitions based on the split method.
    Using first valid, this method splits as soon as the left partition becomes valid.
    If the left partition never becomes valid, the partitions are not updated.

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
    
    valid_split_index = np.where((cumsums >= C[0]) & (cumsums <= C[1]))[0]
    if valid_split_index.size > 0:
        idx = valid_split_index[0] + 1
        partitions[i] = new_partition[:idx]
        partitions[j] = new_partition[idx:]
        i_valid = True
        j_valid = C[0] <= sum(partitions[j]) <= C[1]
        return (i_valid, j_valid)

    return (False, False) 



def first_valid_recomb(X=None, k=10, tol=0.01, max_iter=10000, verbose=False):
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
        first_valid_step(partitions, i, j, C)
    
    solution_counts = {}
    valid_count = 0
    invalid_set = {i for i in range(k) if not (C[0] <= sum(partitions[i]) <= C[1])}
    valid_set = set(range(k)) - invalid_set

    for i in tqdm(range(max_iter), disable=not verbose):
        i, j = np.random.choice(k, size=2, replace=False)
        invalid_set.add(i)
        invalid_set.add(j)

        # Perform recomb step
        i_bool, j_bool = first_valid_step(partitions, i, j, C)
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


