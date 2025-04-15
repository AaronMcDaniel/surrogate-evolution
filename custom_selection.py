import numpy as np
from deap import tools
import math
from scipy.spatial import distance
import random


def dbea_selection(population, k, n_objectives, ref_dirs=None):
    """
    DBEA (Diversity-Based Evolutionary Algorithm) selection for DEAP.
    
    Parameters:
    ----------
    population : list
        List of individuals
    k : int
        Number of individuals to select
    n_objectives : int
        Number of objectives
    ref_dirs : numpy.ndarray, optional
        Reference direction set. If None, it will be generated.
        
    Returns:
    -------
    list
        Selected individuals
    """
    # If no reference directions provided, generate them
    if ref_dirs is None:
        ref_dirs = generate_reference_directions(n_objectives)
    
    # Extract objective values from population
    objectives = np.array([ind.fitness.values for ind in population])
    
    # Normalize objectives
    normalized_objectives = normalize_objectives(objectives)
    
    # Partition population into subspaces
    subspaces = partition_population(normalized_objectives, ref_dirs)
    
    # Apply diversity-first sorting
    fronts = diversity_first_sort(population, subspaces, normalized_objectives, ref_dirs)
    
    # Select individuals based on fronts
    selected = []
    front_idx = 0
    
    # Add complete fronts as long as possible
    while len(selected) + len(fronts[front_idx]) <= k:
        selected.extend(fronts[front_idx])
        front_idx += 1
        if front_idx == len(fronts):
            break
    
    # If we need to select individuals from the last front
    if len(selected) < k and front_idx < len(fronts):
        # Sort the last front by perpendicular distance to reference direction
        last_front = fronts[front_idx]
        distances = []
        
        for i, ind in enumerate(last_front):
            ind_obj = normalized_objectives[population.index(ind)]
            ref_dir_idx = subspaces[population.index(ind)]
            ref_dir = ref_dirs[ref_dir_idx]
            
            # Calculate perpendicular distance
            proj = np.dot(ind_obj, ref_dir) / np.dot(ref_dir, ref_dir) * ref_dir
            perp_dist = np.linalg.norm(ind_obj - proj)
            distances.append((i, perp_dist))
        
        # Sort by perpendicular distance (ascending)
        distances.sort(key=lambda x: x[1])
        
        # Add individuals from the last front based on distance
        for i, _ in distances[:k - len(selected)]:
            selected.append(last_front[i])
    
    return selected


def generate_reference_directions(n_objectives, n_partitions=12):
    """
    Generate reference directions using Das and Dennis's approach for small number of objectives
    or two-layered approach for many objectives.
    
    Parameters:
    ----------
    n_objectives : int
        Number of objectives
    n_partitions : int
        Number of partitions
        
    Returns:
    -------
    numpy.ndarray
        Reference directions
    """
    if n_objectives <= 5:
        # Use Das and Dennis's systematic approach
        return das_dennis_reference_directions(n_objectives, n_partitions)
    else:
        # Use two-layered approach for many objectives
        return two_layered_reference_directions(n_objectives, n_partitions)


def das_dennis_reference_directions(n_objectives, n_partitions):
    """
    Generate reference directions using Das and Dennis's systematic approach.
    
    Parameters:
    ----------
    n_objectives : int
        Number of objectives
    n_partitions : int
        Number of partitions
        
    Returns:
    -------
    numpy.ndarray
        Reference directions
    """
    def generate_recursive(ref_dirs, ref_dir, left, total, index):
        if index == n_objectives - 1:
            ref_dir[index] = left / total
            ref_dirs.append(ref_dir.copy())
        else:
            for i in range(left + 1):
                ref_dir[index] = i / total
                generate_recursive(ref_dirs, ref_dir, left - i, total, index + 1)
    
    ref_dirs = []
    ref_dir = np.zeros(n_objectives)
    generate_recursive(ref_dirs, ref_dir, n_partitions, n_partitions, 0)
    return np.array(ref_dirs)


def two_layered_reference_directions(n_objectives, n_partitions):
    """
    Generate reference directions using two-layered approach for many objectives.
    
    Parameters:
    ----------
    n_objectives : int
        Number of objectives
    n_partitions : int
        Number of partitions
        
    Returns:
    -------
    numpy.ndarray
        Reference directions
    """
    # Inner layer with divisions = n_partitions
    inner_dirs = das_dennis_reference_directions(n_objectives, n_partitions)
    
    # Outer layer with divisions = 2
    outer_dirs = das_dennis_reference_directions(n_objectives, 2)
    
    # Scale the outer layer
    outer_dirs = outer_dirs * 0.5
    
    # Combine both layers
    ref_dirs = np.vstack((inner_dirs, outer_dirs))
    
    return ref_dirs


def normalize_objectives(objectives):
    """
    Normalize objectives using extreme points and intercepts.
    
    Parameters:
    ----------
    objectives : numpy.ndarray
        Objective values of all individuals
        
    Returns:
    -------
    numpy.ndarray
        Normalized objective values
    """
    n_points, n_objectives = objectives.shape
    
    # Find extreme points (maximum value for each objective)
    extreme_points = np.zeros((n_objectives, n_objectives))
    for i in range(n_objectives):
        # Find the solution with the best value for objective i
        weight = np.zeros(n_objectives)
        weight[i] = 1
        
        # Use the achievement scalarizing function (weighted Tchebycheff)
        asf = np.max(objectives / weight, axis=1)
        extreme_points[i] = objectives[np.argmin(asf)]
    
    # Compute intercepts
    try:
        # Solve linear system to find intercepts
        b = np.ones(n_objectives)
        intercepts = np.linalg.solve(extreme_points, b)
        intercepts = 1 / intercepts
    except np.linalg.LinAlgError:
        # If the extreme points are not well-defined, use maximum values
        intercepts = np.max(objectives, axis=0)
    
    # Find the minimum value (utopia point)
    ideal_point = np.min(objectives, axis=0)
    
    # Normalize
    normalized_objectives = (objectives - ideal_point) / (intercepts - ideal_point)
    
    return normalized_objectives


def partition_population(normalized_objectives, ref_dirs):
    """
    Partition population into subspaces based on reference directions.
    
    Parameters:
    ----------
    normalized_objectives : numpy.ndarray
        Normalized objective values
    ref_dirs : numpy.ndarray
        Reference directions
        
    Returns:
    -------
    list
        Index of subspace (reference direction) for each individual
    """
    subspaces = []
    
    for ind_obj in normalized_objectives:
        # Calculate cosine similarity (or acute angle) with each reference direction
        cos_theta = np.dot(ind_obj, ref_dirs.T) / (np.linalg.norm(ind_obj) * np.linalg.norm(ref_dirs, axis=1))
        
        # Assign to the reference direction with minimum angle (maximum cosine)
        subspace_idx = np.argmax(cos_theta)
        subspaces.append(subspace_idx)
    
    return subspaces


def diversity_first_sort(population, subspaces, normalized_objectives, ref_dirs):
    """
    Apply diversity-first sorting to assign individuals to fronts.
    
    Parameters:
    ----------
    population : list
        List of individuals
    subspaces : list
        Index of subspace for each individual
    normalized_objectives : numpy.ndarray
        Normalized objective values
    ref_dirs : numpy.ndarray
        Reference directions
        
    Returns:
    -------
    list of lists
        Fronts containing individuals
    """
    n_ref_dirs = ref_dirs.shape[0]
    
    # Create a list of lists to store individuals in each subspace
    subspace_individuals = [[] for _ in range(n_ref_dirs)]
    
    # Assign individuals to their subspaces
    for i, ind in enumerate(population):
        subspace_idx = subspaces[i]
        subspace_individuals[subspace_idx].append((i, ind))
    
    # Apply nondominated sorting in each subspace
    subspace_ranks = []
    for subspace_idx in range(n_ref_dirs):
        if not subspace_individuals[subspace_idx]:
            continue
        
        individuals = [ind for _, ind in subspace_individuals[subspace_idx]]
        indices = [idx for idx, _ in subspace_individuals[subspace_idx]]
        
        # Perform nondominated sorting
        pareto_fronts = tools.sortNondominated(individuals, len(individuals))
        
        # Assign ranks
        for rank, front in enumerate(pareto_fronts):
            for ind in front:
                ind_idx = individuals.index(ind)
                original_idx = indices[ind_idx]
                subspace_ranks.append((original_idx, rank, subspace_idx))
    
    # Sort by rank
    subspace_ranks.sort(key=lambda x: x[1])
    
    # Apply diversity-first sorting
    fronts = []
    front = []
    current_rank = 0
    
    visited_subspaces = set()
    remaining_individuals = set(range(len(population)))
    
    while remaining_individuals:
        front = []
        visited_subspaces = set()
        
        # For each rank, select one individual from each subspace
        for original_idx, rank, subspace_idx in subspace_ranks:
            if rank == current_rank and subspace_idx not in visited_subspaces and original_idx in remaining_individuals:
                front.append(population[original_idx])
                visited_subspaces.add(subspace_idx)
                remaining_individuals.remove(original_idx)
        
        # If no individual was selected with the current rank, move to the next rank
        if not front:
            current_rank += 1
            continue
        
        fronts.append(front)
    
    return fronts


def compute_perpendicular_distance(point, ref_dir):
    """
    Compute perpendicular distance from a point to a reference direction.
    
    Parameters:
    ----------
    point : numpy.ndarray
        Point in objective space
    ref_dir : numpy.ndarray
        Reference direction
        
    Returns:
    -------
    float
        Perpendicular distance
    """
    # Normalize the reference direction
    ref_dir_norm = ref_dir / np.linalg.norm(ref_dir)
    
    # Project the point onto the reference direction
    proj = np.dot(point, ref_dir_norm) * ref_dir_norm
    
    # Compute perpendicular distance
    perp_dist = np.linalg.norm(point - proj)
    
    return perp_dist

def lexicase_selection(population, n_objectives, k=1, epsilon=0.05):
    random_order = list(range(n_objectives))
    selected = []
    remaining_pop = population.copy()
    while len(selected) < k and remaining_pop:
        candidates = remaining_pop.copy()
        random.shuffle(random_order)
        for obj_idx in random_order:
            if len(candidates) <= 1: break
            obj_values = [ind.fitness.wvalues[obj_idx] for ind in candidates]
            threshold = np.percentile(np.array(obj_values), 1-epsilon)
            candidates = [ind for ind in candidates if ind.fitness.wvalues[obj_idx] >= threshold]
        if not candidates:
            break
        chosen = random.choice(candidates)
        selected.append(chosen)
        remaining_pop.remove(chosen)

    print("Lexicase individuals:", len(selected), "| random sampling for ", k - len(selected), flush=True)
    if len(selected) < k:
        selected.extend(random.sample(population, k - len(selected)))
    return selected

    