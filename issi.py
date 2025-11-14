"""
Island-based Simulated Surrogate Injection (ISSI) Algorithm

This module implements the ISSI algorithm for neural architecture search,
designed to manage and mitigate low-fidelity evaluator bias through 
parallel island evolution with strategic high-fidelity consolidation phases.
"""

import copy
import random
import numpy as np
from deap import creator, base, tools


class ISSIAlgorithm:
    """
    Island-based Simulated Surrogate Injection implementation.
    
    The ISSI algorithm operates on the philosophy of managing bias rather than trusting
    low-fidelity evaluations. It uses multiple isolated parallel searches (islands) 
    to cultivate architectural diversity, with expensive high-fidelity evaluation
    as periodic ground-truth tournaments.
    """
    
    def __init__(self, pipeline):
        """
        Initialize ISSI with reference to pipeline for accessing methods and config.
        
        Args:
            pipeline: Reference to the main Pipeline object
        """
        self.pipeline = pipeline
        
        # ISSI Parameters (matching design document)
        self.N_total = 2000      # Total population size in LF phase
        self.N_hf = 30          # Number of individuals for HF evaluation
        self.N_islands = 5       # Number of isolated sub-populations
        self.N_island_pop = self.N_total // self.N_islands      # Population per island (N_total / N_islands)
        self.G_lf_exploitation = 20   # LF generations for exploitation island
        self.G_lf_exploration = [5, 20, 40, 100]  # LF generations for exploration islands
        self.E_island = 4        # Elites preserved per island per generation
        
        # Configurable parent selection percentages
        self.exploitation_parent_pct = 0.10  # 10% of island pop for high selection pressure
        self.exploration_parent_pct = 0.20   # 20% of island pop for lower selection pressure
    
    def run_cycle(self, hf_evaluated_individuals: list):
        """
        Execute one complete ISSI cycle.
        
        Args:
            hf_evaluated_individuals: Dict of HF-evaluated individuals (30 + 4 elites = 34 total)
        
        Returns:
            Dict of 30 selected individuals for next HF evaluation cycle
        """
        print('Beginning Island-based Simulated Surrogate Injection (ISSI)')
        
        # Phase 1: Consolidation & Re-seeding
        print('Phase 1: Consolidation & Re-seeding')
        island_seeds, exploitation_island_id = self._consolidation_phase(hf_evaluated_individuals)
        
        # Phase 2: Exploration Phase - Parallel Island Evolution
        print('Phase 2: Multi-Island Exploration Phase')
        final_islands = self._exploration_phase(island_seeds, exploitation_island_id)
        
        # Phase 3: Final Selection / Island Teardown
        print('Phase 3: Final Selection and Island Teardown')
        final_candidates = self._final_selection_phase(final_islands, exploitation_island_id)
        
        print(f'ISSI completed: {len(final_candidates)} candidates selected for next HF evaluation')
        return final_candidates
    
    def _consolidation_phase(self, hf_evaluated_candidates: list) -> tuple[list[list], int]:
        """
        Phase 1: Strategic island seeding using best performers.
        
        Args:
            hf_evaluated_candidates: Dict of individuals already evaluated with HF by main pipeline
        
        Returns:
            tuple: (island_seeds, exploitation_island_id)
        """
        # Input individuals are already HF-evaluated by the main pipeline (30 + 4 elites = 34)
        candidate_list = hf_evaluated_candidates
        print(f'Processing {len(candidate_list)} HF-evaluated individuals (already evaluated by main pipeline)')
        
        # Step 1: Select seeds for exploitation island - top 4 using SPEA2
        exploitation_seeds = tools.selSPEA2(candidate_list, min(4, len(candidate_list)))
        print(f'Selected {len(exploitation_seeds)} seeds for exploitation island using SPEA2')
        
        # Step 2: Select candidates for exploration islands - top 10 using NSGA-II
        exploration_candidate_pool = tools.selNSGA2(candidate_list, min(10, len(candidate_list)))
        print(f'Selected {len(exploration_candidate_pool)} candidates for exploration island seeding')
        
        # Step 3: Create island seeds
        island_seeds = []
        exploitation_island_id = 0  # First island is always exploitation
        
        # Exploitation island gets the 4 best models
        island_seeds.append(exploitation_seeds)
        
        # Each exploration island gets 4 randomly sampled from the top 10
        for i in range(self.N_islands - 1):  # 4 exploration islands
            if len(exploration_candidate_pool) >= 4:
                exploration_seeds = random.sample(exploration_candidate_pool, 4)
            else:
                # If less than 4 available, use all and pad with random candidates
                exploration_seeds = list(exploration_candidate_pool)
                while len(exploration_seeds) < 4 and candidate_list:
                    additional = random.choice(candidate_list)
                    if additional not in exploration_seeds:
                        exploration_seeds.append(additional)
            
            island_seeds.append(exploration_seeds)
            print(f'Exploration island {i+1} seeded with {len(exploration_seeds)} individuals')
        
        print(f'Island 0 designated as exploitation island')
        
        return island_seeds, exploitation_island_id
    
    def _exploration_phase(self, island_seeds: list[list], exploitation_island_id: int) -> list[dict]:
        """
        Phase 2: Parallel evolution across multiple islands with different strategies.
        
        Returns:
            list: Final populations for each island
        """
        final_islands = []
        
        for island_id in range(self.N_islands):
            print(f'Evolving Island {island_id}...')
            
            # Island-specific parameters
            if island_id == exploitation_island_id:
                # Exploitation Island
                generations = self.G_lf_exploitation
                parent_size = int(self.N_island_pop * self.exploitation_parent_pct)
                mutation_prob = 0.60
                crossover_prob = 0.25
                print(f'  Exploitation Island: {generations} gens, {parent_size} parents ({self.exploitation_parent_pct*100:.0f}% of pop)')
            else:
                # Exploration Islands  
                gen_idx = (island_id - (1 if island_id > exploitation_island_id else 0)) % len(self.G_lf_exploration)
                generations = self.G_lf_exploration[gen_idx]
                parent_size = int(self.N_island_pop * self.exploration_parent_pct)
                mutation_prob = 0.85
                crossover_prob = 0.15
                print(f'  Exploration Island: {generations} gens, {parent_size} parents ({self.exploration_parent_pct*100:.0f}% of pop)')
            
            # Island Buildup - populate to full size with mutated seeds
            island_seeds_for_this = island_seeds[island_id] if island_id < len(island_seeds) else []
            island_pop = self._build_island_population(island_seeds_for_this, self.N_island_pop)
            
            # Parallel Evolution within this island
            island_pop = self._evolve_island(
                island_pop, generations, parent_size, self.E_island,
                mutation_prob, crossover_prob, island_id
            )
            
            final_islands.append(island_pop)
            print(f'Island {island_id} evolution completed with {len(island_pop)} individuals')
        
        return final_islands
    
    def _final_selection_phase(self, final_islands, exploitation_island_id):
        """
        Phase 3: Quota and wildcard selection to prepare candidates for next cycle.
        
        Returns:
            dict: Final selected candidates
        """
        # Quota Selection - champions from each island
        quota_champions = []
        
        # From exploitation island: top 8 champions
        exploitation_island = final_islands[exploitation_island_id]
        exploitation_champions = tools.selNSGA2(list(exploitation_island.values()), 
                                               min(8, len(exploitation_island)))
        quota_champions.extend(exploitation_champions)
        print(f'Selected {len(exploitation_champions)} champions from exploitation island')
        
        # From each exploration island: top 4 champions each
        exploration_count = 0
        for island_id, island_pop in enumerate(final_islands):
            if island_id != exploitation_island_id:
                island_champions = tools.selNSGA2(list(island_pop.values()), 
                                                min(4, len(island_pop)))
                quota_champions.extend(island_champions)
                exploration_count += len(island_champions)
        
        print(f'Selected {exploration_count} champions from exploration islands')
        
        # Wildcard Selection - global best performers
        all_individuals = []
        for island_pop in final_islands:
            all_individuals.extend(list(island_pop.values()))
        
        # Remove quota champions from global pool to avoid duplicates
        quota_hashes = {self.pipeline._Pipeline__get_hash(str(champ)) for champ in quota_champions}
        wildcard_pool = [ind for ind in all_individuals 
                        if self.pipeline._Pipeline__get_hash(str(ind)) not in quota_hashes]
        
        wildcard_count = max(0, self.N_hf - len(quota_champions))
        wildcards = tools.selNSGA2(wildcard_pool, min(wildcard_count, len(wildcard_pool)))
        
        print(f'Selected {len(wildcards)} wildcard champions from global pool')
        
        # Combine quota and wildcard selections
        final_candidates = quota_champions + wildcards
        final_pop = {self.pipeline._Pipeline__get_hash(str(ind)): ind for ind in final_candidates}
        
        return final_pop
    

    
    def _build_island_population(self, seeds: list, target_size: int) -> dict:
        """
        Build island population by creating heavily mutated versions of seed architectures.
        
        Args:
            seeds: List of seed individuals for this island
            target_size: Target population size for the island
            
        Returns:
            dict: Island population with hash keys
        """
        if not seeds:
            # If no seeds, create random population
            pop = self.pipeline.toolbox.population(n=target_size)
            return {self.pipeline._Pipeline__get_hash(str(ind)): ind for ind in pop}
        
        island_pop = {}
        
        # Add original seeds
        for seed in seeds:
            hash_val = self.pipeline._Pipeline__get_hash(str(seed))
            island_pop[hash_val] = seed
        
        # Fill remaining slots with mutated versions
        while len(island_pop) < target_size:
            # Select random seed to mutate
            base_seed = random.choice(seeds)
            
            # Create mutated version of the seed
            mutated = copy.deepcopy(base_seed)
            
            # Apply single mutation
            try:
                mutants = self.pipeline.mutate(mutated)
                if mutants:
                    mutated = random.choice(mutants)
            except:
                continue
            
            hash_val = self.pipeline._Pipeline__get_hash(str(mutated))
            if hash_val not in island_pop:  # Avoid duplicates
                island_pop[hash_val] = mutated
            
            # Safety check to avoid infinite loop
            if len(island_pop) >= target_size * 2:  # If we've tried twice the target
                break
        
        # If still under target, fill with random individuals
        while len(island_pop) < target_size:
            random_ind = self.pipeline.toolbox.individual()
            hash_val = self.pipeline._Pipeline__get_hash(str(random_ind))
            if hash_val not in island_pop:
                island_pop[hash_val] = random_ind
        
        return island_pop
    
    def _evolve_island(self, island_pop: dict, generations: int, parent_size: int, elite_size: int, 
                      mutation_prob: float, crossover_prob: float, island_id: int):
        """
        Evolve a single island with specified parameters.
        
        Args:
            island_pop: Current island population
            generations: Number of generations to evolve
            parent_size: Number of parents to select
            elite_size: Number of elites to preserve
            mutation_prob: Mutation probability
            crossover_prob: Crossover probability
            island_id: Island identifier for logging
            
        Returns:
            dict: Evolved island population
        """
        print(f'    Evolving island {island_id} for {generations} generations')
        
        # Convert to list for DEAP operations
        current_pop = list(island_pop.values())
        
        # Store original toolbox settings to restore later
        original_mutations = {}
        original_crossovers = {}
        
        # Temporarily adjust mutation and crossover probabilities
        for mutation in self.pipeline.mutations.keys():
            original_mutations[mutation] = self.pipeline.mutations[mutation]
            self.pipeline.mutations[mutation] = mutation_prob
        
        for crossover in self.pipeline.crossovers.keys():
            original_crossovers[crossover] = self.pipeline.crossovers[crossover]
            self.pipeline.crossovers[crossover] = crossover_prob
            
        try:
            for gen in range(generations):
                # Set fitnesses using low-fidelity surrogate
                if self.pipeline.surrogate_enabled and self.pipeline.gen_count > 1:
                    _, current_pop = self.pipeline.surrogate.set_fitnesses(
                        self.pipeline.sub_surrogates, 
                        self.pipeline.cls_genome_scaler, 
                        self.pipeline.reg_genome_scaler, 
                        current_pop
                    )
                
                # Select parents using vanilla NSGA-II
                parents = tools.selNSGA2(current_pop, parent_size)
                
                # Preserve elites
                elites = tools.selNSGA2(current_pop, elite_size)
                
                # Create new population through crossover and mutation
                new_pop = list(elites)  # Start with elites
                
                # Fill rest through mating
                while len(new_pop) < len(current_pop):
                    if len(parents) >= 2:
                        parent1 = random.choice(parents)
                        parent2 = random.choice(parents)
                        
                        try:
                            offspring = self.pipeline.cross([parent1, parent2])
                            for child in offspring:
                                if len(new_pop) < len(current_pop):
                                    # Apply mutations
                                    mutants = self.pipeline.mutate(child)
                                    final_child = random.choice(mutants) if mutants else child
                                    new_pop.append(final_child)
                        except:
                            # If crossover fails, add a mutated parent
                            if len(new_pop) < len(current_pop):
                                try:
                                    mutants = self.pipeline.mutate(copy.deepcopy(parent1))
                                    final_child = random.choice(mutants) if mutants else parent1
                                    new_pop.append(final_child)
                                except:
                                    new_pop.append(copy.deepcopy(parent1))
                
                current_pop = new_pop[:len(island_pop)]  # Ensure consistent size
                
                if (gen + 1) % 10 == 0:  # Progress update every 10 generations
                    print(f'    Island {island_id}: Generation {gen + 1}/{generations}')
        
        finally:
            # Restore original probabilities
            for mutation in original_mutations:
                self.pipeline.mutations[mutation] = original_mutations[mutation]
            for crossover in original_crossovers:
                self.pipeline.crossovers[crossover] = original_crossovers[crossover]
        
        # Convert back to dictionary format
        final_pop = {self.pipeline._Pipeline__get_hash(str(ind)): ind for ind in current_pop}
        return final_pop
    
