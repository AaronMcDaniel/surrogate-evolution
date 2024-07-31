# Precognition Optimization
Predicting algorithm performance for optimization

## Description
Frame-level aerial object detection is critical for aerial object localization and rapid sense-and-avoid (SAA) maneuvers. Neural network models used for object detection typically have 100+ million parameters, resulting in a train and evaluation that can take over 1,200 hours for a single architecture. We aim to build a scalable, efficient AutoML pipeline using an evolutionary Neural Architecture Search (NAS) algorithm to automate the search and design of state-of-the-art neural networks for a variety of image data applications.

Our NAS consists of a surrogate evaluator to predict model performance and reduce the number of individuals having to be trained an evaluated to get high-performing solutions.

![Approach](https://wiki.gtri.gatech.edu/download/attachments/330374279/image-2024-7-17_11-43-40.png?version=1&modificationDate=1721231020000&api=v2)

## Usage
Usage instructions can be found [here](https://wiki.gtri.gatech.edu/display/EMADE/Evolution+Instructions?src=contextnavpagetreemode)

### Generating Pareto Front/Hypervolume Graphs
 - Go to the main function in analysis/pareto_front.py
 - Create pandas dataframes of every out.csv you want to graph
 - Create a dataframe entry in the dataframes list with graphing information:
   - The actual pandas dataframe
   - A name that represents the evolution
   - Colors to indicate different parts
     - Overall pareto optimal points
     - recalculated pareto optimal points
     - Overall pareto optimal points from a previous generation
     - recalculated pareto optimal points from a previous generation
   - A marker to use when plotting points
 - Create a pandas dataframe for every benchmark you want graphed
 - Create a benchmark entry in the benchmarks list
   - The pandas dataframe
   - A name
   - A color
   - A marker
 - Set your effective max values to graph by editing the bounds_limits
   - Alternates min then max for every objective, in order defined in objectives list
 - Set a bounds_margin to expand the graph by a certain percent past the effective maximums (0.1 = 10% expansion in every direction)

## Support
Additional background and documentation can be found [here](https://wiki.gtri.gatech.edu/display/EMADE/Summer+GRIP+2024+-+Precognition+Optimization?src=contextnavpagetreemode)

## Roadmap
 - Two stage surrogate evaluator
 - Different surrogate types (like a transformer model)
 - Different genome and surrogate encoding strategies
 - Variable length objective lists for graphing pareto fronts
 - Exploring KAN strategies (grid extension, pruning)
 - Seeding deap so that it generates individuals in the same order

## Authors and acknowledgment
Contributors:
- Aaron McDaniel
- Ethan Harpster
- Tristan Thakur
- Vineet Kulkarni 

## Project status
Ongoing
