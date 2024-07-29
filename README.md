# Precognition Optimization
Predicting algorithm performance for optimization

## Description
Frame-level aerial object detection is critical for aerial object localization and rapid sense-and-avoid (SAA) maneuvers. Neural network models used for object detection typically have 100+ million parameters, resulting in a train and evaluation that can take over 1,200 hours for a single architecture. We aim to build a scalable, efficient AutoML pipeline using an evolutionary Neural Architecture Search (NAS) algorithm to automate the search and design of state-of-the-art neural networks for a variety of image data applications.

Our NAS consists of a surrogate evaluator to predict model performance and reduce the number of individuals having to be trained an evaluated to get high-performing solutions.

![Approach](https://wiki.gtri.gatech.edu/download/attachments/330374279/image-2024-7-17_11-43-40.png?version=1&modificationDate=1721231020000&api=v2)

## Visuals
Some poster images can go here

## Usage
Usage instructions can be found [here](https://wiki.gtri.gatech.edu/display/EMADE/Evolution+Instructions?src=contextnavpagetreemode)

## Support
Additional background and documentation can be found [here](https://wiki.gtri.gatech.edu/display/EMADE/Summer+GRIP+2024+-+Precognition+Optimization?src=contextnavpagetreemode)

## Roadmap
 - Two stage surrogate evaluator
 - Different surrogate types (like a transformer model)
 - Different genome and surrogate encoding strategies

## Authors and acknowledgment
Contributors:
- Aaron McDaniel
- Ethan Harpster
- Tristan Thakur
- Vineet Kulkarni 

## Project status
Ongoing
