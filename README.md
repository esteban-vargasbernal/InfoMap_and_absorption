# An adaptation of InfoMap to absorbing random walks using absorption-scaled graphs.

This repository contains the Python code that generates numarical results of the manuscript "An adaptation of InfoMap to absorbing random walks using absorption-scaled graphs".

Using the file "Map_absorption.py", we computes the standard map and the map for absorbing random walks (defined in Section 4) for all the possible partitions of a three-node network. With this code, we generate Figures 3 and 5.

Using the file "Toy_examples.py", we execute Algorithm 1 on two toy examples of Section 5. With this code, we generate Figures 7 and 9.

Using the file "Disease_quantities.py", we execute Algorithm 3 to generate the mean epidemiological quantities of Figures 10 a--c in Section 6.2. The input network of Algorithm 1 is specified in the files "NodesG.csv", "EdgesG.csv", "BetNodes.csv". The absorption configurations in the initial stage, peak-duration stage and final stage are specified in the files "InitialStage.csv", "PeakStage.csv" and "FinalStage.csv", respectively.

Using the file "Map_stages.py", we compute a map for absorbing random walks for the network of Section 6 and different parameter configurations (as defined in Algorithm 3). With this code, we  generate Figure 10 d. The absorption-configuration files are in the folder "deltas".

Using The file "Community_stages.py", we execute Algorithm 1 on the network of Section 6.2 to generate Figure 11.
