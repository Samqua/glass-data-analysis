# glass-data-analysis
Data-scientific attempts to solve the mobility problem of atoms near the glass transition

This was an attempt at G. Biroli's data challenge  (https://challengedata.ens.fr/en/challenge/46/solve_a_major_physics_problem_by_finding_how_to_predict_the_dynamics_of_glass-forming_liquids.html) of predicting the long-term mobilities of atoms in the liquid state just above the glass transition.

The data consists of 1877 observations of 3000 positions (under periodic boundary conditions) and 3000 velocities (for a total of 6000 features/variables). For each observation one hopes to make a set of 800 binary predictions about the long-term mobilities of a subset of the 1000 atoms. It is therefore a multi-label classification problem over an extremely sparse data set.

I tried MLP, DNN classifiers in TensorFlow (with many combinations of hyperparameters), k-NN, ensemble methods, and kernel methods, together with all sorts of dimensionality reduction techniques such as principal component analysis, linear discriminant analysis, and even unsupervised methods like self-organizing maps (and ensembles thereof). Basically nothing worked. The input space is high dimensional with a high intrinsic dimension (i.e. the data can only be adequately explained with a large number of features because the physical system is so complex). If the data set were larger, I might have had a chance.
