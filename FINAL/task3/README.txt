The PCA and t-SNE plots show a clear contrast in how well the two dimensionality-reduction methods separate the face classes. 
In the PCA visualization, the clusters are loosely formed, overlapping substantially, and lacking distinct boundaries; 
this is expected because PCA is a linear technique that projects data onto directions of maximum global variance, 
which often fails to separate classes when the underlying structure is nonlinear. In comparison, 
the t-SNE visualization produces significantly tighter and more distinct clusters, 
with clear grouping of samples from the same identity and visibly larger gaps between different identities. 

This happens because t-SNE preserves local neighborhood structure and emphasizes small-scale distances, 
making it far better suited for revealing high-dimensional clustering patterns in face embeddings. 
Overall, t-SNE provides a much clearer and more interpretable separation of classes, 
showing identity-specific groupings that PCA is unable to capture effectively.
