# Day 11: Unsupervised Learning
-----------------------------------------------
## Date: 08/01/2024
## Topic:
	-Clustring Algorithms
	-K-Means
	-Agglomerative
	-Divisive
	-DBSCAN


### Clustering :
-------------
	-unsupervised learning method
	-used to draw references from datasets consisting of input data without labeled responses.
	-used to find meaningful structures, expalnatory process, generative features, and grouping inherent elements in a set of examples.
	
### Cluster:
--------
	-a collection of data objects.
	-similar to one another within the same cluster.
	-dissimilar to the objects in the cluster.
	
### Cluster analysis:
-----------------
	-a process of finding similarities between data according to the characteristics found in the data and grouping similar data objects into clusters.
	
### Applications in Clustering:
----------------------------
	-Standalone tool: to get insight into data distribution.
	-Pattern analysis/ Pattern Recognition
	-Spatioal Data Analysis
	-Imge Processing

### Good cluster:
-------------
	-A good clustering method will generate high quality clusters with
		-Inter-class similarity
		-Intra-class similarity
	-quality of a cluster results depends on both the similarity measures used by the method and it's implementation.
	-quality of a clustering method is also measured by itsability to discover the hidden pattens in the dataset.
	

### Clustering Measures:
---------------------
	1. Data Matrix
	2. Distance Matrix
		-Jaccard similarity (Binary values)
		-Other distance measues:
			1. Minkowski distance
			2. Manhattan distance
			3. Euclidean distance
			4. Weighted distance
			
### Clustering Techniques:
----------------------
	1. Partitioning Clustering
	2. Hierarchical Clustering
	3. Density based Clustering
	4. Graph method Clustering
	5. Model based Clustering
	
### 1. Partitioning Clustering :
-----------------------------
	-partition the object into k-clusters and each partition forms one cluster.
	-used to optimize an objective criteria similarity function such as when the distance is a major parameter.
	-Eg: K-Means, K-Medoid
	
### 2. Hierarchical Clustering :
-----------------------------
	-Clusters are formed a tree-like structure based on the hierarchy.
	-2 types of tree constructions:
		-Agglomerative ( Bottom-up approach)
		-Divisive ( Top-Down approach)
		
### 3. Density based Clustering :
-----------------------------
	-method where cluster are in dense region having some similarities and differences from the lower dense region of the space.
	-methods works with good accuracy and the ability to merge 2 clusters.
	-Eg: DB-SCAN algorithm.
	
### 4. Graph method Clustering :
-------------------------------
	-data space is formulated into a finite number of cells that form a grid like structure
	-Clustering operations can be done on these grids.
	-Eg: STING algorithm


### 5. Model based Clustering :
---------------------------
	- a model is hypothesized for each of the cluster and tries to find the best fit of thet model.
	
### K-Means Algorithm:
------------------
	-process of partioning method that divides a dataset into 'k' distint clusters based on similarity aiming to minimize the variance within each cluster.
	-Problem of Kmeans: 
		-K-Means is very sensitive to outliers.
	-Solution: 
		-K-Medoid (work on the principle of median values)

  
