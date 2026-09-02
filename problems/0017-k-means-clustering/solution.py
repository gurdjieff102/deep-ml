import numpy as np
def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
	for _ in range(max_iterations):
		points = np.array(points, dtype=float)
		centroids = np.array(initial_centroids, dtype=float)
		distances = []

		for point in points:
			point_distances = []
			for initial_centroid in initial_centroids:
				distance = np.linalg.norm(point - initial_centroid)
				point_distances.append(distance)
			distances.append(point_distances)
		distances = np.array(distances)

		labels = np.argmin(distances, axis=1)
		new_centroids = []
		for i in range(k):
			cluster_points = points[labels == i]
			if len(cluster_points) > 0:
				new_centroids.append(cluster_points.mean(axis=0))
			else:
				new_centroids = centroids[i]
			# new_centroids.append(new_centroids)
		centroids = np.array(new_centroids)
	return [tuple(np.round(centroid, 4)) for centroid in centroids]