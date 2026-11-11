import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

# DEBUG:
# SUGAR
# from cython_bbox import bbox_overlaps as bbox_ious
from .tracking_utils import kalman_filter

# ============================================================================
# IoU (Intersection over Union) Calculation
# ============================================================================

# DEBUG:
# SUGAR
def bbox_overlaps(
		bboxes1,
		bboxes2,
		mode='iou',
		eps=1e-6,
		use_legacy_coordinate=False
):
	"""
	Calculate IoU or IoF (Intersection over Foreground) between two sets of bboxes.

	Computes pairwise overlaps efficiently for all combinations of bboxes in
	bboxes1 and bboxes2. Supports two modes:
	  - 'iou': standard Intersection over Union (A ∩ B) / (A ∪ B)
	  - 'iof': Intersection over Foreground (A ∩ B) / A

	Args:
		bboxes1 (np.ndarray): Shape (n, 4), format [x1, y1, x2, y2] (top-left, bottom-right)
		bboxes2 (np.ndarray): Shape (k, 4), same format
		mode (str): 'iou' or 'iof'. Defaults to 'iou'.
		eps (float): Small epsilon to avoid division by zero. Defaults to 1e-6.
		use_legacy_coordinate (bool): If True, adds 1 to width/height for VOC-style
			coordinate offsets (x2 - x1 + 1). Defaults to False.

	Returns:
		np.ndarray: Shape (n, k) containing pairwise overlap scores in [0, 1].

	Note:
		If bboxes1 has more elements than bboxes2, they are swapped for efficiency.
	"""
	assert mode in ['iou', 'iof'], f"Mode must be 'iou' or 'iof', got {mode}"

	# Adjust coordinate metric based on legacy flag
	extra_length = 1.0 if use_legacy_coordinate else 0.0

	# Convert to float32 for numerical stability
	bboxes1 = bboxes1.astype(np.float32)
	bboxes2 = bboxes2.astype(np.float32)

	rows = bboxes1.shape[0]
	cols = bboxes2.shape[0]
	ious = np.zeros((rows, cols), dtype=np.float32)

	# Handle empty case
	if rows * cols == 0:
		return ious

	# Optimization: swap if bboxes1 is larger (reduces outer loop iterations)
	exchange = False
	if bboxes1.shape[0] > bboxes2.shape[0]:
		bboxes1, bboxes2 = bboxes2, bboxes1
		ious = np.zeros((cols, rows), dtype=np.float32)
		exchange = True

	# Pre-compute areas for all boxes
	area1 = (bboxes1[:, 2] - bboxes1[:, 0] + extra_length) * (
			bboxes1[:, 3] - bboxes1[:, 1] + extra_length)
	area2 = (bboxes2[:, 2] - bboxes2[:, 0] + extra_length) * (
			bboxes2[:, 3] - bboxes2[:, 1] + extra_length)

	# Compute pairwise overlaps
	for i in range(bboxes1.shape[0]):
		# Find intersection region
		x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
		y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
		x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
		y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])

		# Compute intersection area
		overlap = np.maximum(x_end - x_start + extra_length, 0) * np.maximum(
			y_end - y_start + extra_length, 0)

		# Compute union or foreground based on mode
		if mode == 'iou':
			union = area1[i] + area2 - overlap
		else:  # iof mode
			union = area1[i] if not exchange else area2

		# Avoid division by zero
		union = np.maximum(union, eps)
		ious[i, :] = overlap / union

	# Transpose back if arrays were swapped
	if exchange:
		ious = ious.T

	return ious

# ============================================================================
# Track-to-Detection Matching
# ============================================================================

def merge_matches(m1, m2, shape):
	"""
	Merge two sequential matching triplets using sparse matrices.

	Given two sets of matches (m1 and m2) that form a chain (A→B and B→C),
	compute the transitive closure to find direct matches A→C.

	Args:
		m1 (np.ndarray): Shape (n_match1, 2), matches between group O and P
		m2 (np.ndarray): Shape (n_match2, 2), matches between group P and Q
		shape (tuple): (O, P, Q) dimensions for the three groups

	Returns:
		tuple: (matches, unmatched_O, unmatched_Q)
			- matches: Direct matches between O and Q
			- unmatched_O: Indices in O not matched to Q
			- unmatched_Q: Indices in Q not matched to O
	"""
	O, P, Q = shape
	m1 = np.asarray(m1)
	m2 = np.asarray(m2)

	# Build sparse adjacency matrices: O→P and P→Q
	M1 = scipy.sparse.coo_matrix(
		(np.ones(len(m1)), (m1[:, 0], m1[:, 1])),
		shape=(O, P)
	)
	M2 = scipy.sparse.coo_matrix(
		(np.ones(len(m2)), (m2[:, 0], m2[:, 1])),
		shape=(P, Q)
	)

	# Matrix multiplication: (O→P) × (P→Q) = (O→Q)
	mask = M1 * M2
	match = mask.nonzero()
	match = list(zip(match[0], match[1]))

	# Find unmatched indices
	unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
	unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

	return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
	"""
	Filter assignment indices to keep only matches below cost threshold.

	Given a set of proposed matches (indices), filter by cost and return
	valid matches and unmatched indices.

	Args:
		cost_matrix (np.ndarray): Shape (n, m), precomputed cost matrix
		indices (np.ndarray): Shape (k, 2), proposed [track_idx, detection_idx] pairs
		thresh (float): Cost threshold; keep only pairs with cost ≤ thresh

	Returns:
		tuple: (matches, unmatched_a, unmatched_b)
	"""
	matched_cost = cost_matrix[tuple(zip(*indices))]
	matched_mask = (matched_cost <= thresh)

	matches = indices[matched_mask]
	unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
	unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

	return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
	"""
	Solve the linear assignment problem using the Hungarian algorithm (LAP).

	Finds the minimum-cost perfect matching between tracks and detections.
	Uses the lap.lapjv() function for efficient computation.

	Args:
		cost_matrix (np.ndarray): Shape (n_tracks, n_detections), cost of pairing
		thresh (float): Cost threshold; discard matches with cost > thresh

	Returns:
		tuple: (matches, unmatched_a, unmatched_b)
			- matches: Shape (k, 2), valid [track_idx, detection_idx] pairs
			- unmatched_a: Indices of unmatched tracks
			- unmatched_b: Indices of unmatched detections
	"""
	if cost_matrix.size == 0:
		return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
	matches, unmatched_a, unmatched_b = [], [], []
	cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
	for ix, mx in enumerate(x):
		if mx >= 0:
			matches.append([ix, mx])
	unmatched_a = np.where(x < 0)[0]
	unmatched_b = np.where(y < 0)[0]
	matches = np.asarray(matches)
	return matches, unmatched_a, unmatched_b


# ============================================================================
# Distance Metrics
# ============================================================================


def ious(atlbrs, btlbrs):
	"""
	Compute pairwise IoU between two sets of bounding boxes.

	Wraps the bbox_overlaps() function to compute IoU values in [0, 1].
	Input bboxes are converted to contiguous arrays for cache efficiency.

	Args:
		atlbrs: Bounding boxes set A, format [x1, y1, x2, y2]
		btlbrs: Bounding boxes set B, same format

	Returns:
		np.ndarray: Shape (len(atlbrs), len(btlbrs)), pairwise IoU values
	"""
	ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.complex128)
	if ious.size == 0:
		return ious

	ious = bbox_overlaps(
		np.ascontiguousarray(atlbrs, dtype=np.complex128),
		np.ascontiguousarray(btlbrs, dtype=np.complex128)
	)

	return ious


def iou_distance(atracks, btracks):
	"""
	Compute distance cost matrix based on IoU (1 - IoU).

	Converts STrack objects (or raw bboxes) to bounding box format,
	computes IoU, and returns cost = 1 - IoU (so higher IoU = lower cost).

	Args:
		atracks: List of STrack objects or bounding boxes [x1, y1, x2, y2]
		btracks: List of STrack objects or bounding boxes

	Returns:
		np.ndarray: Shape (len(atracks), len(btracks)), cost matrix from [0, 1]
			where 0 = perfect match (IoU=1) and 1 = no overlap (IoU=0)
	"""
	# Extract bounding boxes if inputs are STrack objects
	if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
		atlbrs = atracks
		btlbrs = btracks
	else:
		atlbrs = [track.tlbr for track in atracks]
		btlbrs = [track.tlbr for track in btracks]
	_ious = ious(atlbrs, btlbrs)
	cost_matrix = 1 - _ious

	return cost_matrix

def pixel_d(atlbr, btlbr):
	"""
	Compute mean Euclidean distance between 4 corners of two bounding boxes.

	Extracts the four corners (top-left, top-right, bottom-left, bottom-right)
	from each bbox and computes the mean L2 distance.

	Args:
		atlbr: Bbox A, format [x1, y1, x2, y2]
		btlbr: Bbox B, same format

	Returns:
		float: Mean distance between corresponding corners
	"""
	# Define corners of axis-aligned bounding box (AABB)
	aAABB = np.array([
		[atlbr[0], atlbr[1]],  # top-left
		[atlbr[0], atlbr[3]],  # bottom-left
		[atlbr[2], atlbr[1]],  # top-right
		[atlbr[2], atlbr[3]]   # bottom-right
	])
	bAABB = np.array([
		[btlbr[0], btlbr[1]],
		[btlbr[0], btlbr[3]],
		[btlbr[2], btlbr[1]],
		[btlbr[2], btlbr[3]]
	])

	# Compute Euclidean distance for each corner pair, then take mean
	return np.mean(np.sqrt(np.sum(np.square(bAABB - aAABB), axis=1)))

def pixel_distance(atracks, btracks):
	"""
	Compute distance cost matrix based on spatial pixel distance.

	Computes the mean Euclidean distance between bbox corners for all pairs.
	Small distance indicates spatial proximity (good match candidate).

	Args:
		atracks: List of STrack objects or bounding boxes
		btracks: List of STrack objects or bounding boxes

	Returns:
		np.ndarray: Shape (len(atracks), len(btracks)), pixel distance cost matrix
			where smaller values indicate closer spatial proximity
	"""
	# Extract bounding boxes if inputs are STrack objects
	if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or \
			(len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
		atlbrs = atracks
		btlbrs = btracks
	else:
		atlbrs = [track.tlbr for track in atracks]
		btlbrs = [track.tlbr for track in btracks]

	cost_matrix = np.zeros((len(atracks), len(btracks)), dtype=np.float32)
	if cost_matrix.size == 0:
		return cost_matrix

	# Use scipy.spatial.distance.cdist with custom metric (pixel_d)
	# Small distance means spatial proximity
	cost_matrix = np.abs(cdist(atlbrs, btlbrs, pixel_d))

	return cost_matrix

def embedding_distance(tracks, detections, metric='cosine'):
	"""
	Compute distance cost matrix based on appearance (embedding) features.

	Extracts learned feature embeddings from tracks and detections,
	computes pairwise distance using the specified metric (e.g., cosine).
	Small distance indicates visual similarity.

	Args:
		tracks: List of STrack objects with .smooth_feat attribute
		detections: List of detection objects with .curr_feat attribute
		metric (str): Distance metric for scipy.spatial.distance.cdist.
			Common options: 'cosine', 'euclidean', 'manhattan'. Defaults to 'cosine'.

	Returns:
		np.ndarray: Shape (len(tracks), len(detections)), embedding distance cost matrix
			where smaller values indicate visual similarity
	"""

	cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.complex128)
	if cost_matrix.size == 0:
		return cost_matrix
	det_features = np.asarray([track.curr_feat for track in detections], dtype=np.complex128)
	#for i, track in enumerate(tracks):
	#cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
	track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.complex128)
	# small distance means similar pairs.
	cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features by ceiling to 0
	return cost_matrix

# ============================================================================
# Kalman Filter Gating and Motion Fusion
# ============================================================================

def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
	"""
	Apply Kalman filter gating: set costs to ∞ for physically impossible matches.

	Uses the Kalman filter's predicted state (mean, covariance) to estimate
	the probability that a detection belongs to a track through the Mahalanobis
	distance. Assignments with distance exceeding a chi-squared threshold are
	marked as inf (impossible to match).

	This is used as a hard constraint before appearance-based matching.

	Args:
		kf (KalmanFilter): Kalman filter instance with gating_distance method
		cost_matrix (np.ndarray): Shape (n_tracks, n_detections), initial cost matrix
		tracks: List of STrack objects with .mean and .covariance
		detections: List of detection objects with .to_xyah() method
		only_position (bool): If True, only use position (x, y); else use (x, y, vx, vy).
			Affects gating threshold selection. Defaults to False.

	Returns:
		np.ndarray: Modified cost matrix with inf values for gated-out pairs

	Note:
		The cost matrix is modified in place.
	"""
	if cost_matrix.size == 0:
		return cost_matrix

	# Select gating threshold based on state dimensionality
	gating_dim = 2 if only_position else 4  # (x, y) vs (x, y, vx, vy)
	gating_threshold = kalman_filter.chi2inv95[gating_dim]

	# Convert detections to measurement format (xyah: center x, center y, aspect, height)
	measurements = np.asarray([det.to_xyah() for det in detections])

	# Apply gating: compute Mahalanobis distance and reject unlikely associations
	for row, track in enumerate(tracks):
		gating_distance = kf.gating_distance(
			track.mean, track.covariance, measurements, only_position)
		# Mark impossible matches with inf cost
		cost_matrix[row, gating_distance > gating_threshold] = np.inf

	return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, logger=None, only_position=False, lambda_=0.98):
	"""
	Fuse motion information into the cost matrix using Kalman filter gating.

	Combines appearance/spatial cost with motion cost via weighted averaging:
	  fused_cost = λ * appearance_cost + (1 - λ) * motion_cost

	This balances visual and motion cues. Tracks/detections with high motion
	disagreement are penalized even if appearance matches well.

	Args:
		kf (KalmanFilter): Kalman filter with gating_distance method
		cost_matrix (np.ndarray): Shape (n_tracks, n_detections), appearance-based cost
		tracks: List of STrack objects with Kalman state
		detections: List of detection objects
		logger: Optional logger for debug output
		only_position (bool): Use only position in gating. Defaults to False.
		lambda_ (float): Blending weight for appearance cost in [0, 1].
			λ=1 → pure appearance; λ=0 → pure motion. Defaults to 0.98.

	Returns:
		np.ndarray: Fused cost matrix

	Note:
		Rows with gating_distance > threshold are marked as inf.
		The cost matrix is modified in place.
	"""
	if cost_matrix.size == 0:
		return cost_matrix

	gating_dim = 2 if only_position else 4
	gating_threshold = kalman_filter.chi2inv95[gating_dim]

	if logger is not None:
		logger.debug(f'Gating threshold: {gating_threshold}')

	measurements = np.asarray([det.to_xyah() for det in detections])

	# Fuse motion and appearance costs
	for row, track in enumerate(tracks):
		# Compute Mahalanobis distance (motion/dynamics cost)
		gating_distance = kf.gating_distance(
			track.mean, track.covariance, measurements, only_position, metric='maha')

		if logger is not None:
			logger.debug(f'Track {track.track_id} gating distance: {gating_distance}')

		# Gate out impossible matches
		cost_matrix[row, gating_distance > gating_threshold] = np.inf

		# Fuse: weighted average of appearance and motion costs
		cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance

	return cost_matrix
