import numpy as np


def select_features(edit_d, orig_features, adv_features):
	"""
	Returns the indices of those words whose vectors change most between
	orig_features and adv_features as a Numpy array of size (samples, edit_d).

	Arguments:
	edit_d: The edit distance, or the number of indices to get per sample
	orig_features: A Numpy array of size (samples, sentence_size, word_size)
	adv_features: A Numpy array of size (samples, sentence_size, word_size)

	By calculating the indices of the words that have been changed most by
	an attack mechanism, we know which words to find candidates for.
	"""

	samples, sentence_size, word_size = orig_features.shape
	mask = np.zeros(word_size)
	residual_rows = (adv_features - orig_features).reshape(
										samples*sentence_size, word_size)
	orig_rows = orig_features.reshape((samples*sentence_size, word_size))
	zero_indices = np.where((orig_rows == tuple(mask)).all(axis=1))
	residual_rows[zero_indices] = mask

	norms = np.linalg.norm(residual_rows, ord=2, axis=1).reshape(samples, 
														 sentence_size)
	sorted_norms = np.argsort(norms)
	rev_indices = sorted_norms[:, -edit_d:]

	# rev_indices = np.argpartition(norms, np.argmin(norms, axis=0))[:, -edit_d:]
	return np.flip(rev_indices, axis=1)

