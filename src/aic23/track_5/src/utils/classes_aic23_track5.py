"""
### 7 classes has object on dataset
1, motorbike   0
2, DHelmet     1
3, DNoHelmet   2
4, P1Helmet    3
5, P1NoHelmet  4
6, P2Helmet    5
7, P2NoHelmet  6
"""
def get_list_7_classses():
	"""Non of P0 passenger in list

		None of [P0Helmet, P0NoHelmet]

	Returns:
		(list): list of label
	"""
	classes = ['motorbike', 'DHelmet', 'DNoHelmet', 'P1Helmet',
			   'P1NoHelmet', 'P2Helmet', 'P2NoHelmet']

	return classes


"""
### 3 classes
1, motorbike  0
2, Helmet     1
3, NoHelmet   2
"""
def get_list_3_classses():
	"""Non of P0 passenger in list

		None of [P0Helmet, P0NoHelmet]

	Returns:
		(list): list of label
	"""
	classes = ['motorbike', 'Helmet', 'NoHelmet']

	return classes


"""
### 2 classes
1, motorbike  0
2, driver     1
"""
def get_list_2_classses():
	"""Non of P0 passenger in list

		None of [P0Helmet, P0NoHelmet]

	Returns:
		(list): list of label
	"""
	classes = ['motorbike', 'driver']

	return classes


"""
### 1 class
1, motorbike  0
"""
def get_list_1_classses():
	"""Non of P0 passenger in list

		None of [P0Helmet, P0NoHelmet]

	Returns:
		(list): list of label
	"""
	classes = ['motorbike']

	return classes
