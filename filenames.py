

def next_model_filename():
	file_nums = [int(file.replace('model', '').replace('.h5', '')) 
				for file in os.listdir("models")]
	if not file_nums: 
		return 'models/model0.h5'
	return 'models/model' + str(max(file_nums) + 1) + '.h5'


def model_file(model, data_size):
	size_name = str(data_size//1000) + 'k'
	model_file = 'models/' + model + '_' + size_name + '.h5'
	return model_file


def data_file(data_size):
	size_name = str(data_size//1000) + 'k'
	test_file = 'embed_text/fb_test_vectors_' + size_name + '.npy'
	return test_file
