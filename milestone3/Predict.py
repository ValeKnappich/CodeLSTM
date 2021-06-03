#!/usr/bin/python3

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model', help="Path of trained model.", required=True)
parser.add_argument(
    '--source', help="Folder path of all test files.", required=True)
parser.add_argument(
    '--destination', help="Path to output json file of extracted predictions.", required=True)

def predict(model,test_files):
	"""
	TODO: Implement your method for predicting the index here.
	"""
	raise Exception("Method evaluate not implemented.")
	

def load_model(source):
	"""
	TODO: Implement your code to load the serialized model.
	"""
	raise Exception("Method evaluate not implemented.")

def write_predictions(destination,predictions):
	"""
	TODO: Implement your code to write predictions to file. For format
	of the JSON file refer to sample_predictions.json file.
	"""
	raise Exception("Method write_predictions not implemented.")

if __name__ == "__main__":
	args = parser.parse_args()
	
	# load the serialized model
	model = load_model(args.model)

	# predict incorrect location for each test example.
	predictions = predict(model,args.source)

	# write predictions to file
	write_predictions(args.destination,predictions)

	