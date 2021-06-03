#!/usr/bin/python3
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--source', help="Folder path of all training files.", required=True)
parser.add_argument(
    '--destination', help="Path to save your trained model.", required=True)


def train_model(model,source):
	"""
	TODO: Implement your method for training the model here.
	"""
	raise Exception("Method train_model not implemented.")

def save_model(model, destination):
	"""
	TODO: Implement your method for saving the training the model here.
	"""
	raise Exception("Method save_model not implemented.")
	

if __name__ == "__main__":
	args = parser.parse_args()
	train_model(model,args.source)

	save_model(model,args.destination)
