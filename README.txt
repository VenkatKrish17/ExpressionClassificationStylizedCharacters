Attachements 

1. preprocess.py
	1.Download the image data set from http://grail.cs.washington.edu/projects/deepexpr/
	2.change the image directory inside preprocess.py and exceute the code
	3. run a seperate execution for test set by choosing 1 character out of 6 and mentioning the directory in preprocess.py
	4.The code creates preprocessed directory and test_set directory
2. imagemodelling.py
	1. Each model is created and executed using two functions. 
		(A) a function which returns the compiled model
			ex- first_model_cnn(weights_file)
			the models takes the weights file as parameter
			if its passed ,
				it will assign the weights and return the compiled model
			else,
				returns the compiled model without weights
		(b)a function which fits and evaluates the models
			This function creates the model by either calling the modelling function with the weights file or None
			takes the boolean parameter use_weights=True to use the existing weights or will fit the model again by calling fit_model()
	2. fit_model() 
		function which fits the model and stores the weight in hdf5 file
	3. evaluate_model()
		function which evaluates against the test set, and creates classification report, confusion matrix, learning curve
	4. ensemble_model()
		takes models as arguments and creates an ensemble model and evaluates it.
	
