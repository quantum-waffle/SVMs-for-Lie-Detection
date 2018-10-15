from socketIO_client import SocketIO
import data_manager as dm, tensor_nonlinear_gaussian_rbf_SVM as svm


def on_connection(*args):
	""" message sent to server to validate connection

	"""
    print('on_connection', args)


def main():
	while True:
		with SocketIO('localhost', 8000) as socketIO:
			entry_file = socketIO.wait()
			socketIO.on('connected', on_connection)
			
			""" receive file from server and send it to svm

			return 0 for truth, 1 for lie
			"""
			response = smv.main(entry_file)

			socketIO.emit(response)
			""" send to the server
				1. array with prediction and %
				2. send two sepparate messages, prediction and then %
			"""

if __name__ == "__main__":
    main()