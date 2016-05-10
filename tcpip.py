import socket



class client():
	def __init__(self,TCP_IP,TCP_PORT):
		self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		print "Establishing TCPIP connection to %s,%s" %(TCP_IP,TCP_PORT) 
		self.s.connect((TCP_IP,TCP_PORT))
		
	def send(self,MESSAGE):
		self.s.send(MESSAGE)

	def receive(self,BUFFER):
		return self.s.recv(BUFFER)
	
	def __exit__(self):
		print "Closing TCPIP connection."
		self.s.close()


class server():
	def __init__(self,TCP_IP,TCP_PORT):
		self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		print "Establishing TCPIP server on  %s,%s" %(TCP_IP,TCP_PORT) 
		self.s.bind((TCP_IP,TCP_PORT))
		self.s.listen(1)
		self.connection =None
		self.waitconnection()

	def waitconnection(self):
		print " Waiting for connection"
		connection, client_address = self.s.accept()
		print "Client connected:", client_address
		self.connection = connection

	def disconnect(self):
		self.connection.close()
		self.connection = None

	def send(self,MESSAGE):
		if not self.connection:
			self.waitconnection()
		self.connection.send(MESSAGE)

	def receive(self,BUFFER):
		if not self.connection:
			self.waitconnection()
		return self.connection.recv(BUFFER)
	
	def __exit__(self):
		print "Closing TCPIP connection."
		self.connection.close()
		self.s.close()
