#!/usr/bin/env python
"""interlock_plottemp.py: The script check apparatus3huletlab@gmail account for emails."""
"""If doesn't receive an email from interlock.vi runs on Andor PC send out notices"""
"""If receive email from Reno@rice.edu for temperature data plot it and send it out """


__author__      = "Ernie Yang"


import email, getpass, imaplib, os
import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from configobj import ConfigObj
import plot_week
import datetime

detach_dir = '.' # directory where to save attachments (default: current)
user = "apparatus3huletlab"#raw_input("Enter your GMail username:")
pwd = "cesium137"#getpass.getpass("Enter your password: ")

def getGmail(usr, pwd):
	# connecting to the gmail imap server
	m = imaplib.IMAP4_SSL("imap.gmail.com")
	m.login(user,pwd)
	m.select("[Gmail]/All Mail") # here you a can choose a mail box like INBOX instead
	# use m.list() to get all the mailboxes
	resp, items = m.search(None, "UNSEEN") # you could filter using the IMAP rules here (check http://www.example-code.com/csharp/imap-search-critera.asp)
	emailid = items[0].split() # getting the mails id
	return m, emailid
	
def parsemail(server,emailid):
	resp, data = server.fetch(emailid, "(RFC822)") # fetching the mail, "`(RFC822)`" means "get the whole stuff", but you can ask for headers only, etc
	email_body = data[0][1] # getting the mail content
	return email.message_from_string(email_body) # parsing the mail content to get a mail object

def deletemail(server,emailid):
	server.store(emailid,'+X-GM-LABELS', '\\Trash')

def sendGmail(usr, pwd,recipients,subject,body):
	server = 'smtp.gmail.com'
	port = 587
	headers = ["From: " + usr,
			   "Subject: " + subject,
			   "To: " +",".join(recipients[0:]),
			   "MIME-Version: 1.0",
			   "Content-Type: text/html"]
			   
	headers = "\r\n".join(headers)
	msg = headers + "\r\n\r\n" + body
	session = smtplib.SMTP(server, port)
	#~ session.set_debuglevel(True)
	session.ehlo()
	session.starttls()
	session.ehlo()
	session.login(usr,pwd)
	session.sendmail(usr, recipients,msg)
	session.quit()
	
def sendGmailwithpng(usr, pwd,recipients,subject,body,file):
	server = 'smtp.gmail.com'
	port = 587
	
	msg = MIMEMultipart()
	msg['Subject'] = subject
	msg['To'] = ",".join(recipients[0:])
	msg['From'] = usr
	
	path,filename = os.path.split(file)
	img = MIMEImage(open(file, 'rb').read(), _subtype="png")
	img.add_header('Content-Disposition', 'attachment', filename=filename)
	msg.attach(img)
	
	part = MIMEText('text', "plain")
	part.set_payload(body)
	msg.attach(part)
			   

	session = smtplib.SMTP(server, port)
	#~ session.set_debuglevel(True)
	session.ehlo()
	session.starttls()
	session.ehlo()
	session.login(usr,pwd)
	session.sendmail(usr, recipients,msg.as_string())
	session.quit()
	
def plotTemp(mail,file):
	print "Try to plottemp"
	print "Downloading report and making plot"
	#~ # we use walk to create a generator so we can iterate on the parts and forget about the recursive headach
	attachmentnum = 1
	
	path, filename = os.path.split(file)
	
	for part in mail.walk():
		# multipart are just containers, so we skip them
		if part.get_content_maintype() == 'multipart':
			continue

		# is this part an attachment ?
		if part.get('Content-Disposition') is None:
			continue
		
		if (attachmentnum!=1):
			filename = filename + "_%d"%(attachmentnum)
		
		attachmentnum = attachmentnum +1
		
		att_path = os.path.join(path, filename)

		#Check if its already there
		if not os.path.isfile(att_path) :
			# finally write the stuff
			fp = open(att_path, 'wb')
			fp.write(part.get_payload(decode=True))
			fp.close()
			
	plot_week.plotreport(file,showplot = 0)
	
	
def sendInterlockWarning(recipients):
	print "Sending out warnning"
	sendGmail(user,pwd,recipients,"APP3 Intelock Warning!","Atomcool didn't receive an \"Intercock still running\" email from interlock!<br> \
	If problem resolved send an email with title \"ResetInterlock\" to this email to reset interlock checking from atomcool.<br>\
	You should receive an email of comfirmation after atomcool revceive the reset email.<br>\
	")
	
def sendInterlockReset(recipients):
	print "Sending out Reset"
	sendGmail(user,pwd,recipients,"APP3 Intelock Reseted!","The interlock from atomcool has been reseted.<br>\
	")
	
	
def checkGmail():
	os.getcwd()
	inifile = os.path.join(os.path.dirname(__file__),'interlock_plottemp.ini')
	print inifile
	config = ConfigObj(inifile)
	#~ config.read(inifile)

	iflag_ini = int(config["Interlock"]["flag"])
	path_to_store_temp_data = config["TempData"]["path"]
	recipients = [pair[1] for pair in config["InterlockMailList"].items()]
	temp_recipients = [pair[1] for pair in config["TemperatureMailList"].items()]
	iflag = 1 
	reset = 0 
	
	server, emailid = getGmail(user,pwd)
	
	for id in emailid:
		
		mail = parsemail(server,id)
		mailfrom = mail["From"]
		mailsubject = mail["Subject"]	
		
		#~ If the mail is temerature report. Down and plot it.
		if (mailfrom =="\"Insight Remote Notification\" <Reno@rice.edu>") & (mailsubject.split(",")[0] == "Insight Job"):
			time = datetime.datetime.now()
			filename = "%d_%02d_%02d_%02d_%02d"%(time.year,time.month,time.day,time.hour,time.minute)
			file = os.path.join(path_to_store_temp_data,filename)
			plotTemp(mail,file)
			sendGmailwithpng(user, pwd,temp_recipients ,"Weekly Temperature Data","See Attachment.",file+".png")
			
		#~ Check interlcok status
		elif (mailfrom =="apparatus3huletlab@gmail.com") & (mailsubject == "Interlock still running"):
			print " I receive an interlock still running email"
			deletemail(server,id)
			iflag = 0 # Set interlock flag to zero if recevie a mail from app 3 interlock 
			os.system('echo \'d *\' | mail -N') # delete the cron message send from root to app3
			
		#~ Rest interlock if receive an mail tile "ResetInterlock"
		elif(mailsubject == "ResetInterlock"):
			print "Reset Interlcok"
			deletemail(server,id)
			sendInterlockReset(recipients)
			reset = 1 
			config["Interlock"]["flag"] = 0
			
	if ((iflag ==1)&(iflag_ini==0)&(reset==0)): 
		sendInterlockWarning(recipients)
		config["Interlock"]["flag"] = 1
		
	#~ config["Interlock"]["message"] = "I ran from crontab"
	
	config.write()
	
	
if __name__ == "__main__":
	checkGmail()
	#~ sendGmail(user,pwd,('erniejazz@gmail.com','erniejazz1984@gmail.com'),"newnew","Test")
	