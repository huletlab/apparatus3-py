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




