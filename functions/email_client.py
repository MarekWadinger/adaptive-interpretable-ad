import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


class EmailClient:
    def __init__(
            self,
            sender_email: str,
            sender_password: str,
            recipient_email: str,
            subject: str):
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipient_email = recipient_email
        self.subject = subject

    def send_email(
        self,
        msg,
    ):
        # Create the email message
        body = MIMEMultipart()
        body['From'] = self.sender_email
        body['To'] = self.recipient_email
        body['Subject'] = self.subject

        # Attach the message to the email
        if isinstance(msg, dict):
            msg = (f"Anomaly detected at {msg['time']}.\n\n"
                   f"Current upper limits on signals are:\n"
                   f"{msg['level_high']}\n\n"
                   f"Current lower operating limits are:\n{msg['level_low']}")
        body.attach(MIMEText(msg, 'plain'))

        # Try to automatically select server based on sender email
        smtps = {
            "gmail.com": "smtp.gmail.com",
            "yahoo.com": "smtp.mail.yahoo.com",
            "outlook.com": "smtp-mail.outlook.com",
            "icloud.com": "smtp.mail.me.com"}
        # Create an SMTP connection
        smtp_server = smtps.get(
            self.sender_email.split('@')[1], 'smtp.mail.com')
        with smtplib.SMTP(smtp_server, 587) as server:
            server.starttls()  # Enable TLS encryption
            server.login(self.sender_email, self.sender_password)

            # Send the email
            server.sendmail(
                self.sender_email,
                self.recipient_email,
                body.as_string())

        print("Email sent successfully!")