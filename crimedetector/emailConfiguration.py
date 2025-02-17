import smtplib
from email.mime.text import MIMEText

def send_alert(camera_name, crime_type, severity):
    user_email = "ishimweinstein@gmail.com"  # Fetch dynamically from DB in real scenario
    subject = f"🚨 Crime Alert: {crime_type} detected!"
    body = f"A crime of type **{crime_type}** was detected on **{camera_name}** with severity score **{severity}**."

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = "noreply@crimewatch.com"
    msg["To"] = user_email

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login("ishimwe.nyanja@gmail.com", "pucl erdi dgja nhnb")  # Use app password
        server.sendmail("noreply@crimewatch.com", user_email, msg.as_string())
        server.quit()
        print(f"📧 Alert email sent to {user_email}")
    except Exception as e:
        print(f"❌ Email failed: {e}")