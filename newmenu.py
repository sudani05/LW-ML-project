import os
import time
import base64
import tempfile
import streamlit as st
import paramiko
import subprocess
import pandas as pd
import pyautogui
import pywhatkit
from twilio.rest import Client
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import tweepy
from instagrapi import Client as InstaClient
import facebook as fb
import requests
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import google.generativeai as genai
import string
import re
import joblib
import psutil
from googlesearch import search
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------ UTILITY FUNCTIONS ------------------
def extract_number(text):
    words_to_digits = {
        "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7,
        "eight": 8, "nine": 9, "ten": 10
    }
    if text.isdigit():
        return int(text)
    return words_to_digits.get(text.lower(), 1)

def ping_host(ip):
    try:
        result = subprocess.run(["ping", "-c", "1", ip], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    except Exception:
        return False

def ssh_connect(ip, username, password):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(ip, username=username, password=password)
        st.session_state["ssh"] = ssh
        st.sidebar.success("SSH connected successfully.")
        return True
    except paramiko.AuthenticationException:
        st.sidebar.error("Authentication failed.")
    except paramiko.SSHException as e:
        st.sidebar.error(f"SSH error: {e}")
    except Exception as e:
        st.sidebar.error(f"Failed to connect: {e}")
    return False

def run_command(ssh, command):
    stdin, stdout, stderr = ssh.exec_command(command)
    return stdout.read().decode(), stderr.read().decode()

def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'\d+', '', text)  # remove numbers
        text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
        text = text.strip()
    else:
        text = ""  # handle NaN or non-string entries
    return text

def get_ram_info():
    virtual_mem = psutil.virtual_memory()
    return {
        "total": virtual_mem.total / (1024 ** 3),
        "available": virtual_mem.available / (1024 ** 3),
        "usage": 100 * (1 - virtual_mem.available / virtual_mem.total)
    }

def configure_gemini():
    api_key = "AIzaSyBVzYPGp-4HF5CaYq0yvR7jWN3g_i_Hhwg"
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def get_all_links(url, domain_only=True):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        links = set()
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(url, href)
            
            if domain_only:
                if urlparse(absolute_url).netloc == urlparse(url).netloc:
                    links.add(absolute_url)
            else:
                links.add(absolute_url)
                
        return list(links)
    except Exception as e:
        st.error(f"Error getting links: {e}")
        return []

def scrape_website_data(url, max_pages=10):
    try:
        visited = set()
        to_visit = [url]
        data = []
        
        while to_visit and len(visited) < max_pages:
            current_url = to_visit.pop(0)
            if current_url in visited:
                continue
                
            try:
                response = requests.get(current_url, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove scripts and styles
                for script in soup(["script", "style"]):
                    script.decompose()
                
                page_data = {
                    "url": current_url,
                    "title": soup.title.string if soup.title else "No title",
                    "text": soup.get_text(separator=' ', strip=True),
                    "links": []
                }
                
                data.append(page_data)
                visited.add(current_url)
                
                # Get new links
                for link in soup.find_all('a', href=True):
                    absolute_url = urljoin(current_url, link['href'])
                    if urlparse(absolute_url).netloc == urlparse(url).netloc:
                        page_data["links"].append(absolute_url)
                        if absolute_url not in visited and absolute_url not in to_visit:
                            to_visit.append(absolute_url)
                
                time.sleep(1)  # Be polite with delay between requests
                
            except Exception as e:
                st.warning(f"Could not scrape {current_url}: {e}")
                continue
                
        return data
    except Exception as e:
        st.error(f"Error during scraping: {e}")
        return []

def save_data(data, format='csv'):
    df = pd.DataFrame(data)
    
    if format == 'csv':
        return df.to_csv(index=False).encode('utf-8')
    elif format == 'json':
        return df.to_json(orient='records').encode('utf-8')
    elif format == 'excel':
        output = pd.ExcelWriter('website_data.xlsx', engine='xlsxwriter')
        df.to_excel(output, index=False)
        output.close()
        with open('website_data.xlsx', 'rb') as f:
            return f.read()

@st.cache_data
def load_titanic_data():
    try:
        return pd.read_csv("Titanic-Dataset.csv")
    except FileNotFoundError:
        st.error("Titanic dataset not found. Please upload it.")
        uploaded = st.file_uploader("Upload Titanic-Dataset.csv", type=["csv"])
        if uploaded:
            with open("Titanic-Dataset.csv", "wb") as f:
                f.write(uploaded.read())
            return pd.read_csv("Titanic-Dataset.csv")
        return None

@st.cache_data
def load_churn_data():
    try:
        return pd.read_csv("Churn_Modelling.csv")
    except FileNotFoundError:
        st.error("Churn dataset not found. Please upload it.")
        uploaded = st.file_uploader("Upload Churn_Modelling.csv", type=["csv"])
        if uploaded:
            with open("Churn_Modelling.csv", "wb") as f:
                f.write(uploaded.read())
            return pd.read_csv("Churn_Modelling.csv")
        return None

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Mega AI Tool Suite", page_icon="🤖", layout="wide")

# ------------------ SIDEBAR ------------------
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose a section",
    ["Home", "Remote Linux", "Automation Tools", "AI Predictors", "Gemini AI Assistants",
     "Missing Value Predictor", "Google Search Tool", "Website Scraper",
     "Titanic Survival Predictor", "Bank Churn Predictor"])

# Display RAM info in sidebar
ram_info = get_ram_info()
st.sidebar.title("System Resources")
st.sidebar.metric("Total RAM", f"{ram_info['total']:.2f} GB")
st.sidebar.metric("Available RAM", f"{ram_info['available']:.2f} GB")
st.sidebar.progress(ram_info['usage'] / 100)
st.sidebar.caption(f"RAM Usage: {ram_info['usage']:.1f}%")

# ------------------ HOME PAGE ------------------
if app_mode == "Home":
    st.title("🤖 Mega AI Tool Suite")
    st.markdown("""
    Welcome to the Mega AI Tool Suite! This application combines several powerful tools:
    
    - *Remote Linux*: Connect to remote servers via SSH and run commands
    - *Automation Tools*: Send messages, emails, SMS, and post to social media
    - *AI Predictors*: Salary prediction, job selection, and health/churn modeling
    - *Gemini AI Assistants*: Legal expert and DevOps career mentor powered by Google Gemini
    - *Missing Value Predictor*: Fill missing Y values using Linear Regression
    - *Google Search Tool*: Perform Google searches directly from the app
    - *Website Scraper*: Download website content and data
    - *Titanic Survival Predictor*: Predict survival on the Titanic
    - *Bank Churn Predictor*: Predict customer churn for banks
    - *System Info*: View system resource usage
    
    Select a section from the sidebar to get started.
    """)

# ------------------ REMOTE LINUX SECTION ------------------
elif app_mode == "Remote Linux":
    st.title("🔌 Remote Linux Management")
    
    # SSH connection sidebar
    st.sidebar.title("SSH Connection")
    ip = st.sidebar.text_input("IP Address")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Test Connection"):
        if ping_host(ip):
            st.sidebar.success("Host is reachable.")
        else:
            st.sidebar.error("Host unreachable.")

    if st.sidebar.button("Connect"):
        if not ip or not username or not password:
            st.sidebar.warning("Please fill in all fields.")
        else:
            ssh_connect(ip, username, password)

    # Linux Commands Tab
    st.header("Linux Commands")
    if "ssh" in st.session_state:
        ssh = st.session_state["ssh"]

        linux_commands = {
            "1. Date": "date",
            "2. Cal": "cal",
            "3. Ifconfig": "ifconfig",
            "4. List Files (ls)": "ls",
            "5. Show Hostname": "hostname",
            "6. Current User": "whoami",
            "7. System Uptime": "uptime",
            "8. Memory Usage": "free -m",
            "9. View Processes": "ps aux",
            "10. Real-time Monitor": "top -b -n 1",
            "11. Disk Usage": "df -h",
            "12. OS & Kernel Info": "uname -a",
            "13. View /etc/shadow": "cat /etc/shadow",
            "14. View /etc/passwd": "cat /etc/passwd",
            "15. View /etc/group": "cat /etc/group"
        }

        selected_command = st.selectbox("Choose Linux Command", list(linux_commands.keys()))
        if st.button("Run Linux Command"):
            out, err = run_command(ssh, linux_commands[selected_command])
            if out:
                st.code(out)
            elif err:
                st.error(err)
            else:
                st.success("Command executed successfully.")

        st.subheader("Run Custom Linux Command")
        custom_cmd = st.text_input("Enter custom Linux command")
        if st.button("Run Custom Command"):
            out, err = run_command(ssh, custom_cmd)
            if out:
                st.code(out)
            elif err:
                st.error(err)
            else:
                st.success("Command executed successfully.")
    else:
        st.warning("Please connect via SSH from the sidebar first.")

# ------------------ AUTOMATION TOOLS SECTION ------------------
elif app_mode == "Automation Tools":
    st.title("🤖 Automation Tools")
    
    choice = st.selectbox("Choose an action", [
        "--Select--", "WhatsApp", "Email", "SMS", "Phone Call",
        "Post Tweet", "Post to Instagram", "Post to Facebook"
    ])

    # WhatsApp Automation
    if choice == "WhatsApp":
        st.info("📲 Send messages via WhatsApp Web")
        number = st.text_input("Phone Number with +91")
        message = st.text_input("Message")
        repeat_text = st.text_input("Repeat count (e.g. one, two)", value="1")

        if st.button("Send WhatsApp Message"):
            if number and message:
                repeat = extract_number(repeat_text)
                try:
                    pywhatkit.sendwhatmsg_instantly(number, "Hi", wait_time=15, tab_close=False)
                    time.sleep(20)
                    for _ in range(repeat):
                        pyautogui.write(message)
                        pyautogui.press("enter")
                        time.sleep(0.5)
                    st.success(f"✅ Sent {repeat} times.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
            else:
                st.warning("Please fill all fields.")

    # Email Automation
    elif choice == "Email":
        st.info("📧 Send email using Gmail App Password")
        sender = st.text_input("Your Email")
        app_pass = st.text_input("App Password", type="password")
        recipient = st.text_input("Recipient Email")
        subject = st.text_input("Subject")
        body = st.text_area("Email Body")

        if st.button("Send Email"):
            if sender and app_pass and recipient:
                try:
                    msg = MIMEMultipart()
                    msg['From'] = sender
                    msg['To'] = recipient
                    msg['Subject'] = subject
                    msg.attach(MIMEText(body, 'plain'))

                    server = smtplib.SMTP('smtp.gmail.com', 587)
                    server.starttls()
                    server.login(sender, app_pass)
                    server.sendmail(sender, recipient, msg.as_string())
                    server.quit()

                    st.success("✅ Email sent successfully!")
                except Exception as e:
                    st.error(f"❌ {e}")
            else:
                st.warning("Fill all required fields.")

    # SMS Automation
    elif choice == "SMS":
        st.info("📩 Send SMS via Twilio")
        sid = st.text_input("Twilio SID", value="ACe7d83400d3e2ca46984d756591f8ed98")
        token = st.text_input("Auth Token", type="password", value="1015bdbb08c100c9675557e150e4f9a5")
        from_num = st.text_input("Twilio Number", value="+16318257254")
        to_num = st.text_input("Recipient Number", value="+91")
        msg = st.text_input("Message")

        if st.button("Send SMS"):
            if sid and token and from_num and to_num and msg:
                try:
                    client = Client(sid, token)
                    message = client.messages.create(body=msg, from_=from_num, to=to_num)
                    st.success(f"✅ SMS sent! SID: {message.sid}")
                except Exception as e:
                    st.error(f"❌ {e}")
            else:
                st.warning("Please fill all fields.")

    # Phone Call Automation
    elif choice == "Phone Call":
        st.info("📞 Call using Twilio")
        sid = st.text_input("Twilio SID", key="csid", value="ACe7d83400d3e2ca46984d756591f8ed98")
        token = st.text_input("Auth Token", type="password", key="ctoken", value="1015bdbb08c100c9675557e150e4f9a5")
        from_num = st.text_input("Twilio Number", value="+16318257254", key="cfrom")
        to_num = st.text_input("Recipient Number", value="+91", key="cto")
        twiml_url = st.text_input(
            "TwiML Bin URL",
            value="https://handler.twilio.com/twiml/EHXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
            help="Paste a valid TwiML Bin URL"
        )

        if st.button("Make Call"):
            if sid and token and from_num and to_num and twiml_url:
                try:
                    client = Client(sid, token)
                    call = client.calls.create(to=to_num, from_=from_num, url=twiml_url)
                    st.success(f"📞 Call initiated! SID: {call.sid}")
                except Exception as e:
                    st.error(f"❌ {e}")
            else:
                st.warning("Fill all fields.")

    # Twitter Automation
    elif choice == "Post Tweet":
        st.info("🐦 Tweet with text and image using Twitter API")

        bearer_token = 'AAAAAAAAAAAAAAAAAAAAAPYw3AEAAAAAT84MYzMPrkVwBRSdtf%2BSjjZTcSo%3DGS2v2vweChS5JbwpdblTq1lHEWjEynakxZlYxG4KAYUFw13mAY'
        consumer_key = 'HRTMFeBVGzMH00FAYkdxwDFJ8'
        consumer_secret = 'QTw8uI60ZDEIXsOwtmMCVF3w5UTqx5Tu5KizD84iRnTBjYGvrX'
        access_token = '1941841332697522176-vISAAeu2l7QvQmNcPlLxl0E7LIfEYl'
        access_token_secret = '68juASxBu7wTgyvMORnGREhoNPAL1TLYkkRsn1jB1yVHK'

        tweet_text = st.text_input("Tweet Text")
        uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

        if st.button("Tweet Now"):
            if tweet_text and uploaded_file:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        tmp.write(uploaded_file.read())
                        image_path = tmp.name

                    # V1 Auth for media
                    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
                    auth.set_access_token(access_token, access_token_secret)
                    api = tweepy.API(auth, wait_on_rate_limit=True)

                    # V2 Auth for tweet
                    client = tweepy.Client(
                        bearer_token=bearer_token,
                        consumer_key=consumer_key,
                        consumer_secret=consumer_secret,
                        access_token=access_token,
                        access_token_secret=access_token_secret,
                        wait_on_rate_limit=True
                    )

                    media = api.media_upload(filename=image_path)
                    media_id = media.media_id_string
                    response = client.create_tweet(text=tweet_text, media_ids=[media_id])

                    st.success("✅ Tweet posted!")
                    st.json(response.data)
                except Exception as e:
                    st.error(f"❌ Error: {e}")
            else:
                st.warning("Upload image and enter text.")

    # Instagram Automation
    elif choice == "Post to Instagram":
        st.info("📸 Post image to Instagram using instagrapi")
        username = st.text_input("Instagram Username")
        password = st.text_input("Instagram Password", type="password")
        caption = st.text_input("Image Caption")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg"])

        if st.button("Upload to Instagram"):
            if not username or not password or not uploaded_file:
                st.warning("⚠️ Please provide all inputs.")
            else:
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.read())

                    try:
                        cl = InstaClient()
                        cl.login(username, password)
                        cl.photo_upload(temp_path, caption)
                        st.success("✅ Successfully uploaded to Instagram!")
                    except Exception as e:
                        st.error(f"❌ Upload failed: {e}")

    # Facebook Automation
    elif choice == "Post to Facebook":
        st.title("📤 Post to Facebook Page")
        message = st.text_area("📜 Enter your post message")
        uploaded_file = st.file_uploader("🖼️ Choose an image to post", type=["jpg", "jpeg", "png"])

        ACCESS_TOKEN = "YOUR_LONG_LIVED_ACCESS_TOKEN"
        PAGE_ID = "YOUR_PAGE_ID"

        if st.button("🚀 Post to Facebook"):
            try:
                graph = fb.GraphAPI(access_token=ACCESS_TOKEN, version='3.0')
                page_info = graph.get_object(f'/{PAGE_ID}?fields=access_token')
                page_access_token = page_info.get("access_token")

                if not page_access_token:
                    st.error("❌ Failed to get Page Access Token.")
                else:
                    if uploaded_file:
                        with open("temp_image.jpg", "wb") as f:
                            f.write(uploaded_file.read())

                        image_post_url = f"https://graph.facebook.com/{PAGE_ID}/photos"
                        image_payload = {"message": message, "access_token": page_access_token}
                        with open("temp_image.jpg", "rb") as img:
                            image_files = {"source": img}
                            response = requests.post(image_post_url, data=image_payload, files=image_files)
                        os.remove("temp_image.jpg")
                    else:
                        post_url = f"https://graph.facebook.com/{PAGE_ID}/feed"
                        response = requests.post(post_url, params={"message": message, "access_token": page_access_token})

                    if response.status_code == 200:
                        st.success("✅ Post was successful!")
                    else:
                        st.error("❌ Post failed.")
                        st.code(response.text)

            except Exception as e:
                st.error(f"⚠️ Error: {e}")

# ------------------ AI PREDICTORS SECTION ------------------
elif app_mode == "AI Predictors":
    st.title("🧠 AI Career & Health Prediction Suite")
    st.markdown("Choose a tab to use: *Salary Prediction*, *Job Selection*")

    # Tabs for AI predictors
    tab1, tab2= st.tabs(["💰 Salary Prediction", "📊 Job Selection Prediction"])

    # TAB 1: SALARY PREDICTION
    with tab1:
        st.header("💼 Salary Prediction Based on Experience")
        if not os.path.exists("SalaryData.csv"):
            st.warning("📂 SalaryData.csv not found. Please upload it.")
            uploaded = st.file_uploader("Upload SalaryData.csv", type=["csv"])
            if uploaded:
                with open("SalaryData.csv", "wb") as f:
                    f.write(uploaded.read())
                st.success("✅ File uploaded. Please reload the app.")
        else:
            try:
                df = pd.read_csv("SalaryData.csv")
                with st.expander("🔍 View Salary Dataset"):
                    st.dataframe(df)

                X = df[["YearsExperience"]]
                y = df[["Salary"]]
                salary_model = LinearRegression()
                salary_model.fit(X, y)

                exp = st.slider("📊 Enter your years of experience:", 0.0, 20.0, step=0.1)
                pred_salary = salary_model.predict([[exp]])[0][0]

                st.subheader("💸 Predicted Salary:")
                st.success(f"Estimated Salary for {exp} years: ₹{pred_salary:,.2f}")
            except Exception as e:
                st.error(f"Error reading SalaryData.csv: {e}")

    # TAB 2: JOB SELECTION PREDICTION
    with tab2:
        st.header("🧑‍💼 Job Selection Predictor")
        if not os.path.exists("dataset.csv"):
            st.warning("📂 dataset.csv not found. Please upload it.")
            uploaded = st.file_uploader("Upload dataset.csv", type=["csv"])
            if uploaded:
                with open("dataset.csv", "wb") as f:
                    f.write(uploaded.read())
                st.success("✅ File uploaded. Please reload the app.")
        else:
            try:
                df = pd.read_csv("dataset.csv")

                # Encode categorical features
                le_q = LabelEncoder()
                le_i = LabelEncoder()
                le_r = LabelEncoder()
                le_res = LabelEncoder()

                df["Qualification"] = le_q.fit_transform(df["Qualification"])
                df["Internship"] = le_i.fit_transform(df["Internship"])
                df["Referral"] = le_r.fit_transform(df["Referral"])
                df["Job_Result"] = le_res.fit_transform(df["Job_Result"])

                X = df[[
                    "Qualification", "Internship", "Comm_Skill", "Tech_Skill_Level",
                    "Certifications", "Interview_Score", "Resume_Score", "Referral"
                ]]
                y = df["Job_Result"]

                job_model = LinearRegression()
                job_model.fit(X, y)

                st.subheader("📥 Candidate Details")
                col1, col2 = st.columns(2)
                with col1:
                    qualification = st.selectbox("Qualification", le_q.classes_.tolist())
                    internship = st.selectbox("Internship", le_i.classes_.tolist())
                    referral = st.selectbox("Referral", le_r.classes_.tolist())
                    certifications = st.number_input("Certifications Count", 0)
                    resume_score = st.number_input("Resume Score (0-10)", 0, 10)
                with col2:
                    comm_skill = st.number_input("Communication Skill (0-10)", 0, 10)
                    tech_skill = st.number_input("Technical Skill Level (0-10)", 0, 10)
                    interview_score = st.number_input("Interview Score (0-10)", 0, 10)

                if st.button("🔍 Predict Job Result"):
                    try:
                        q = le_q.transform([qualification])[0]
                        i = le_i.transform([internship])[0]
                        r = le_r.transform([referral])[0]

                        input_features = [[
                            q, i, comm_skill, tech_skill,
                            certifications, interview_score, resume_score, r
                        ]]

                        result = job_model.predict(input_features)[0]
                        result_index = int(round(result))
                        result_index = max(0, min(result_index, len(le_res.classes_) - 1))
                        final_result = le_res.inverse_transform([result_index])[0]

                        if "Selected" in final_result:
                            st.success(f"✅ Prediction: {final_result}")
                        elif "Not Selected" in final_result:
                            st.error(f"❌ Prediction: {final_result}")
                        else:
                            st.info(f"ℹ️ Prediction: {final_result}")
                    except Exception as e:
                        st.error(f"Prediction error: {e}")
            except Exception as e:
                st.error(f"Error reading dataset.csv: {e}")

# ------------------ GEMINI AI ASSISTANTS SECTION ------------------
elif app_mode == "Gemini AI Assistants":
    st.title("🤖 Gemini AI Multi-Tool Assistant")
    st.markdown("""
    Welcome! This section provides two AI assistants powered by *Google Gemini*:
    - 🧑‍⚖️ Legal Assistant: Explains legal queries in simple terms.
    - 🛠️ DevOps Career Mentor: Offers advice on DevOps roles, tools, and certifications.

    > ⚠️ These are not substitutes for legal or career professionals.
    """)

    # Configure Gemini model
    model = configure_gemini()

    # Tabs for Gemini assistants
    tab1, tab2 = st.tabs(["🧑‍⚖️ Legal Assistant", "🛠️ DevOps Career Mentor"])

    # LEGAL ASSISTANT TAB
    with tab1:
        st.header("🧑‍⚖️ Gemini Legal Assistant")
        question = st.text_area("Enter your legal question", placeholder="e.g., Can my landlord evict me without notice?")
        
        if st.button("Get Legal Help", key="legal_button"):
            if question.strip() == "":
                st.warning("Please enter a legal question.")
            else:
                try:
                    with st.spinner("Processing your question with Gemini..."):
                        response = model.generate_content(
                            f"You are a legal expert. Explain this question in simple terms and provide possible legal steps:\n{question}"
                        )
                        st.success("Here's the response:")
                        st.text_area("AI Legal Help", value=response.text, height=300)
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    # DEVOPS CAREER MENTOR TAB
    with tab2:
        st.header("🛠️ DevOps Career Mentor")

        # Start or reuse chat session
        if "devops_chat" not in st.session_state:
            st.session_state.devops_chat = model.start_chat(
                history=[
                    {
                        "role": "user",
                        "parts": [
                            "You are a DevOps career mentor who guides individuals about DevOps tools, certifications, and job roles in IT."
                        ]
                    }
                ]
            )

        user_question = st.text_area("Enter your question:", placeholder="e.g., I'm good at Linux and Docker. What DevOps jobs should I consider?", key="devops_input")
        
        if st.button("Get Advice", key="devops_button"):
            if user_question.strip():
                try:
                    with st.spinner("Thinking like a DevOps mentor..."):
                        response = st.session_state.devops_chat.send_message(user_question)
                        st.subheader("🧠 Mentor's Response:")
                        st.write(response.text)
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("Please enter your question to get started.")

# ------------------ MISSING VALUE PREDICTOR SECTION ------------------
elif app_mode == "Missing Value Predictor":
    st.title("Fill Missing Y Values using Linear Regression")

    # Step 1: Upload dataset
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Original Dataset", df)

        if 'Y' not in df.columns:
            st.error("The dataset must contain a column named 'Y'")
        else:
            # Step 2: Split dataset into known and unknown Y values
            df_known = df[df['Y'].notnull()]
            df_missing = df[df['Y'].isnull()]

            if df_missing.empty:
                st.success("No missing values in 'Y'. Nothing to predict.")
            else:
                # Step 3: Prepare features (exclude 'Y')
                X_known = df_known.drop(columns=['Y'])
                y_known = df_known['Y']
                X_missing = df_missing.drop(columns=['Y'])

                # Optional: Check if non-numeric columns exist
                if X_known.select_dtypes(include=['object']).shape[1] > 0:
                    st.warning("Non-numeric features detected. Only numeric features will be used.")
                    X_known = X_known.select_dtypes(include=['number'])
                    X_missing = X_missing[X_known.columns]

                # Step 4: Train Linear Regression model
                model = LinearRegression()
                model.fit(X_known, y_known)

                # Step 5: Predict missing Y values
                y_pred = model.predict(X_missing)

                # Step 6: Replace missing values in original dataset
                df.loc[df['Y'].isnull(), 'Y'] = y_pred

                st.write("### Updated Dataset with Missing Y Filled", df)

                # Step 7: Allow download of updated dataset
                csv = df.to_csv(index=False)
                st.download_button("Download Updated CSV", data=csv, file_name="updated_dataset.csv", mime='text/csv')

# ------------------ GOOGLE SEARCH TOOL SECTION ------------------
elif app_mode == "Google Search Tool":
    st.title("🔍 Google Search in Streamlit")
    
    query = st.text_input("Enter search query:")
    
    if query:
        st.info(f"Searching Google for: *{query}*…")
        try:
            # Perform Google search
            urls = list(search(query, num_results=10))  
            st.success(f"Found {len(urls)} results.")
            
            # Display results
            for url in urls:
                st.write(f"[{url}]({url})")
                try:
                    # Fetch and parse page content
                    resp = requests.get(url, timeout=5)
                    soup = BeautifulSoup(resp.text, "html.parser")
                    title = soup.title.string.strip() if soup.title else "No title"
                    snippet_tag = soup.find("meta", attrs={"name": "description"})
                    snippet = snippet_tag["content"].strip() if snippet_tag else ""
                except Exception as e:
                    title, snippet = "Error fetching content", str(e)
                
                st.markdown(f"**{title}**\n\n{snippet}")
                st.write("---")
        except Exception as e:
            st.error(f"Search failed: {str(e)}")

# ------------------ WEBSITE SCRAPER SECTION ------------------
elif app_mode == "Website Scraper":
    st.title("Website Data Downloader")

    url = st.text_input("Enter website URL to download:", "https://example.com")
    max_pages = st.number_input("Maximum pages to scrape:", min_value=1, max_value=100, value=10)
    domain_only = st.checkbox("Only scrape pages from the same domain", value=True)
    download_format = st.selectbox("Download format:", ["csv", "json", "excel"])

    if st.button("Download Website Data"):
        if not url:
            st.error("Please enter a URL")
        elif not is_valid_url(url):
            st.error("Please enter a valid URL (include http:// or https://)")
        else:
            with st.spinner(f"Scraping {url} (this may take a while)..."):
                data = scrape_website_data(url, max_pages)
                
            if data:
                st.success(f"Successfully scraped {len(data)} pages!")
                st.download_button(
                    label="Download Data",
                    data=save_data(data, download_format),
                    file_name=f"website_data.{download_format}",
                    mime="text/csv" if download_format == 'csv' else
                         "application/json" if download_format == 'json' else
                         "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                # Show preview
                st.subheader("Preview of scraped data")
                st.dataframe(pd.DataFrame(data).head())
            else:
                st.error("No data was scraped. Please check the URL and try again.")

    st.warning("""
    **Important Notes:**
    1. Always check the website's `robots.txt` file and terms of service before scraping.
    2. This is a basic scraper - some websites may require more advanced techniques.
    3. Be respectful with scraping frequency to avoid overwhelming servers.
    """)

# ------------------ TITANIC SURVIVAL PREDICTOR SECTION ------------------
elif app_mode == "Titanic Survival Predictor":
    st.title("🚢 Titanic Dataset Explorer & Survival Predictor")
    
    # Load data
    df = load_titanic_data()
    if df is None:
        st.stop()
    
    # Show raw data
    if st.checkbox("Show Raw Data"):
        st.write(df)

    # Clean data
    df_clean = df.copy()
    df_clean = df_clean.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")
    df_clean.dropna(subset=['Age', 'Embarked', 'Sex'], inplace=True)

    # Encode categorical features
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    df_clean['Sex'] = le_sex.fit_transform(df_clean['Sex'])
    df_clean['Embarked'] = le_embarked.fit_transform(df_clean['Embarked'])

    # Sidebar input for prediction
    st.sidebar.header("🎯 Predict Survival")

    pclass = st.sidebar.selectbox("Passenger Class (1=1st, 2=2nd, 3=3rd)", [1, 2, 3])
    sex = st.sidebar.selectbox("Sex", ['male', 'female'])
    age = st.sidebar.slider("Age", 0, 80, 30)
    sibsp = st.sidebar.slider("Siblings/Spouses Aboard", 0, 5, 0)
    parch = st.sidebar.slider("Parents/Children Aboard", 0, 5, 0)
    fare = st.sidebar.slider("Fare Paid", 0.0, 500.0, 50.0)
    embarked = st.sidebar.selectbox("Port of Embarkation", df_clean['Embarked'].unique())

    # Preprocess user input
    sex_encoded = le_sex.transform([sex])[0]
    embarked_encoded = embarked  # Already encoded after cleaning

    user_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex_encoded],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked_encoded]
    })

    # Train model
    X = df_clean.drop('Survived', axis=1)
    y = df_clean['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Accuracy
    acc = accuracy_score(y_test, model.predict(X_test))
    st.sidebar.markdown(f"✅ **Model Accuracy:** {acc:.2%}")

    # Prediction
    if st.sidebar.button("Predict"):
        prediction = model.predict(user_data)[0]
        result = "🟢 Survived" if prediction == 1 else "🔴 Did Not Survive"
        st.sidebar.markdown(f"### Prediction: {result}")

    # Visualizations
    st.subheader("📊 Data Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Survival Count")
        fig1, ax1 = plt.subplots()
        sns.countplot(data=df, x='Survived', hue='Sex', ax=ax1)
        ax1.set_xticklabels(['Died', 'Survived'])
        st.pyplot(fig1)

    with col2:
        st.markdown("### Age Distribution")
        fig2, ax2 = plt.subplots()
        sns.histplot(data=df, x='Age', bins=30, kde=True, ax=ax2)
        st.pyplot(fig2)

    st.markdown("### Class Distribution")
    fig3, ax3 = plt.subplots()
    sns.countplot(data=df, x='Pclass', hue='Survived', ax=ax3)
    st.pyplot(fig3)

    st.markdown("---")
    st.markdown("🔧 Built with Streamlit & Scikit-learn | Dataset: Titanic CSV")

# ------------------ BANK CHURN PREDICTOR SECTION ------------------
elif app_mode == "Bank Churn Predictor":
    st.title("🏦 Bank Customer Churn Prediction")
    st.write("This app predicts whether a customer will leave the bank based on various features.")

    # Load data
    df = load_churn_data()
    if df is None:
        st.stop()

    # Drop unnecessary columns
    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

    # Encode categorical variables
    le_gender = LabelEncoder()
    le_geo = LabelEncoder()
    df['Gender'] = le_gender.fit_transform(df['Gender'])
    df['Geography'] = le_geo.fit_transform(df['Geography'])

    # Features & Target
    X = df.drop('Exited', axis=1)
    y = df['Exited']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    st.metric("Model Accuracy", f"{acc*100:.2f}%")

    st.header("📥 Predict Customer Churn")

    # Input form
    geography = st.selectbox("Geography", le_geo.classes_)
    gender = st.selectbox("Gender", le_gender.classes_)
    credit_score = st.slider("Credit Score", 300, 900, 650)
    age = st.slider("Age", 18, 92, 40)
    tenure = st.slider("Tenure (Years)", 0, 10, 3)
    balance = st.number_input("Account Balance", 0.0, 300000.0, 50000.0, step=1000.0)
    num_of_products = st.slider("Number of Products", 1, 4, 1)
    has_cr_card = st.selectbox("Has Credit Card?", [0, 1])
    is_active_member = st.selectbox("Is Active Member?", [0, 1])
    estimated_salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0, step=1000.0)

    # Prediction
    if st.button("Predict"):
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Geography': [le_geo.transform([geography])[0]],
            'Gender': [le_gender.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary]
        })

        prediction = model.predict(input_data)[0]
        result = "🔴 Customer is likely to leave." if prediction == 1 else "🟢 Customer is likely to stay."
        st.subheader("Prediction Result")
        st.success(result if prediction == 0 else result)

    # Show data visualization
    with st.expander("📊 View Data Visualizations"):
        st.subheader("Churn Distribution by Features")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Age distribution
        sns.histplot(data=df, x='Age', hue='Exited', multiple='stack', bins=30, ax=axes[0, 0])
        axes[0, 0].set_title('Age Distribution by Churn')
        
        # Geography distribution
        sns.countplot(data=df, x='Geography', hue='Exited', ax=axes[0, 1])
        axes[0, 1].set_title('Geography Distribution by Churn')
        
        # Gender distribution
        sns.countplot(data=df, x='Gender', hue='Exited', ax=axes[1, 0])
        axes[1, 0].set_title('Gender Distribution by Churn')
        axes[1, 0].set_xticklabels(le_gender.classes_)
        
        # Balance distribution
        sns.boxplot(data=df, x='Exited', y='Balance', ax=axes[1, 1])
        axes[1, 1].set_title('Balance Distribution by Churn')
        axes[1, 1].set_xticklabels(['Stayed', 'Left'])
        
        plt.tight_layout()
        st.pyplot(fig)

# ------------------ SYSTEM INFO SECTION ------------------
elif app_mode == "System Info":
    st.title("💻 System Resource Monitor")
    
    # Get system information
    ram_info = get_ram_info()
    cpu_usage = psutil.cpu_percent()
    disk_usage = psutil.disk_usage('/')
    boot_time = psutil.boot_time()
    
    # Display metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total RAM", f"{ram_info['total']:.2f} GB")
        st.metric("Available RAM", f"{ram_info['available']:.2f} GB")
        st.progress(ram_info['usage'] / 100)
        st.caption(f"RAM Usage: {ram_info['usage']:.1f}%")
    
    with col2:
        st.metric("CPU Usage", f"{cpu_usage}%")
        st.progress(cpu_usage / 100)
        st.metric("System Uptime", time.strftime("%H:%M:%S", time.gmtime(time.time() - boot_time)))
    
    with col3:
        st.metric("Total Disk Space", f"{disk_usage.total / (1024**3):.2f} GB")
        st.metric("Used Disk Space", f"{disk_usage.used / (1024**3):.2f} GB")
        st.progress(disk_usage.percent / 100)
        st.caption(f"Disk Usage: {disk_usage.percent:.1f}%")
    
    # Additional system information
    with st.expander("Detailed System Information"):
        st.subheader("CPU Information")
        st.write(f"Physical Cores: {psutil.cpu_count(logical=False)}")
        st.write(f"Logical Cores: {psutil.cpu_count(logical=True)}")
        st.write(f"Current Frequency: {psutil.cpu_freq().current:.2f} MHz")
        
        st.subheader("Network Information")
        net_io = psutil.net_io_counters()
        st.write(f"Bytes Sent: {net_io.bytes_sent / (1024**2):.2f} MB")
        st.write(f"Bytes Received: {net_io.bytes_recv / (1024**2):.2f} MB")
        
        st.subheader("Running Processes")
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'username']):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        st.dataframe(pd.DataFrame(processes).head(20))