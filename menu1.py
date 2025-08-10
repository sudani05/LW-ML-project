import streamlit as st
import subprocess
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Unified Dashboard", layout="centered")
st.sidebar.title("Navigation")

main_choice = st.sidebar.selectbox(
    "Choose Category",
    [
        "Machine Learning", "Windows", "Python",
        "DevOps", "Cloud", "Agentic AI", "Full-Stack"
    ]
)

# Sub-choice selection for relevant domains
sub_choice = None
if main_choice == "DevOps":
    sub_choice = st.sidebar.selectbox("DevOps Tools", ["Jenkins", "Docker", "Kubernetes"])
elif main_choice == "Cloud":
    sub_choice = st.sidebar.selectbox("Cloud Services", ["AWS EC2", "S3 Buckets", "IAM Roles"])
elif main_choice == "Agentic AI":
    sub_choice = st.sidebar.selectbox("AI Projects", ["Chatbot", "Emotion Detector", "Image Classifier"])
elif main_choice == "Full-Stack":
    sub_choice = st.sidebar.selectbox("Stack Components", ["HTML/CSS", "React/Node", "Database"])
elif main_choice == "Python":
    sub_choice = st.sidebar.selectbox(
        "Automating Tasks",
        [
            "Send Whatsap msg", "Send E-mail", "Send Text msg", "Make a Phone call",
            "Post on Linkedin", "Post on Twitter", "Post on Facebook", "Post on Instagram", "Send watsap image"
        ]
    )

# =========================
# MACHINE LEARNING SECTION
# =========================
def machine_learning_section():
    st.header("🧠 Machine Learning Models")
    ml_model = st.selectbox("Select ML Model", [
        "Machine Learning Model-1",
        "Machine Learning Model-2",
        "Machine Learning Model-3",
        "Machine Learning Model-4",
        "Machine Learning Model-5"
    ])

    if ml_model == "Machine Learning Model-1":
        st.subheader("Salary Prediction using Linear Regression")
        try:
            dataset = pd.read_csv("Salary_dataset.csv")
            x = dataset["YearsExperience"].values.reshape(-1, 1)
            y = dataset["Salary"].values.reshape(-1, 1)
            model = LinearRegression()
            model.fit(x, y)
            years_exp = st.number_input("Enter Years of Experience:", min_value=0.0, max_value=50.0, step=0.1)
            if st.button("Predict Salary"):
                prediction = model.predict([[years_exp]])
                st.success(f"Predicted Salary for {years_exp} years experience: ₹ {prediction[0][0]:,.2f}")
                st.info(f"Slope (coefficient): {model.coef_[0][0]:.2f}")
                st.info(f"Intercept (bias): {model.intercept_[0]:.2f}")
        except Exception as e:
            st.error(f"Error: {e}")

    elif ml_model == "Machine Learning Model-2":
        st.subheader("Startup Profit Prediction using Multiple Linear Regression")
        try:
            dataset = pd.read_csv("50_Startups.csv")
            dataset = pd.get_dummies(dataset, columns=['State'], drop_first=True)
            x = dataset[['R&D Spend', 'Administration', 'Marketing Spend', 'State_Florida', 'State_New York']]
            y = dataset['Profit']
            model = LinearRegression()
            model.fit(x, y)
            st.markdown("### Enter Startup Investment Details")
            rd_spend = st.number_input("R&D Spend (₹)", min_value=0.0, step=1000.0, format="%.2f")
            admin = st.number_input("Administration Spend (₹)", min_value=0.0, step=1000.0, format="%.2f")
            marketing = st.number_input("Marketing Spend (₹)", min_value=0.0, step=1000.0, format="%.2f")
            state = st.selectbox("State", ["California", "Florida", "New York"])
            state_florida = 1 if state == "Florida" else 0
            state_newyork = 1 if state == "New York" else 0
            if st.button("Predict Profit"):
                input_data = [[rd_spend, admin, marketing, state_florida, state_newyork]]
                prediction = model.predict(input_data)
                st.success(f"Predicted Profit: ₹ {prediction[0]:,.2f}")
                st.info(f"Slope (coefficients): {model.coef_}")
                st.info(f"Intercept (bias): {model.intercept_:.2f}")
        except Exception as e:
            st.error(f"Error: {e}")

    else:
        st.info(f"{ml_model} is under construction 🚧.")

# =========================
# WINDOWS SECTION
# =========================
def windows_section():
    st.header("🪟 Windows Command Executor (Local)")
    user_input = st.text_input("Enter Windows command (e.g. dir, ipconfig, whoami, cd, systeminfo)")
    if st.button("Run Windows Command"):
        if user_input.strip() == "":
            st.warning("Please enter a command.")
        else:
            try:
                result = subprocess.getoutput(user_input)
                st.code(result, language="bat")
            except Exception as e:
                st.error(f"Error: {e}")

# =========================
# PYTHON AUTOMATION SECTION
# =========================
def python_automation_section():
    st.header("🐍 Python Automation")
    if sub_choice == "Send Whatsap msg":
        st.subheader("Send WhatsApp Message")
        phone = st.text_input("Enter phone number with country code (e.g. +919876543210)")
        message = st.text_area("Enter your message")
        hour = st.number_input("Hour (24h format)", min_value=0, max_value=23)
        minute = st.number_input("Minute", min_value=0, max_value=59)
        if st.button("Send Message"):
            import pywhatkit
            try:
                pywhatkit.sendwhatmsg(phone, message, int(hour), int(minute))
                st.success("WhatsApp message scheduled successfully!")
            except Exception as e:
                st.error(f"Error: {e}")

    elif sub_choice == "Send E-mail":
        st.subheader("Send Email")
        sender = st.text_input("Sender Email")
        password = st.text_input("Email Password", type="password")
        receiver = st.text_input("Receiver Email")
        subject = st.text_input("Subject")
        body = st.text_area("Email Body")
        if st.button("Send Email"):
            import smtplib
            from email.message import EmailMessage
            try:
                msg = EmailMessage()
                msg.set_content(body)
                msg['Subject'] = subject
                msg['From'] = sender
                msg['To'] = receiver

                with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                    smtp.login(sender, password)
                    smtp.send_message(msg)
                st.success("Email sent successfully!")
            except Exception as e:
                st.error(f"Error: {e}")

    elif sub_choice == "Send Text msg":
        st.subheader("Send SMS (Text Message)")
        st.warning("Requires Twilio account and credentials")
        account_sid = st.text_input("Twilio Account SID")
        auth_token = st.text_input("Auth Token", type="password")
        from_number = st.text_input("Twilio Phone Number (e.g., +12025550123)")
        to_number = st.text_input("Recipient Number (e.g., +919876543210)")
        message = st.text_area("Message")
        if st.button("Send SMS"):
            try:
                from twilio.rest import Client
                client = Client(account_sid, auth_token)
                client.messages.create(body=message, from_=from_number, to=to_number)
                st.success("SMS sent successfully!")
            except Exception as e:
                st.error(f"Error: {e}")

    elif sub_choice == "Make a Phone call":
        st.subheader("Make a Phone Call (Twilio)")
        st.warning("Requires Twilio account and verified phone number")
        account_sid = st.text_input("Twilio Account SID")
        auth_token = st.text_input("Auth Token", type="password")
        from_number = st.text_input("Twilio Number")
        to_number = st.text_input("Your Number")
        call_url = st.text_input("URL with TwiML instructions", value="http://demo.twilio.com/docs/voice.xml")
        if st.button("Call Now"):
            try:
                from twilio.rest import Client
                client = Client(account_sid, auth_token)
                call = client.calls.create(url=call_url, to=to_number, from_=from_number)
                st.success("Call initiated successfully!")
            except Exception as e:
                st.error(f"Error: {e}")

    elif sub_choice == "Post on Linkedin":
        st.subheader("LinkedIn Posting (Manual)")
        st.info("LinkedIn API requires approval. For now, copy-paste content manually.")
        post = st.text_area("Post Content")
        if st.button("Generate Preview"):
            st.success("Here's your post:")
            st.write(post)

    elif sub_choice == "Post on Twitter":
        st.subheader("Tweet Something")
        tweet = st.text_area("Compose your tweet")
        st.warning("Requires Twitter API credentials")
        if st.button("Post Tweet"):
            try:
                import tweepy
                api_key = st.text_input("API Key")
                api_secret = st.text_input("API Secret", type="password")
                access_token = st.text_input("Access Token")
                access_secret = st.text_input("Access Token Secret", type="password")
                auth = tweepy.OAuthHandler(api_key, api_secret)
                auth.set_access_token(access_token, access_secret)
                api = tweepy.API(auth)
                api.update_status(tweet)
                st.success("✅ Tweet posted!")
            except Exception as e:
                st.error(f"Error: {e}")

    elif sub_choice == "Post on Facebook":
        st.subheader("Facebook Post")
        st.warning("Facebook API requires Graph API token and app review.")
        post = st.text_area("What’s on your mind?")
        if st.button("Preview"):
            st.success("Preview:")
            st.write(post)

    elif sub_choice == "Post on Instagram":
        st.subheader("Instagram Post (Manual/Third-Party Tool Recommended)")
        st.warning("Instagram posting via API is restricted. Use tools like Instabot or Meta Graph API.")
        caption = st.text_area("Write your caption")
        if st.button("Preview"):
            st.info("Caption Preview:")
            st.write(caption)

    elif sub_choice == "Send watsap image":
        st.subheader("Send WhatsApp Image")
        phone = st.text_input("Phone Number (with country code)")
        image_path = st.text_input("Image Path (full path on system)")
        caption = st.text_input("Caption")
        time_hour = st.number_input("Hour", min_value=0, max_value=23)
        time_minute = st.number_input("Minute", min_value=0, max_value=59)
        if st.button("Send Image"):
            try:
                import pywhatkit
                pywhatkit.sendwhats_image(phone, image_path, caption, time_hour, time_minute)
                st.success("Image sent successfully via WhatsApp!")
            except Exception as e:
                st.error(f"Error: {e}")

    else:
        st.error("Unknown Python Task.")

# =========================
# DEVOPS SECTION
# =========================
def devops_section():
    st.header("⚙ DevOps Tools")
    st.info(f"Selected: {sub_choice}")
    st.write("This is a placeholder for DevOps tools integration.")
    st.write("You can add Jenkins, Docker, or Kubernetes controls here.")

# =========================
# CLOUD SECTION
# =========================
def cloud_section():
    st.header("☁ Cloud Services")
    st.info(f"Selected: {sub_choice}")
    st.write("This is a placeholder for Cloud services integration.")
    st.write("You can add AWS EC2, S3, or IAM controls here.")

# =========================
# AGENTIC AI SECTION
# =========================
def agentic_ai_section():
    st.header("🤖 Agentic AI")
    st.info(f"Selected: {sub_choice}")
    chatbot_option = st.selectbox("Select Chatbot Version", [
        "Chatbot 1 - Gemini Finance Bot",
        "Chatbot 2 - FAQ Bot",
        "Chatbot 3 - Sentiment-Aware",
        "Chatbot 4 - Generative AI"
    ])

    if chatbot_option == "Chatbot 1 - Gemini Finance Bot":
        st.markdown("### Ask me anything about *Stock Market & Finance*")
        import google.generativeai as genai
        genai.configure(api_key="YOUR_GEMINI_API_KEY")
        model = genai.GenerativeModel("models/gemini-1.5-flash")

        def finance_bot(question):
            try:
                prompt = f"You are a finance education assistant. Explain this in simple words:\n\n{question}"
                response = model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                return f"Error: {str(e)}"

        question = st.text_input("Enter your finance-related question", placeholder="e.g., What is SIP?")
        if st.button("Get Answer"):
            if question.strip():
                answer = finance_bot(question)
                st.success("Gemini's Answer:")
                st.write(answer)
            else:
                st.warning("Please enter a question first.")

        st.markdown("#### Try an example:")
        example = st.selectbox("Example Questions", [
            "",
            "What is SIP?",
            "Difference between stocks and mutual funds",
            "Explain compound interest",
            "What is Nifty 50?",
            "Best books to learn investing"
        ])
        if example:
            st.info(f"Gemini's Answer to: {example}")
            st.write(finance_bot(example))

    elif chatbot_option == "Chatbot 2 - FAQ Bot":
        st.write("Chatbot 2: Answering FAQs... (Coming soon)")

    elif chatbot_option == "Chatbot 3 - Sentiment-Aware":
        st.write("Chatbot 3: Detecting your emotion and responding accordingly... (Coming soon)")

    elif chatbot_option == "Chatbot 4 - Generative AI":
        st.write("Chatbot 4: Generating AI-based dynamic response... (Coming soon)")

    else:
        st.error("Invalid chatbot selected.")

# =========================
# FULL-STACK SECTION
# =========================
def full_stack_section():
    st.header("🖥 Full-Stack Components")
    st.info(f"Selected: {sub_choice}")
    st.write("This is a placeholder for Full-Stack components integration.")
    st.write("You can add HTML/CSS, React/Node, or Database controls here.")

# =========================
# MAIN ROUTER
# =========================
if main_choice == "Machine Learning":
    machine_learning_section()
elif main_choice == "Windows":
    windows_section()
elif main_choice == "Python":
    python_automation_section()
elif main_choice == "DevOps":
    devops_section()
elif main_choice == "Cloud":
    cloud_section()
elif main_choice == "Agentic AI":
    agentic_ai_section()
elif main_choice == "Full-Stack":
    full_stack_section()
else:
    st.write("Please select a category from the sidebar.")