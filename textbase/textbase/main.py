from textbase import bot, Message
from textbase.models import OpenAI
from typing import List
import PyPDF2
import docx2txt
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Load your OpenAI API key
OpenAI.api_key = "aecb0e364a723cda3b1fc3dfdb96d1cd55c04a0a6b32cf0dc1939016d50919f0"

# Prompt for GPT-3.5 Turbo
SYSTEM_PROMPT = "Hello, Please enter your resume to check your resume score"

# Function to extract text from PDF or Word documents
def extract_text(file_path):
    if file_path.lower().endswith('.pdf'):
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfFileReader(pdf_file)
            text = ''
            for page_num in range(pdf_reader.numPages):
                text += pdf_reader.getPage(page_num).extractText()
    elif file_path.lower().endswith(('.doc', '.docx')):
        text = docx2txt.process(file_path)
    else:
        raise ValueError("Unsupported file format")

    return text

# Function to calculate cosine similarity between two texts
def calculate_similarity(text1, text2):
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity([vectors[0]], [vectors[1]])[0][0]

# Function to parse and score the resume
def parse_and_score_resume(resume_text, job_description):
    # Tokenize the text using spaCy
    resume_doc = nlp(resume_text)
    job_doc = nlp(job_description)

    # Calculate cosine similarity between the resume and job description
    similarity_score = calculate_similarity(resume_text, job_description)

    return similarity_score

# Initialize the chatbot
@bot()
def on_message(message_history: List[Message], state: dict = None):
    # Extract user messages from the conversation history and check for the role
    user_messages = [message.content for message in message_history if message.role == 'user']
    user_messages_text = ' '.join(user_messages)

    # Check if the user wants to upload a resume
    if "resume" in user_messages_text.lower():
        resume_path = input("Please upload your resume (PDF or Word document): ")
        resume_text = extract_text(resume_path)

        # Get user's job description
        job_description = input("Please enter the job description or job posting: ")

        # Parse and score the resume
        score = parse_and_score_resume(resume_text, job_description)

        # Provide feedback
        bot_response = f"Resume Score: {score}\n"
        if score >= 0.7:
            bot_response += "Congratulations! Your resume is a strong match for this job."
        elif 0.5 <= score < 0.7:
            bot_response += "Your resume has potential, but there is room for improvement."
        else:
            bot_response += "Your resume may need significant improvements to match this job."
    else:
        # Include the user's latest message in the prompt for context
        personalized_prompt = SYSTEM_PROMPT
        if user_messages:
            personalized_prompt += f"User: {user_messages[-1]}\n"

        # Generate GPT-3.5 Turbo response if the user doesn't mention a resume
        bot_response = OpenAI.generate(
            system_prompt=personalized_prompt,
            message_history=message_history,
            model="gpt-3.5-turbo",
        )

    response = {
        "data": {
            "messages": [
                {
                    "data_type": "STRING",
                    "value": bot_response
                }
            ],
            "state": state
        },
        "errors": []
    }

    return {
        "status_code": 200,
        "response": response
    }

if __name__ == "__main__":
    # Initialize the chatbot
    my_chatbot = on_message(None)

    # Start the conversation loop
    while True:
        user_input = input("You: ")
        user_message = Message(content=user_input, role="user")
        my_chatbot.send(user_message)

        # Get and print the bot's response
        bot_response = my_chatbot.receive()
        print("Bot:", bot_response['response']['data']['messages'][0]['value'])
