import os
import json
import google.generativeai as genai
from PyPDF2 import PdfReader
from flask import Flask, request, jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from flask_cors import CORS
from dotenv import load_dotenv

# Load API Key from .env file
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# Initialize Google AI Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=api_key)

# Function to extract text from uploaded PDFs
def load_pdfs(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

# Function to create FAISS vector database
def create_vector_db(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents([text])
    
    # Store embeddings in FAISS
    vector_db = FAISS.from_documents(docs, embeddings)
    return vector_db

# Function to generate question paper using Gemini AI
def generate_question_paper(params, vector_db):
    retriever = vector_db.as_retriever()
    docs = retriever.get_relevant_documents(params["chapters"])

    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7, api_key=api_key)
    
    # Define dynamic prompt template
    prompt_template = PromptTemplate.from_template(f"""
        You are an AI that generates question papers. Create a structured question paper based on the provided details:
        
        School Name: {params['school_name']}
        Exam Name: {params['exam_name']}
        Chapters: {params['chapters']}
        Difficulty Level: {params['difficulty_level']}
        Study Medium: {params['study_medium']}
        Time Duration: {params['time_duration']} minutes
        Total Marks: {params['total_marks']}
        Additional Requirements: {params['additional_requirements']}
        
        Study Material:
        {{context}}
        
        Generate a well-structured question paper.
    """)

    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt_template)
    response = chain.run(input_documents=docs, question=params["chapters"])
    return response

# Flask API Endpoint
@app.route("/generate-paper", methods=["POST"])
def generate_paper():
    try:
        data = request.form  # Get form data from the app
        files = request.files.getlist("pdf_files")  # Uploaded PDFs
        
        extracted_text = load_pdfs(files)
        vector_db = create_vector_db(extracted_text)

        # Get required parameters
        params = {
            "school_name": data.get("school_name", "Unknown School"),
            "exam_name": data.get("exam_name", "Unknown Exam"),
            "chapters": data.get("chapters", "General"),
            "difficulty_level": data.get("difficulty_level", "Medium"),
            "study_medium": data.get("study_medium", "English"),
            "time_duration": data.get("time_duration", "60"),
            "total_marks": data.get("total_marks", "100"),
            "additional_requirements": data.get("additional_requirements", "None"),
        }

        # Generate question paper
        question_paper = generate_question_paper(params, vector_db)
        
        return jsonify({"success": True, "question_paper": question_paper})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
