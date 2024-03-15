import shutil
import tempfile
import zipfile
import os
import csv
import PyPDF2
import docx2txt
import re
import pandas as pd
import spacy
from spacy.matcher import Matcher
from geotext import GeoText
from fastapi import FastAPI, Response
from fastapi import FastAPI, UploadFile, File

app = FastAPI()

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Initialize the matcher
matcher = Matcher(nlp.vocab)

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    extracted_text = ""

    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        total_pages = len(reader.pages)

        for page_number in range(total_pages):
            page = reader.pages[page_number]
            text = page.extract_text()
            extracted_text += text

    return extracted_text


# Function to extract text from Word document
def extract_text_from_docx(file_path):
    extracted_text = docx2txt.process(file_path)
    return extracted_text


# Function to extract email addresses from text
def get_email_addresses(string):
    r = re.compile(r'[\w\.-]+@[\w\.-]+')
    return r.findall(string)


# Function to extract phone numbers from text
def get_phone_numbers(string):
    r = re.compile(
        r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')
    phone_numbers = r.findall(string)
    return [re.sub(r'\D', '', num) for num in phone_numbers]


# Function to extract name from text
def extract_name(resume_text):
    nlp_text = nlp(resume_text)

    # First name and Last name are always Proper Nouns
    pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]

    matcher.add('NAME', [pattern], on_match=None)

    matches = matcher(nlp_text)

    for match_id, start, end in matches:
        span = nlp_text[start:end]
        return span.text


# Function to extract skills from text
def extract_skills(resume_text):
    nlp_text = nlp(resume_text)

    # Removing stop words and implementing word tokenization
    tokens = [token.text for token in nlp_text if not token.is_stop]

    skills = pd.read_csv("skills.csv")
    skillset = []

    # Check for one-grams
    for token in tokens:
        if token.lower() in skills:
            skillset.append(token)

    # Check for bi-grams and tri-grams
    for token in nlp_text.noun_chunks:
        token = token.text.lower().strip()
        if token in skills:
            skillset.append(token)

    return list(set(skillset))


# Function to extract location from text
def extract_location(resume_text):
    places = GeoText(resume_text)
    locations = places.cities

    return list(set(locations))


# Function to extract education degrees from text
EDUCATION = [
    'bachelor', 'bachelors', 'bsc', 'b.sc', 'b.a.', 'bachelors degree',
    'master', 'masters', 'msc', 'm.sc', 'm.a.', 'masters degree',
    'phd', 'doctorate', 'ph.d.', 'doctoral degree',
    'associate', 'associate degree',
    'diploma', 'advanced diploma', 'postgraduate diploma',
    'certificate', 'professional certificate', 'vocational certificate',
    'honours', 'dual degree', 'integrated degree', 'executive degree',
    'licentiate', 'specialist',
    'undergraduate', 'graduate', 'ssc', 'hsc', 'cbse', 'icse', 'isc'
]

STOPWORDS = set(spacy.lang.en.stop_words.STOP_WORDS)

def extract_education(resume_text):
    doc = nlp(resume_text)
    education = []

    for sent in doc.sents:
        sent_text = sent.text.lower()
        for edu in EDUCATION:
            if edu in sent_text and edu not in STOPWORDS:
                year_match = re.search(r'((20|19)\d{2})', sent_text)
                if year_match:
                    education.append((edu, year_match.group()))
                else:
                    education.append(edu)

    return education


# Function to extract educational institutes from text
def extract_educational_institutes(resume_text):
    sub_patterns = [
        '[A-Z][a-z]* University', '[A-Z][a-z]* Educational Institute',
        'University of [A-Z][a-z]*', 'Ecole [A-Z][a-z]*'
    ]
    pattern = '({})'.format('|'.join(sub_patterns))
    matches = re.findall(pattern, resume_text)
    return matches


# Function to extract experience from text
def extract_experience(resume_text):
    nlp_text = nlp(resume_text)
    experience = []

    matcher.add('EXPERIENCE', None, [{'POS': 'NOUN'}, {'LOWER': 'experience'}])

    matches = matcher(nlp_text)

    for match_id, start, end in matches:
        span = nlp_text[start:end]
        experience.append(span.text)

    return experience


# Function to process resumes and extract information
def process_resumes(resumes_folder):
    output_dict = {}

    for filename in os.listdir(resumes_folder):
        file_path = os.path.join(resumes_folder, filename)

        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        elif filename.endswith(".docx"):
            text = extract_text_from_docx(file_path)
        else:
            print(f"Invalid file: {filename}")
            continue

        email = get_email_addresses(text)
        phone_number = get_phone_numbers(text)
        name = extract_name(text)
        skills = extract_skills(text)
        locations = extract_location(text)
        education = extract_education(text)
        institutes = extract_educational_institutes(text)
        experience = extract_experience(text)

        resume_info = {
            "Email": email,
            "Phone Number": phone_number,
            "Name": name,
            "Skills": skills,
            "Locations": locations,
            "Degree": education,
            "Educational Institute": institutes,
            "Experience": experience
        }

        output_dict[filename] = resume_info

    return output_dict


@app.get("/")
def welcome():
    return {"message": "This is Resume Parsing"}


# FastAPI endpoint for processing resumes
@app.post("/process_resumes")
def process_resumes_endpoint(resumes_folder: UploadFile = File(...)):
    # Create a temporary directory to extract the zip folder
    temp_directory = tempfile.mkdtemp()

    try:
        # Save the uploaded zip file to the temporary directory
        zip_file_path = os.path.join(temp_directory, resumes_folder.filename)
        with open(zip_file_path, "wb") as file:
            file.write(resumes_folder.file.read())

        # Extract the zip file
        with zipfile.ZipFile(zip_file_path, "r") as zip_file:
            zip_file.extractall(temp_directory)

        # Process the resumes and extract information
        output_dict = process_resumes(temp_directory)

        # Write the extracted information to a CSV file
        csv_file_path = os.path.join(temp_directory, "data.csv")
        dump_dict_to_csv(list(output_dict.values()), csv_file_path)

        # Read the CSV file
        with open(csv_file_path, "rb") as file:
            csv_content = file.read()

        # Remove the temporary directory
        shutil.rmtree(temp_directory)

        return Response(content=csv_content, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=data.csv"})
    
    except Exception as e:
        return {"message": f"Error processing resumes: {str(e)}"}


# Function to write a list of dictionaries to a CSV file
def dump_dict_to_csv(dictionary_list, csv_file):
    fieldnames = ['Name', 'Email', 'Phone Number', 'Skills', 'Locations', 'Degree', 'Educational Institute',
                  'Experience']
    
    for dictionary in dictionary_list:
        fieldnames.extend(dictionary.keys())

    fieldnames = list(set(fieldnames))  # Remove duplicates

    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in dictionary_list:
            encoded_row = {k: v.encode('utf-8') if isinstance(v, str) else v for k, v in row.items()}
            writer.writerow(encoded_row)
