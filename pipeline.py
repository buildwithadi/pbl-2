import io
import re
import nltk
from PyPDF2 import PdfReader
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# --- NLTK Data Download ---
# This is necessary for tokenization, stopwords, and lemmatization.
# It's safe to run this multiple times; it will only download if missing.
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpus/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# --- Global Variables ---
# Initialize the tools we need. This is more efficient than
# creating them inside the function on every API call.
STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

# --- Main Public Functions ---

def get_similarity_score(resume_text: str, jd_text: str) -> float:
    """
    Calculates the cosine similarity score between two texts.
    
    This function processes both texts, cleans them, vectorizes them
    using TF-IDF, and then computes the similarity.
    
    Args:
        resume_text: The raw text extracted from the resume.
        jd_text: The raw text from the job description.
        
    Returns:
        A float between 0.0 and 1.0 representing the similarity.
    """
    # 1. Clean both text inputs
    cleaned_resume = _clean_text(resume_text)
    cleaned_jd = _clean_text(jd_text)
    
    # 2. Put texts into a list (corpus) for the vectorizer
    corpus = [cleaned_resume, cleaned_jd]
    
    # 3. Initialize and fit the TF-IDF Vectorizer
    # TfidfVectorizer converts text to a matrix of TF-IDF features
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # 4. Calculate Cosine Similarity
    # This computes the similarity between the first doc (resume) and the second (JD)
    # The result is a 2x2 matrix; we want the value at [0, 1] (or [1, 0])
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Get the similarity score between document 0 and document 1
    score = similarity_matrix[0, 1]
    
    return float(score)


def extract_text_from_file(file, content_type: str) -> str:
    """
    Extracts raw text from an uploaded file (PDF or DOCX).
    
    Args:
        file: The file-like object (e.g., from request.files).
        content_type: The MIME type of the file.
        
    Returns:
        The extracted raw text as a string.
    """
    text = ""
    try:
        if content_type == 'application/pdf':
            # Handle PDF
            # We read the file into memory as bytes
            file_bytes = io.BytesIO(file.read())
            reader = PdfReader(file_bytes)
            for page in reader.pages:
                text += page.extract_text() or ""
                
        elif content_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            # Handle DOCX
            file_bytes = io.BytesIO(file.read())
            document = docx.Document(file_bytes)
            for para in document.paragraphs:
                text += para.text + "\n"
        else:
            # Should be caught by FastAPI, but as a fallback
            return "" 
            
    except Exception as e:
        print(f"Error extracting text: {e}")
        # Return whatever text was extracted, even if partial
        return text 
        
    return text

# --- Internal Helper Functions ---

def _clean_text(text: str) -> str:
    """
    Internal function to clean and pre-process a single string.
    
    1. Lowercase
    2. Remove punctuation/special characters
    3. Tokenize (split into words)
    4. Remove stopwords
    5. Lemmatize (reduce words to root form)
    """
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 3. Tokenize
    tokens = word_tokenize(text)
    
    # 4. Remove stopwords and 5. Lemmatize
    cleaned_tokens = []
    for token in tokens:
        if token not in STOPWORDS and len(token) > 1: # Remove stopwords and single-letter tokens
            cleaned_tokens.append(LEMMATIZER.lemmatize(token))
            
    # Join tokens back into a single string
    return " ".join(cleaned_tokens)