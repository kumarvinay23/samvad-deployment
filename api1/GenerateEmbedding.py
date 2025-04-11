import os
import uuid

from langchain.schema import Document
from flask import Flask, request, jsonify
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.schema import Document as DocxDocument
from werkzeug.utils import secure_filename
import PyPDF2
import tiktoken
from flask_cors import CORS, cross_origin


from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Environment variables (set these!)
OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")
MONGODB_ATLAS_URI= os.environ.get("MONGODB_ATLAS_URI", default=None)
MONGODB_DATABASE_NAME = os.environ.get("MONGODB_DATABASE_NAME", default=None)
MONGODB_COLLECTION_NAME = os.environ.get("MONGODB_COLLECTION_NAME", default=None)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")


# Data Source (Singleton Pattern)
class VectorDBDataSource:
    _instance = None

    def __new__(cls, openai_api_key, mongodb_atlas_uri, mongodb_database_name, mongodb_collection_name):
        if cls._instance is None:
            cls._instance = super(VectorDBDataSource, cls).__new__(cls)
            cls._instance.initialize(openai_api_key, mongodb_atlas_uri, mongodb_database_name, mongodb_collection_name)
        return cls._instance

    def initialize(self, openai_api_key, mongodb_atlas_uri, mongodb_database_name, mongodb_collection_name):
        client = MongoClient(mongodb_atlas_uri)
        db = client[mongodb_database_name]
        collection = db[mongodb_collection_name]
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key,model ="text-embedding-ada-002")
        self.vector_db = MongoDBAtlasVectorSearch(
            collection, embeddings, index_name="openai_ada_002_vector_index"
        )

    def get_vector_db(self):
        return self.vector_db

# Initialize Data Source
vector_db_source = VectorDBDataSource(OPENAI_API_KEY, MONGODB_ATLAS_URI, MONGODB_DATABASE_NAME, MONGODB_COLLECTION_NAME)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_extension(filename):
    """
    Extracts the file extension from a filename.

    Args:
        filename (str): The filename.

    Returns:
        str or None: The file extension (without the leading dot), or None if no extension is found.
    """
    try:
        _, extension = os.path.splitext(filename)
        if extension:
            return extension[1:].lower()  # Remove the dot and convert to lowercase
        else:
            return None  # No extension found
    except Exception:
        return None # in case of some strange error.



def chunk_document(document_content, chunk_size=500, chunk_overlap=100):
    tokens = tokenizer.encode(document_content)
    chunks = []
    for i in range(0, len(tokens), chunk_size - chunk_overlap):
        chunk = tokens[i:i + chunk_size]
        chunks.append(tokenizer.decode(chunk))
    return chunks
def generate_unique_id():
    return str(uuid.uuid4())

def process_document(filepath, file_type):
    if file_type == "txt":
        with open(filepath, "r", encoding="utf-8") as f:
            document_content = f.read()
    elif file_type == "pdf":
        with open(filepath, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            document_content = "".join(page.extract_text() for page in reader.pages)
    elif file_type == "docx":
        doc = DocxDocument(filepath)
        document_content = "\n".join(paragraph.text for paragraph in doc.paragraphs)
    else:
        raise ValueError("Unsupported file type")

    chunks = chunk_document(document_content)
    documents = []
    for i, chunk in enumerate(chunks):
        document = Document(
            page_content=chunk,
            metadata={
                "source": os.path.basename(filepath),
                "chunk": i,
            },
        )
        documents.append(document)

    for doc in documents:
        if 'id' not in doc.metadata:  # Check metadata
            doc.metadata['id'] = generate_unique_id() # Modify metadata dictionary.
    return documents

@app.route('/upload', methods=['POST'])
@cross_origin()
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            file_type = get_file_extension(filename)
            documents =process_document(filepath, file_type)
            vector_db_source.get_vector_db().add_documents(documents)

            return jsonify({'message': 'File uploaded and embeddings created successfully'}), 201

        except UnicodeDecodeError:
            return jsonify({'error': 'Binary files are not yet supported in this example.'}), 400
        except Exception as e:
            return jsonify({'error': f'Error processing file: {e}'}), 500

    else:
        return jsonify({'error': 'Invalid file type'}), 400



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

