from langchain.memory import ConversationBufferMemory
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from pymongo import MongoClient
from transformers import pipeline
from googletrans import Translator
import speech_recognition as sr
import os
import io
import requests
from pydub import AudioSegment
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
WEATHER_API_KEY = os.environ.get("YOUR_WEATHER_API_KEY")  # Replace with your weather API key
MONGODB_ATLAS_URI = os.environ.get("MONGODB_ATLAS_URI", default=None)
MONGODB_DATABASE_NAME = os.environ.get("MONGODB_DATABASE_NAME", default=None)
MONGODB_COLLECTION_NAME = os.environ.get("MONGODB_COLLECTION_NAME", default=None)


# --- Initialize Components ---
llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
translator = Translator()
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Initialize the memory


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

user_memory = {}

# Initialize Data Source
vector_db_source = VectorDBDataSource(OPENAI_API_KEY, MONGODB_ATLAS_URI, MONGODB_DATABASE_NAME, MONGODB_COLLECTION_NAME)

def get_or_create_memory(user_id: str) -> ConversationBufferMemory:
    if user_id not in user_memory:
        print(f"Creating new memory for user: {user_id}")
        user_memory[user_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True, human_prefix="User", ai_prefix="Assistant")
    return user_memory[user_id]


def get_weather(city: str, chat_history: str, user_input: str) -> str:
    """Useful for getting the current weather in a city."""
    try:
        response = requests.get(f"https://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}")
        response.raise_for_status()
        weather_data = response.json()
        return f"The current temperature in {city} is {weather_data['current']['temp_c']}°C. Previous conversation: {chat_history}. User input: {user_input}"
    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data: {e}"
    except KeyError:
        return "Could not parse weather information from the API."

def search_knowledge_base(query: str, chat_history: str, user_input: str) -> str:
    """Useful for answering questions about Study material. Use this to find information from the knowledge base."""
    try:
        docs = vector_db_source.get_vector_db().similarity_search(query)
        if not docs:
            return "No relevant information found in the knowledge base."
        context = "\n".join([doc.page_content for doc in docs])
        prompt = ChatPromptTemplate.from_template(
            """
            Answer the following question based on the provided context.

            Context:
            {context}

            Previous conversation:
            {chat_history}

            User input:
            {user_input}

            Question: {question}
            Answer:
            """
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return llm_chain.invoke({"context": context, "chat_history": chat_history, "user_input": user_input, "question": query})['text']
    except Exception as e:
        return f"Error searching knowledge base: {e}"

# --- Main Chatbot Logic ---

async def process_voice_input(audio_data):
    """Converts audio to text using Speech Recognition."""
    try:
        r = sr.Recognizer()
        audio_stream = io.BytesIO(audio_data)
        audio = sr.AudioFile(audio_stream)
        with audio as source:
            r.adjust_for_ambient_noise(source)
            audio = r.record(source)
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Could not understand audio."
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"



from quart import Quart, request, jsonify
from quart_cors import cors

app = Quart(__name__)
app = cors(app)

# --- Main Chatbot Logic ---

@app.route('/samvad', methods=['POST'])
async def chatbot_response():
    try:
        data = await request.get_json()
        user_input_raw = data['user_input']
        voice_input = data.get('voice_input', None)
        detected_language = 'en'
        english_input = user_input_raw
        user_id = request.headers.get('user-id')

        if voice_input:
            user_input_text = await process_voice_input(voice_input)
            if "Could not understand audio." in user_input_text or "Could not request results" in user_input_text:
                return jsonify({"response": user_input_text, "language": "en"})
            user_input = user_input_text
        else:
            user_input = user_input_raw

        # Detect language and translate to English if necessary
        try:
            detected_language_obj = await translator.detect(user_input)
            detected_language = detected_language_obj.lang
            if detected_language != 'en':
                translates_english_input = await translator.translate(user_input, dest='en')
                english_input = translates_english_input.text
                print(f"Translated input ({detected_language}): {english_input}")
        except Exception as e:
            print(f"Language detection/translation error: {e}")

        # Retrieve chat history from memory
        memory = get_or_create_memory(user_id)
        chat_history = memory.load_memory_variables(user_id)

        # Decide whether to use a tool or general conversation
        classification_result = classifier(english_input, ["weather", "knowledge", "general conversation"])
        predicted_label = classification_result['labels'][0]
        print(f"Predicted Label: {predicted_label}")

        if predicted_label == "weather":
            words = english_input.split()
            city = words[-1] if words else "London"
            response_text = get_weather(city, chat_history, user_input)
        elif predicted_label == "knowledge":
            response_text = search_knowledge_base(english_input, chat_history, user_input)
        elif predicted_label == "general conversation":
            response_text = "I'm not sure how to respond to that."
        else:
            response_text = "I'm not sure how to respond to that."

        # Save the conversation to memory
        memory.save_context({"input": user_input}, {"output": response_text})

        # Translate the response back to the user's language
        translated_response = response_text
        if detected_language != 'en' and "Error" not in response_text:
            try:
                translated_response_obj = await translator.translate(response_text, dest=detected_language)
                translated_response = translated_response_obj.text
            except Exception as e:
                print(f"Translation back to {detected_language} error: {e}")

        return jsonify({"response": translated_response, "language": detected_language})

    except Exception as e:
        return jsonify({"response": f"An error occurred: {e}", "language": "en"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)


