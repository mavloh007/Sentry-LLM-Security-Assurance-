from flask import Flask, render_template, request, jsonify
import os
import sys
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.chatbot.withdrawal_chatbot import WithdrawalChatbot
from src.db.supabase_client import SupabaseDB
from dotenv import load_dotenv, find_dotenv

load_dotenv()
dotenv_path = find_dotenv(".env", usecwd=True)
load_dotenv(dotenv_path, override=True)

app = Flask(__name__)

# Initialize Supabase database connection
try:
    db = SupabaseDB()
    if not db.health_check():
        print("ERROR: Could not connect to Supabase database")
        raise Exception("Supabase connection failed")
    print("✓ Connected to Supabase database")
except Exception as e:
    print(f"ERROR initializing database: {e}")
    raise

# Initialize chatbot with Supabase backend
bot = WithdrawalChatbot(db=db)
print("✓ Withdrawal Chatbot initialized with Supabase backend")

@app.route('/')
def index():
    # Renders the HTML template
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Call the actual chatbot logic
    try:
        response = bot.chat(user_message)
    except Exception as e:
        response = f"Error: {str(e)}"
    
    return jsonify({"response": response})

if __name__ == '__main__':
    # Running Flask in debug mode for development
    app.run(debug=True, port=3000)
