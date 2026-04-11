from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
import sys
from functools import wraps
from uuid import uuid4

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Ensure repo root is on sys.path (important for WSGI deployments where CWD may differ)
project_root = BASE_DIR
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.chatbot.withdrawal_chatbot import WithdrawalChatbot
from src.db.supabase_client import SupabaseDB
from dotenv import load_dotenv, find_dotenv

load_dotenv()
dotenv_path = find_dotenv(".env", usecwd=True)
load_dotenv(dotenv_path, override=True)

templates_dir = os.path.join(BASE_DIR, "frontend", "templates")
static_dir = os.path.join(BASE_DIR, "frontend", "static")

# Keep URL paths stable (/static/...) while hosting files in frontend/static
app = Flask(
    __name__,
    template_folder=templates_dir,
    static_folder=static_dir,
    static_url_path="/static",
)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-in-production")

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

# Initialize chatbot ONCE — reused across all requests
bot = WithdrawalChatbot(db=db)
print("✓ Withdrawal Chatbot initialized (cached instance)")

# ===================================
# Authentication Decorator
# ===================================
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            if request.path.startswith('/api/'):
                return jsonify({"error": "Unauthorized"}), 401
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def _derive_conversation_title(user_message: str) -> str:
    text = (user_message or "").strip()
    if not text:
        return "New Conversation"

    words = text.split()
    title = " ".join(words[:8]).strip()
    if len(words) > 8:
        title += "..."
    return title[:80] if title else "New Conversation"


def _set_active_conversation(user_id: str, conversation_id: str) -> bool:
    conversation = db.get_conversation(conversation_id)
    if not conversation or conversation.get("user_id") != user_id:
        return False
    session["conversation_id"] = conversation_id
    return True

# ===================================
# Auth Routes
# ===================================

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()

        if not email or not password:
            return render_template('register.html', error='Email and password required')

        if password != confirm_password:
            return render_template('register.html', error='Passwords do not match')

        if len(password) < 6:
            return render_template('register.html', error='Password must be at least 6 characters')

        try:
            # Sign up with Supabase Auth
            response = db.client.auth.sign_up({
                "email": email,
                "password": password
            })

            user_id = response.user.id

            # Create user in our users table
            db.create_user(
                user_id=user_id,
                email=email,
                metadata={"signup_method": "email", "created_at": str(uuid4())}
            )

            # Auto-login after signup
            session['user_id'] = user_id
            session['email'] = email
            session.permanent = True

            return redirect(url_for('chat'))

        except Exception as e:
            error_msg = str(e)
            if 'already registered' in error_msg.lower():
                error_msg = 'Email already registered. Please login instead.'
            return render_template('register.html', error=error_msg)

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        if not email or not password:
            return render_template('login.html', error='Email and password required')

        try:
            # Sign in with Supabase Auth
            response = db.client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })

            user_id = response.user.id

            # Store in session
            session['user_id'] = user_id
            session['email'] = email
            session.permanent = True

            return redirect(url_for('chat'))

        except Exception as e:
            return render_template('login.html', error='Invalid email or password')

    return render_template('login.html')


@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return redirect(url_for('login'))


# ===================================
# Chat Routes (Protected)
# ===================================

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('chat'))


@app.route('/chat')
@login_required
def chat():
    user_id = session.get("user_id")
    if user_id and not session.get("conversation_id"):
        conv = db.create_conversation(user_id=user_id, title="Withdrawal Bot Session")
        session["conversation_id"] = conv.get("id")
    return render_template('index.html', email=session.get('email'))


@app.route('/api/chat', methods=['POST'])
@login_required
def send_chat():
    data = request.get_json(silent=True) or {}
    user_message = data.get('message', '')
    requested_title = (data.get('conversation_title') or '').strip()
    force_new = bool(data.get('new_session'))
    user_id = session.get('user_id')
    conversation_id = (data.get("conversation_id") or "").strip()
    if not conversation_id and not force_new:
        conversation_id = session.get("conversation_id")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Switch to a different existing conversation if requested
        if conversation_id and conversation_id != session.get("conversation_id"):
            if not _set_active_conversation(user_id, conversation_id):
                return jsonify({"error": "Conversation not found"}), 404

        # Create a new conversation if needed
        if user_id and not conversation_id:
            conv_title = requested_title or _derive_conversation_title(user_message)
            conv = db.create_conversation(user_id=user_id, title=conv_title)
            conversation_id = conv.get("id")
            session["conversation_id"] = conversation_id

        # Use the cached chatbot — pass per-request identity
        response = bot.chat(user_message, user_id=user_id, conversation_id=conversation_id)

        # Return the conversation list in the same response (avoids a 2nd round-trip)
        conversations = db.get_user_conversations(user_id)

        return jsonify({
            "response": response,
            "conversation_id": conversation_id,
            "conversations": conversations,
            "active_conversation_id": conversation_id,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/conversation-history', methods=['GET'])
@login_required
def get_history():
    """Get conversation history for logged-in user"""
    user_id = session.get('user_id')

    try:
        history = db.get_user_conversations(user_id)
        return jsonify({"conversations": history})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/conversations', methods=['GET'])
@login_required
def list_conversations():
    user_id = session.get('user_id')
    try:
        conversations = db.get_user_conversations(user_id)
        return jsonify({
            "conversations": conversations,
            "active_conversation_id": session.get("conversation_id"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/conversations/<conversation_id>/select', methods=['POST'])
@login_required
def select_conversation(conversation_id):
    user_id = session.get('user_id')
    if not _set_active_conversation(user_id, conversation_id):
        return jsonify({"error": "Conversation not found"}), 404
    return jsonify({"status": "ok", "active_conversation_id": conversation_id})


@app.route('/api/conversations/<conversation_id>/messages', methods=['GET'])
@login_required
def get_conversation_messages(conversation_id):
    user_id = session.get('user_id')
    conversation = db.get_conversation(conversation_id)
    if not conversation or conversation.get("user_id") != user_id:
        return jsonify({"error": "Conversation not found"}), 404

    try:
        messages = db.get_conversation_history(conversation_id, limit=200)
        return jsonify({"conversation_id": conversation_id, "messages": messages})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=3000)
