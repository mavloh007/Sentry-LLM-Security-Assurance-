from src.chatbot.withdrawal_chatbot import WithdrawalChatbot
from src.db.supabase_client import SupabaseDB
from dotenv import load_dotenv
from uuid import uuid5, NAMESPACE_DNS
import os


def main():
    load_dotenv()

    # Initialize Supabase database
    db = SupabaseDB()

    # Verify connection
    health = db.health_check()
    if not health:
        print("Error: Could not connect to Supabase database")
        return

    print("Connected to Supabase database.")

    # Initialize chatbot (shared instance — no per-user state)
    bot = WithdrawalChatbot(db=db)

    # Create a stable local test user and conversation for the CLI session
    user_id = str(uuid5(NAMESPACE_DNS, "local-test-user"))
    if not db.get_user(user_id):
        db.create_user(user_id=user_id, email="local@test.local", metadata={"type": "local_chatbot"})

    conv = db.create_conversation(user_id=user_id, title="CLI Session")
    conversation_id = conv["id"]

    print("\nSGBank Withdrawal Assistant Ready.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        response = bot.chat(user_input, user_id=user_id, conversation_id=conversation_id)
        print("\nAssistant:", response, "\n")


if __name__ == "__main__":
    main()
