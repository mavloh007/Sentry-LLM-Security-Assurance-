from src.chatbot.withdrawal_chatbot import WithdrawalChatbot
from src.db.supabase_client import SupabaseDB
from dotenv import load_dotenv
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

    # Initialize chatbot (auto-creates user_id and conversation)
    bot = WithdrawalChatbot(db=db)

    print("\nSGBank Withdrawal Assistant Ready.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        response = bot.chat(user_input)
        print("\nAssistant:", response, "\n")


if __name__ == "__main__":
    main()
