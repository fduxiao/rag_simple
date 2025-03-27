from ..chatbot import Chatbot


class ChatResponder:
    def __init__(self, chatbot: Chatbot, default_limit=3):
        self.chatbot = chatbot
        self.default_limit = default_limit

    def respond_to(self, user_input):
        # parse user input
        if user_input.startswith("/retrieve "):
            user_input = user_input[len("/retrieve ") :]
            self.chatbot.retrieve(user_input, limit=1)
            return

        self.chatbot.retrieve(user_input, limit=self.default_limit).drain()
        self.chatbot.chat(user_input).print()
