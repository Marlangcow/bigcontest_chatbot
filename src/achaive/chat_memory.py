class AIChatHistory:
    def __init__(self, max_messages=50):
        self.history = []
        self.max_messages = max_messages

    def add_message(self, role, content):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_messages:
            self.history = self.history[-self.max_messages :]
