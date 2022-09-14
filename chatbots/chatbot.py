from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from chatbots.conversation import Conversation
import torch
class Chatbot():
    def __init__(self,model_checkpt):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpt)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpt)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

    def reply(self, usr):
        raise NotImplementedError


class FinetunedBlenderbot(Chatbot):
    def __init__(self):
        super().__init__("Adapting/dialogue_agent_nlplab2022")
        self.conv = Conversation(self.model, self.tokenizer, 128,self.device)

    def reply(self, usr):
        response = self.conv.add_user_input(usr)
        return response


class DummyBot:
    def __init__(self):
        super().__init__()

    def reply(self, usr):
        return 'This is the response from a dummy chatbot'