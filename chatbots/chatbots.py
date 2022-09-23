from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from chatbots.blenderbot_conversation import Conversation
from .kbqa_diagnosis import Conversation as kbqa_conv
import torch


class BaseChatbot():
    def __init__(self):
        return

    def reply(self,usr):
        raise NotImplementedError


class Chatbot_with_checkpoint(BaseChatbot):
    def __init__(self, model_checkpt):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpt)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpt)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

    def reply(self, usr):
        raise NotImplementedError


class FinetunedBlenderbot(Chatbot_with_checkpoint):
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


class KBQA_diagnosis(BaseChatbot):
    def __init__(self,neo4j_password:str):
        super().__init__()
        self.conv = kbqa_conv(neo4j_password)

    def reply(self, usr):
        return self.conv.reply(usr)