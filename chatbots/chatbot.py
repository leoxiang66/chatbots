from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from chatbots.conversation import Conversation
class Chatbot():
    def __init__(self,model_checkpt,device):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpt)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpt)
        self.device = device

    def reply(self, usr):
        raise NotImplementedError


class FinetunedBlenderbot(Chatbot):
    def __init__(self,device):
        super().__init__("Adapting/dialogue_agent_nlplab2022",device)
        self.conv = Conversation(self.model, self.tokenizer, 128, device)

    def reply(self, usr):
        response = self.conv.add_user_input(usr)
        return response