import random
from py2neo import Graph
from transformers import AutoTokenizer, AutoModelForTokenClassification,AutoModelForSequenceClassification
from transformers import pipeline , TextClassificationPipeline



class Template:
    def __init__(self):
        self.schema = {
    "定义":{
        "slot_list" : ["疾病和诊断"],
        "cql_template" : "MATCH(p:疾病) WHERE p.name= '<Disease>' RETURN p.desc",
        "reply_template" : "<Disease>是这样的：\n",
        "ask_template" : "您问的是<Disease>的定义吗？",
    },
    "病因":{
        "slot_list" : ["疾病和诊断"],
        "cql_template" : "MATCH(p:疾病) WHERE p.name= '<Disease>' RETURN p.cause",
        "reply_template" : "<Disease>疾病的原因是：\n",
        "ask_template" : "您问的是疾病<Disease>的原因吗？",
    },
    "预防":{
        "slot_list" : ["疾病和诊断"],
        "cql_template" : "MATCH(p:疾病) WHERE p.name= '<Disease>' RETURN p.prevent",
        "reply_template" : "关于<Disease>疾病您可以这样预防：\n",
        "ask_template" : "请问您问的是疾病<Disease>的预防措施吗？",
    },
    "临床表现(病症表现)":{
        "slot_list" : ["疾病和诊断"],
        "cql_template" : "MATCH(p:疾病)-[r:has_symptom]->(q:症状) WHERE p.name= '<Disease>' RETURN q.name",
        "reply_template" : "<Disease>疾病的病症表现一般是这样的：\n",
        "ask_template" : "您问的是疾病<Disease>的症状表现吗？",
    },
    "相关病症":{
        "slot_list" : ["疾病和诊断"],
        "cql_template" : "MATCH(p:疾病)-[r:acompany_with]->(q:疾病) WHERE p.name= '<Disease>' RETURN q.name",
        "reply_template" : "<Disease>疾病的具有以下并发疾病：\n",
        "ask_template" : "您问的是疾病<Disease>的并发疾病吗？",
    },
    "治疗方法":{
        "slot_list" : ["疾病和诊断"],
        "cql_template" : ["MATCH(p:疾病) WHERE p.name= '<Disease>' RETURN p.cure_way",
                        "MATCH(p:疾病)-[r:recommand_drug]->(q) WHERE p.name= '<Disease>' RETURN q.name",
                        "MATCH(p:疾病)-[r:recommand_recipes]->(q) WHERE p.name= '<Disease>' RETURN q.name"],
        "reply_template" : "<Disease>疾病的治疗方式、可用的药物、推荐菜肴有：\n",
        "ask_template" : "您问的是疾病<Disease>的治疗方法吗？",
    },
    "所属科室":{
        "slot_list" : ["疾病和诊断"],
        "cql_template" : "MATCH(p:疾病)-[r:cure_department]->(q:科室) WHERE p.name= '<Disease>' RETURN q.name",
        "reply_template" : "得了<Disease>可以挂这个科室哦：\n",
        "ask_template" : "您想问的是疾病<Disease>要挂什么科室吗？",
    },
    "传染性":{
        "slot_list" : ["疾病和诊断"],
        "cql_template" : "MATCH(p:疾病) WHERE p.name= '<Disease>' RETURN p.easy_get",
        "reply_template" : "<Disease>较为容易感染这些人群：\n",
        "ask_template" : "您想问的是疾病<Disease>会感染哪些人吗？",
    },
    "治愈率":{
        "slot_list" : ["疾病和诊断"],
        "cql_template" : "MATCH(p:疾病) WHERE p.name= '<Disease>' RETURN p.cured_prob",
        "reply_template" : "得了<Disease>的治愈率为：",
        "ask_template" : "您想问<Disease>的治愈率吗？",
    },
    "治疗时间":{
        "slot_list" : ["疾病和诊断"],
        "cql_template" : "MATCH(p:疾病) WHERE p.name= '<Disease>' RETURN p.cure_lasttime",
        "reply_template" : "疾病<Disease>的治疗周期为：",
        "ask_template" : "您想问<Disease>的治疗周期吗？",
    },
    "化验/体检方案":{
        "slot_list" : ["疾病和诊断"],
        "cql_template" : "MATCH(p:疾病)-[r:need_check]->(q:检查) WHERE p.name= '<Disease>' RETURN q.name",
        "reply_template" : "得了<Disease>需要做以下检查：\n",
        "ask_template" : "您是想问<Disease>要做什么检查吗？",
    },
    "禁忌":{
        "slot_list" : ["疾病和诊断"],
        "cql_template" : "MATCH(p:疾病)-[r:not_eat]->(q:食物) WHERE p.name= '<Disease>' RETURN q.name",
        "reply_template" : "得了<Disease>切记不要吃这些食物哦：\n",
        "ask_template" : "您是想问<Disease>不可以吃的食物是什么吗？",
    },
    "其他": None
}
        self.dunno_templates = [
            "人类的语言太难了！！",
            "没有理解您说的意思哦~",
            "非常抱歉，我还不知道如何回答您，我正在努力学习中~",
            "额~似乎有点不理解你说的是啥呢~~",
            "您说的我有点不明白，您可以换个问法问我哦~",
            "很抱歉没有理解你的意思呢~",
        ]

        self.greeting_templates = [
            "hi",
            "你好呀",
            "我是智能医疗诊断机器人，有什么可以帮助你吗",
            "hi，你好，你可以叫我小智",
            "你好，你可以问我一些关于疾病诊断的问题哦"
        ]

        self.goodbye_templates = [
            "再见，很高兴为您服务",
            "bye",
            "再见，感谢使用我的服务",
            "再见啦，祝你健康"
        ]

        self.self_intro_templates = [
            "我是小智，你的智能健康顾问",
            "你可以叫我小智哦~",
            "我是医疗诊断机器人小智"
        ]





class Conversation:
    def __init__(self,neo4j_password:str) -> None:
        self.context = None  # 正在讨论的疾病
        self.templates = Template()
        self.password = neo4j_password
        self.kg = Graph("neo4j+s://f54cadff.databases.neo4j.io:7687", auth=("neo4j", self.password))
        self.ner_pipeline = pipeline('ner',model=AutoModelForTokenClassification.from_pretrained("Adapting/bert-base-chinese-finetuned-NER-biomedical",
                                                                revision='7f63e3d18b1dc3cc23041a89e77be21860704d2e')
                                     ,tokenizer =AutoTokenizer.from_pretrained("Adapting/bert-base-chinese-finetuned-NER-biomedical") )
        self.ir_pipeline = TextClassificationPipeline(model=AutoModelForSequenceClassification.from_pretrained("nlp-guild/bert-base-chinese-finetuned-intent_recognition-biomedical"), tokenizer=AutoTokenizer.from_pretrained("nlp-guild/bert-base-chinese-finetuned-intent_recognition-biomedical"))

    def reply(self, user_query: str):
        if any(x in user_query for x in ['hello', 'hi', '你好', '嗨嗨', '你好啊']):
            return random.choice(self.templates.greeting_templates)
        elif any(x in user_query for x in ['bye', 'goodbye', '88', '再见', '谢谢', '感谢', '拜拜', '白白', '好的']):
            return random.choice(self.templates.goodbye_templates)
        elif any(x in user_query for x in ['名字', '是谁', 'who are you', '你是']):
            return random.choice(self.templates.self_intro_templates)

        user_intent = self.readable_results_ir(1, user_query)[0]['label']
        if user_intent == '其他':
            return random.choice(self.templates.dunno_templates)
        else:
            # get the slot list for this intent
            schema = self.templates.schema[user_intent]
            slot_list = schema['slot_list']

            ne = self.readable_results_ner(user_query)
            slots = {}

            for slot in slot_list:
                slot_value = ne.get(slot, None)
                if slot_value is None:
                    if slot == '疾病和诊断' and self.context is not None:
                        slot_value = [self.context]
                        slots[slot] = slot_value
                    else:
                        return f'请您提供一下信息: {slot}.'
                else:
                    slots[slot] = slot_value

            # 到这里时该intent所有必要的slots已填充并保存于 $slots$
            disease = slots['疾病和诊断'][0]
            self.context = disease
            cql_templates = schema['cql_template']

            knowledge = ''

            if isinstance(cql_templates, list):
                for _ in cql_templates:
                    knowledge += self.query_kg(_.replace('<Disease>', disease)) + '\n'
            else:
                knowledge += self.query_kg(cql_templates.replace('<Disease>', disease))

            response = schema['reply_template'].replace('<Disease>', disease)
            response += knowledge

            # context management

            return response

    def query_kg(self, cql: str):
        try:
            ret = ''
            data = self.kg.run(cql).data()
            for i in data:
                tmp = list(i.values())[0]
                if isinstance(tmp, list):
                    ret += ', '.join(tmp)
                else:
                    ret += tmp

                ret += ', '

            return ret[:-2] + '.'
        except Exception as e:
            print(e)
            return ''


    def readable_results_ner(self,query):
        result = self.ner_pipeline(query)

        tag_set = [
            'B_手术',
            'I_疾病和诊断',
            'B_症状',
            'I_解剖部位',
            'I_药物',
            'B_影像检查',
            'B_药物',
            'B_疾病和诊断',
            'I_影像检查',
            'I_手术',
            'B_解剖部位',
            'O',
            'B_实验室检验',
            'I_症状',
            'I_实验室检验'
        ]

        # tag2id = lambda tag: tag_set.index(tag)
        id2tag = lambda id: tag_set[id]

        results_in_word = {}
        j = 0
        while j < len(result):
            i = result[j]
            entity = id2tag(int(i['entity'][i['entity'].index('_')+1:]))
            token = i['word']
            if entity.startswith('B'):
                entity_name = entity[entity.index('_')+1:]

                word = token
                j = j+1
                while j<len(result):
                    next = result[j]
                    next_ent = id2tag(int(next['entity'][next['entity'].index('_')+1:]))
                    next_token = next['word']

                    if next_ent.startswith('I') and next_ent[next_ent.index('_')+1:] == entity_name:
                        word += next_token
                        j += 1

                        if j >= len(result):
                            # results_in_word.append((entity_name,word))
                            if entity_name not in results_in_word.keys():
                                results_in_word[entity_name] = [word]
                            else:
                                 results_in_word[entity_name].append(word)
                    else:
                        # results_in_word.append((entity_name,word))
                        if entity_name not in results_in_word.keys():
                            results_in_word[entity_name] = [word]
                        else:
                            results_in_word[entity_name].append(word)
                        break

            else:
                j += 1

        return results_in_word

    def readable_results_ir(self,top_k: int, usr_query: str):
        label_set = [
            '定义',
            '病因',
            '预防',
            '临床表现(病症表现)',
            '相关病症',
            '治疗方法',
            '所属科室',
            '传染性',
            '治愈率',
            '禁忌',
            '化验/体检方案',
            '治疗时间',
            '其他'
        ]

        raw = self.ir_pipeline(usr_query, top_k=top_k)

        def f(x):
            index = int(x['label'][6:])
            x['label'] = label_set[index]

        for i in raw:
            f(i)
        return raw




