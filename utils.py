from statistics import mean
from torch.utils.data import Dataset
from collections import OrderedDict
import xml.etree.ElementTree as ET
from transformers import T5ForConditionalGeneration as T5, AutoTokenizer
import torch
import openai  # For GPT-3 API ...
import os
import multiprocessing
import json
import numpy as np
import random
import torch
# import torchtext
import re
import random
import time
import datetime
import pandas as pd


# https://review-of-my-life.blogspot.com/2017/11/python-dict-shuffle.html
def shuffleDict(d):
    keys = list(d.keys())
    random.shuffle(keys)
    [(key, d[key]) for key in keys]
    random.shuffle(keys)
    [(key, d[key]) for key in keys]
    random.shuffle(keys)
    keys = [(key, d[key]) for key in keys]
    # keys = d(keys)
    return dict(keys)


def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def print_now(return_flag=0):
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    now = now.strftime('%Y/%m/%d %H:%M:%S')
    if return_flag == 0:
        print(now)
    elif return_flag == 1:
        return now
    else:
        pass


# Sentence Generator (Decoder) for GPT-3 ...
def decoder_for_gpt3(args, input, max_length, i, k, n, rd,stop):
    # GPT-3 API allows each users execute the API within 20 times in a minute ...
    time.sleep(args.api_time_interval)

    # https://beta.openai.com/account/api-keys
    openai.api_key = os.getenv("OPENAI_API_KEY")
    # print(openai.api_key)

    # Specify engine ...
    # Instruct GPT3
    if args.model == "gpt3":
        engine = "text-ada-001"
    elif args.model == "gpt3-medium":
        engine = "text-babbage-001"
    elif args.model == "gpt3-large":
        engine = "text-curie-001"
    elif args.model == "gpt3-xl":
        engine = "text-davinci-002"
    elif args.model == "codex":
        engine = "code-davinci-002"
    elif args.model == "codex-001":
        engine = "code-davinci-001"
    else:
        raise ValueError("model is not properly defined ...")

    response = openai.Completion.create(
        engine=engine,
        prompt=input,
        max_tokens=max_length,
        temperature=k,
        stop=stop,
        n=n
    )

    return [text["text"] for text in response["choices"]]


class Decoder():
    def __init__(self, args):
        print_now()
        if args.model == 'UL2':
            self.model = T5.from_pretrained("google/ul2", low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
                                            cache_dir='/home/zmj/CoT/UL2/weight')
            self.tokenizer = AutoTokenizer.from_pretrained("google/ul2")
            device_map = {
                6: [0, 1, 2, 3, 4],
                7: [5, 6, 7, 8, 9, 10, 11, 12, 13, ],
                8: [14, 15, 16, 17, 18, 19, 20, 21, 22, ],
                9: [23, 24, 25, 26, 27, 28, 29, 30, 31],
            }
            self.model.parallelize(device_map)
        else:
            self.rd = 0

    def decode(self, args, input, max_length, i, k, n,stop,is_turn_to_declarative=False):
        try:
            if args.model in ("gpt3", "gpt3-medium", "gpt3-large", "gpt3-xl", "codex", "codex-001"):
                response = decoder_for_gpt3(args, input, max_length, i, k, n, self.rd,stop)
                if self.rd != 3:
                    self.rd += 1
                else:
                    self.rd = 1
            elif args.model in ("UL2"):
                if is_turn_to_declarative:
                    input_ids = self.tokenizer('[S2S] ' + input[81:] + '\" <extra_id_0> \"', return_tensors="pt").input_ids.to(6)
                else:
                    input_ids = self.tokenizer('[S2S] ' + input + '<extra_id_0>', return_tensors="pt").input_ids.to(6)
                output = self.tokenizer.batch_decode(
                    self.model.generate(input_ids, temperature=k, do_sample=True, top_k=10, num_return_sequences=n,
                                        max_length=args.max_length_cot))
                response = [i.replace('<pad> ', '').replace('<pad>', '').replace('</s>', '').split('<extra')[0] for i in output]
        except:
            response = []
        return response


def data_reader(args):
    questions = []
    answers = []
    decoder = json.JSONDecoder()

    if args.dataset == "aqua":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                choice = "(" + "(".join(json_res["options"])
                choice = choice.replace("(", " (").replace(")", ") ")
                choice = "Answer Choices:" + choice
                questions.append(json_res["question"].strip() + " " + choice)
                answers.append(json_res["correct"])

    elif args.dataset == "gsm8k":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                questions.append(json_res["question"].strip())
                answers.append(json_res["answer"].split("#### ")[-1])

    elif args.dataset == "commonsensqa":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                choice = "Answer Choices:"
                for c in json_res["question"]["choices"]:
                    choice += " ("
                    choice += c["label"]
                    choice += ") "
                    choice += c["text"]
                questions.append(json_res["question"]["stem"].strip() + " " + choice)
                answers.append(json_res["answerKey"])

    elif args.dataset in ("addsub", "multiarith", "singleeq"):
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["sQuestion"].strip()
                a = str(line["lSolutions"][0])
                if a[-2:] == ".0":
                    a = a[:-2]
                questions.append(q)
                answers.append(a)

    elif args.dataset == "strategyqa":
        with open(args.dataset_path) as f:
            json_data = json.load(f)["examples"]
            for line in json_data:
                q = line["input"].strip()
                a = int(line["target_scores"]["Yes"])
                if a == 1:
                    a = "yes"
                else:
                    a = "no"
                questions.append(q)
                answers.append(a)

    elif args.dataset == "svamp":
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["Body"].strip() + " " + line["Question"].strip()
                a = str(line["Answer"])
                if a[-2:] == ".0":
                    a = a[:-2]
                questions.append(q)
                answers.append(a)

    elif args.dataset in ("bigbench_date", "object_tracking"):
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            json_data = json_data["examples"]
            if args.dataset == "bigbench_date":
                choice_index = ['A', 'B', 'C', 'D', 'E', 'F']
            elif args.dataset in ("object_tracking"):
                choice_index = ['A', 'B', 'C']
            else:
                raise ValueError("dataset is not properly defined ...")
            for line in json_data:
                q = line["input"].strip()
                if args.dataset == "bigbench_date":
                    choice = "Answer Choices:"
                    # Randomly shuffle the answer choice dictionary because the original answer is always A ...
                    choice_dic = shuffleDict(line["target_scores"])
                elif args.dataset == "object_tracking":
                    choice = "\nWhich choice is true ? Answer Choices:"
                    choice_dic = line["target_scores"]
                else:
                    raise ValueError("dataset is not properly defined ...")
                for i, key_value in enumerate(choice_dic.items()):
                    key, value = key_value
                    choice += " ("
                    choice += choice_index[i]
                    choice += ") "
                    choice += key
                    if value == 1:
                        a = key
                        # a = key
                q = q
                questions.append(q)
                answers.append(a)

    elif args.dataset in ("coin_flip", "last_letters"):
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            json_data = json_data["examples"]
            for line in json_data:
                q = line["question"]
                a = line["answer"]
                questions.append(q)
                answers.append(a)

    elif args.dataset in ("meddialog"):
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = 'Patient Description: '+ line["description"] + '\nPatient: ' + ':'.join(line['utterances'][-2].split(':')[1:])
                a = ':'.join(line['utterances'][-1].split(':')[1:])
                questions.append(q)
                answers.append(a)

    else:
        raise ValueError("dataset is not properly defined ...")

    q_len_list = []
    for q in questions:
        q_len_list.append(len(q.split(" ")))
    q_len_mean = mean(q_len_list)

    print("dataset : {}".format(args.dataset))
    print("data size : {}".format(len(answers)))
    print("average num of words for each sample : {}".format(q_len_mean))

    return questions, answers


# Create dataset object before dataloader ...
class MyDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.questions, self.answers = data_reader(args)
        self.len = len(self.questions)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        input = self.questions[index]
        output = self.answers[index]
        return input, output


def setup_data_loader(args):
    # fix randomness of dataloader to ensure reproducibility
    # https://pytorch.org/docs/stable/notes/randomness.html
    fix_seed(args.random_seed)
    worker_seed = torch.initial_seed() % 2 ** 32
    print("worker_seed : {}".format(worker_seed))

    def seed_worker(worker_id):
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(worker_seed)

    dataloader_num_workers = multiprocessing.cpu_count()
    dataloader_num_workers = min(dataloader_num_workers, args.max_num_worker)
    print("dataloader_num_workers: " + str(dataloader_num_workers))

    dataset = MyDataset(args)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             shuffle=True,
                                             batch_size=args.minibatch_size,
                                             drop_last=False,
                                             num_workers=dataloader_num_workers,
                                             worker_init_fn=seed_worker,
                                             generator=g,
                                             pin_memory=True)

    return dataloader


# ver 0.2
def answer_cleansing(args, pred):
    if args.method in ("few_shot", "few_shot_cot",'verifier', 'verifier_cot','verifier_TF_cot', "zero_shot_verifier_cot"):
        preds = pred.split(args.direct_answer_trigger_for_fewshot)
        answer_flag = True if len(preds) > 1 else False
        pred = preds[-1]

    if args.dataset in ("aqua", "commonsensqa"):
        pred = re.findall(r'A|B|C|D|E', pred)
    elif args.dataset == "bigbench_date":
        pred = [pred[1:-1]]
    elif args.dataset in ("object_tracking"):
        pred = re.findall(r'A|B|C', pred)
    elif args.dataset in ("gsm8k", "addsub", "multiarith", "svamp", "singleeq"):
        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
    elif args.dataset in ("strategyqa", "coin_flip"):
        pred = pred.lower()
        pred = re.sub("\"|\'|\n|\.|\s|\:|\,", " ", pred)
        pred = pred.split(" ")
        pred = [i for i in pred if i in ("yes", "no")]
    elif args.dataset == "last_letters":
        pred = re.sub("\"|\'|\n|\.|\s", "", pred)
        pred = [pred]
    else:
        raise ValueError("dataset is not properly defined ...")

    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        if args.method in ("few_shot", "few_shot_cot", "verifier","verifier_cot",'verifier_TF_cot',"zero_shot_verifier_cot"):
            if answer_flag:
                # choose the first element in list ...
                pred = pred[0]
            else:
                # choose the last element in list ...
                pred = pred[-1]
        elif args.method in ("zero_shot", "zero_shot_cot"):
            # choose the first element in list ...
            pred = pred[0]
        else:
            raise ValueError("method is not properly defined ...")

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]

    return pred


def answer_cleansing_verifier(args, pred):
    if args.method in ("few_shot", "few_shot_cot",'verifier', 'verifier_cot','verifier_TF_cot', "zero_shot_verifier_cot"):
        preds = pred.split(args.direct_answer_trigger_for_fewshot)
        answer_flag = True if len(preds) > 1 else False
        pred = preds[-1]

    if args.dataset in ("aqua"):
        pred = re.findall(r'A|B|C|D|E', pred)
    elif args.dataset in ("object_tracking"):
        pred = re.findall(r'A|B|C', pred)
    elif args.dataset in ("gsm8k", "addsub", "multiarith", "svamp", "singleeq", "bigbench_date"):
        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
    elif args.dataset in ("strategyqa", "coin_flip", "commonsensqa"):
        pred = pred.lower()
        pred = re.sub("\"|\'|\n|\.|\s|\:|\,", " ", pred)
        pred = pred.split(" ")
        pred = [i for i in pred if i in ("yes", "no")]
    elif args.dataset == "last_letters":
        pred = re.sub("\"|\'|\n|\.|\s", "", pred)
        pred = [pred]
    else:
        raise ValueError("dataset is not properly defined ...")

    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        if args.method in ("few_shot", "few_shot_cot",'verifier', "verifier_cot",'verifier_TF_cot', "zero_shot_verifier_cot"):
            if answer_flag:
                # choose the first element in list ...
                pred = pred[0]
            else:
                # choose the last element in list ...
                pred = pred[-1]
        elif args.method in ("zero_shot", "zero_shot_cot"):
            # choose the first element in list ...
            pred = pred[0]
        else:
            raise ValueError("method is not properly defined ...")

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]

    return pred


def create_demo_text(args, cot_flag):
    x, z, y, c = [], [], [], []

    # example sentences ...    
    if args.dataset in ("multiarith", "gsm8k", "addsub", "svamp", "singleeq"):

        x.append(
            "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?")
        z.append(
            "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.")
        y.append("6")

        x.append("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?")
        z.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
        y.append("5")

        x.append(
            "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?")
        z.append(
            "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.")
        y.append("39")

        x.append(
            "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?")
        z.append(
            "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.")
        y.append("8")

        x.append(
            "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?")
        z.append(
            "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.")
        y.append("9")

        x.append(
            "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?")
        z.append(
            "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.")
        y.append("29")

        x.append(
            "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?")
        z.append(
            "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.")
        y.append("33")

        x.append("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?")
        z.append(
            "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.")
        y.append("8")

    elif args.dataset in ("aqua"):
        x.append(
            "John found that the average of 15 numbers is 40. If 10 is added to each number then the mean of the numbers is? ")
        c.append(
            "Answer Choices: (A) 50 (B) 45 (C) 65 (D) 78 (E) 64")
        z.append(
            "If 10 is added to each number, then the mean of the numbers also increases by 10. So the new mean would be 50.")
        y.append("A")

        x.append("If a / b = 3/4 and 8a + 5b = 22, then find the value of a. ")
        c.append("Answer Choices: (A) 1/2 (B) 3/2 (C) 5/2 (D) 4/2 (E) 7/2")
        z.append(
            "If a / b = 3/4, then b = 4a / 3. So 8a + 5(4a / 3) = 22. This simplifies to 8a + 20a / 3 = 22, which means 44a / 3 = 22. So a is equal to 3/2.")
        y.append("B")

        x.append(
            "A person is traveling at 20 km/hr and reached his destiny in 2.5 hr then find the distance? ")
        c.append(
            "Answer Choices: (A) 53 km (B) 55 km (C) 52 km (D) 60 km (E) 50 km")
        z.append("The distance that the person traveled would have been 20 km/hr * 2.5 hrs = 50 km.")
        y.append("E")

        x.append(
            "How many keystrokes are needed to type the numbers from 1 to 500? ")
        c.append("Answer Choices: (A) 1156 (B) 1392 (C) 1480 (D) 1562 (E) 1788")
        z.append(
            "There are 9 one-digit numbers from 1 to 9. There are 90 two-digit numbers from 10 to 99. There are 401 three-digit numbers from 100 to 500. 9 + 90(2) + 401(3) = 1392.")
        y.append("B")

    elif args.dataset in ("commonsensqa"):
        x.append(
            "What do people use to absorb extra ink from a fountain pen? ")
        c.append("Answer Choices: (A) shirt pocket (B) calligrapher's hand (C) inkwell (D) desk drawer (E) blotter")
        z.append(
            "The answer must be an item that can absorb ink. Of the above choices, only blotters are used to absorb ink.")
        y.append("E")

        x.append(
            "What home entertainment equipment requires cable? ")
        c.append("Answer Choices: (A) radio shack (B) substation (C) television (D) cabinet")
        z.append(
            "The answer must require cable. Of the above choices, only television requires cable.")
        y.append("C")

        x.append(
            "The fox walked from the city into the forest, what was it looking for? ")
        c.append("Answer Choices: (A) pretty flowers (B) hen house (C) natural habitat (D) storybook")
        z.append(
            "The answer must be something in the forest. Of the above choices, only natural habitat is in the forest.")
        y.append("C")

        x.append(
            "Sammy wanted to go to where the people were. Where might he go? ")
        c.append("Answer Choices: (A) populated areas (B) race track (C) desert (D) apartment (E) roadblock")
        z.append(
            "The answer must be a place with a lot of people. Of the above choices, only populated areas have a lot of people.")
        y.append("A")

        x.append(
            "Where do you put your grapes just before checking out? ")
        c.append("Answer Choices: (A) mouth (B) grocery cart (C) super market (D) fruit basket (E) fruit market")
        z.append(
            "The answer should be the place where grocery items are placed before checking out. Of the above choices, grocery cart makes the most sense for holding grocery items.")
        y.append("B")

        x.append(
            "Google Maps and other highway and street GPS services have replaced what? ")
        c.append("Answer Choices: (A) united states (B) mexico (C) countryside (D) atlas")
        z.append(
            "The answer must be something that used to do what Google Maps and GPS services do, which is to give directions. Of the above choices, only atlases are used to give directions.")
        y.append("D")

        x.append(
            "Before getting a divorce, what did the wife feel who was doing all the work? ")
        c.append("Answer Choices: (A) harder (B) anguish (C) bitterness (D) tears (E) sadness")
        z.append(
            "The answer should be the feeling of someone getting divorced who was doing all the work. Of the above choices, the closest feeling is bitterness.")
        y.append("C")

    elif args.dataset in ("bigbench_date"):
        x.append(
            "2015 is coming in 36 hours. What is the date one week from today in MM/DD/YYYY?")
        z.append(
            "If 2015 is coming in 36 hours, then it is coming in 2 days. 2 days before 01/01/2015 is 12/30/2014, so today is 12/30/2014. So one week from today will be 01/05/2015.")
        y.append("01/05/2015")

        x.append(
            "The first day of 2019 is a Tuesday, and today is the first Monday of 2019. What is the date today in MM/DD/YYYY?")
        z.append(
            "If the first day of 2019 was Tuesday, then 01/01/2019 was a Tuesday. Today is the first monday, would be six days later. So today is 01/07/2019.")
        y.append("01/07/2019")

        x.append(
            "The concert was scheduled to be on 06/01/1943, but was delayed by one day to today. What is the date 10 days ago in MM/DD/YYYY?")
        z.append(
            "One day after 06/01/1943 is 06/02/1943, so today is 06/02/1943. 10 days before today is 05/23/1943.")
        y.append("05/23/1943")

        x.append(
            "It is 4/19/1969 today. What is the date 24 hours later in MM/DD/YYYY?")
        z.append(
            "Today is 04/19/1969. 24 hours later is one day after today, which would be 04/20/1969.")
        y.append("04/20/1969")

        x.append(
            "Jane thought today is 3/11/2002, but today is in fact Mar 12, which is 1 day later. What is the date 24 hours later in MM/DD/YYYY?")
        z.append(
            "Today is 03/12/2002. So the date 24 hours later will be 03/13/2002")
        y.append("03/13/2002")

        x.append(
            "Jane was born on the last day of Feburary in 2001. Today is her 16-year-old birthday. What is the date yesterday in MM/DD/YYYY?")
        z.append(
            "The last day of February is the 28th, so Jane was born on 02/28/2001. Today is her 16-year old birthday, so today is 02/28/2017. So yesterday was 02/27/2017. ")
        y.append("02/27/2017")

    else:
        raise ValueError("dataset is not properly defined ...")

    # randomize order of the examples ...
    index_list = list(range(len(x)))
    random.shuffle(index_list)
    if args.FN != 0:
        index_list = index_list[:args.FN]

    # Concatenate demonstration examples ...
    demo_text = ""
    for i in index_list:
        if len(c) > 0:
            if cot_flag:
                demo_text += "Q: " + x[i] + c[i] + "\nA: " + z[i] + " " + \
                             args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
            else:
                demo_text += "Q: " + x[i] + c[i] + "\nA: " + \
                             args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
        if args.dataset in ('meddialog'):
            if cot_flag:
                demo_text += x[i] + "\nRecord Report: " + z[i] + "\n" + \
                             args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
            else:
                demo_text += x[i] + "\n" + \
                             args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
        else:
            if cot_flag:
                demo_text += "Q: " + x[i] + "\nA: " + z[i] + " " + \
                             args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
            else:
                demo_text += "Q: " + x[i] + "\nA: " + \
                             args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"

    return demo_text



def create_verifier_demo_text(args, cot_flag):
    x, z, y = [], [], []

    # example sentences ...
    if args.dataset in ("multiarith", "gsm8k", "addsub", "svamp", "singleeq"):

        x.append(
            "\"There are 'X' trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. The grove workers planted 6 trees today.\" What is the answer of 'X'?")
        z.append(
            "There are X trees originally. The grove workers planted 6 trees today. Then there were 21 trees after some more were planted. So, we can write the following equation:\n<code>X + 6 = 21\n X = 21 - 6\n X = 15\n</code>\n")
        y.append("15")

        x.append(
            "\"If there are 'X' cars in the parking lot and 2 more cars arrive, There are 5 cars in the parking lot.\" What is the answer of 'X'?")
        z.append(
            "There are originally X cars. 2 more cars arrive and there are 5 cars finally. So:\n<code> X + 2 = 5\n X = 5 - 2\n X = 3\n</code>\n")
        y.append("3")

        x.append(
            "\"Leah had 'X' chocolates and her sister had 42. If they ate 35, they have 39 pieces left in total.\" What is the answer of 'X'?")
        z.append(
            "Originally, Leah had X chocolates. Her sister had 42. So in total they had:\n<code> X + 42 = Y\n</code>\n. After eating 35, they had 39, so\n<code> Y = 35 + 39\n Y = 74\n X + 42 = 74\n X = 74 - 42\n X = 32\n</code>\n")
        y.append("32")

        x.append(
            "\"Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 'X' lollipops. Jason gave Denny 8 lollipops.\" What is the answer of 'X'?")
        z.append(
            "Jason started with 20 lollipops. Then he had X after giving some to Denny and gave Denny 8.\n<code> 20 - X = 8\n X = 12\n</code>\n")
        y.append("12")

        x.append(
            "\"Shawn has 'X' toys. For Christmas, he got two toys each from his mom and dad. He has 9 toys now.\" What is the answer of 'X'?")
        z.append(
            "Shawn started with X toys. If he got 2 toys each from his mom and dad, then that is 4 more toys.\n<code> X + 4 = 9\n X = 9 - 4\n X = 5\n</code>\n")
        y.append("5")

        x.append(
            "\"There were 'X' computers in the server room. Five more computers were installed each day, from monday to thursday. There are 29 computers in the server room.\" What is the answer of 'X'?")
        z.append(
            "There were originally X computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. And there are 29 computers.\n<code> X + 20 = 29\n X = 29 - 20\n X = 9\n</code>\n")
        y.append("9")

        x.append(
            "\"Michael had 58 golf balls. On tuesday, he lost 'X' golf balls. On wednesday, he lost 2 more. He had 33 golf balls at the end of Wednesday.\" What is the answer of 'X'?")
        z.append(
            "Michael started with 58 golf balls. After losing X on tuesday and he lost 2 more on wednesday, He had 33 golf balls. So, we can write the following equation:\n<code> 58 - X - 2 = 33\n 58 - X = 35\n X = 23\n</code>\n")
        y.append("23")

        x.append(
            "\"Olivia has $'X'. She bought five bagels for $3 each. She has 8 dollars left.\" What is the answer of 'X'?")
        z.append(
            "Olivia had X dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. She has 8 dollars left finally.\n<code> X - 15 = 8\n X = 8 + 15\n X = 23\n</code>\n")
        y.append("23")

    elif args.dataset in ("aqua"):
        x.append(
            "\"John found that the average of 15 numbers is 'X'. If 10 is added to each number then the mean of the numbers is 50.\" What is the answer of 'X'?")
        z.append(
            "If 10 is added to each number, then the mean of the numbers also increases by 10. The new mean would be 50.\n<code> X + 10 = 50\n X = 40\n</code>\n")
        y.append("40")

        x.append("\"If a / b = 'X' and 8a + 5b = 22, then the value of a is 3/2.\" What is the answer of 'X'?")
        z.append(
            "If a / b = X, then 8a + 5b = 22 and a = 3/2, so \n<code> 8 * 3/2 + 5b = 22\n 5b = 22 - 12 = 10\n b = 2\n X = a / b = 3/2 / 2 = 3/4\n</code>\n")
        y.append("3/4")

        x.append(
            "\"A person is traveling at 'X' km/hr and reached his destiny in 2.5 hr then find the distance is 50km.\" What is the answer of 'X'?")
        z.append("The distance that the person traveled would have been \n<code> X km/hr * 2.5 hrs = 50 k\n X = 20\n</code>\n")
        y.append("20")

        x.append(
            "\"There were 'X' computers in the server room. Five more computers were installed each day, from monday to thursday. There are 29 computers in the server room.\" What is the answer of 'X'?")
        z.append(
            "There were originally X computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. And there are 29 computers.\n<code> X + 20 = 29\n X = 29 - 20\n X = 9\n<code>")
        y.append("9")

    elif args.dataset in ("commonsensqa"):
        x.append(
            "\"People use blotter to absorb extra ink from a fountain pen.\" Judge whether this statement is normal (yes or no).")
        z.append(
            "The Blotter is used to absorb extra ink from a fountain pen.")
        y.append("Yes")

        x.append(
            "\"Television requires cable.\" Judge whether this statement is normal (yes or no).")
        z.append(
            "The Television is an electrical appliance, it needs electricity, so it requires cables")
        y.append("Yes")

        x.append(
            "\"The fox walked from the city into the forest, it was looking for a hen house.\" Judge whether this statement is normal (yes or no).")
        z.append(
            "The hen house is not in the forest, so the fox does not go to the hen house.")
        y.append("No")

        x.append(
            "\"Sammy wanted to go to where the people were. He might go populated areas.\" Judge whether this statement is normal (yes or no).")
        z.append(
            "There are many people in the populated areas, so they really go here.")
        y.append("Yes")

        x.append(
            "\"The grapes are put in the fruit market just before checking out.\" Judge whether this statement is normal (yes or no).")
        z.append(
            "The answer should be the place where grocery items are placed before checking out. But the fruit market is not suitable place where grocery items are placed.")
        y.append("No")

        x.append(
            "\"Google Maps and other highway and street GPS services have replaced the united states.\" Judge whether this statement is normal (yes or no).")
        z.append(
            "The united states is a country and Google Maps is a map, so Google Maps cannot replace the united states")
        y.append("No")

        x.append(
            "\"The wife who was doing all the work felt bitterness before getting a divorce.\" Judge whether this statement is normal (yes or no).")
        z.append(
            "The wife divorced who was doing all the work. So she felt bitterness.")
        y.append("Yes")

    elif args.dataset in ("bigbench_date"):
        x.append(
            "\"'X' is coming in 36 hours. One week from today is 01/05/2015.\" What is the answer of 'X'?")
        z.append(
            "If The date one week from today is 01/05/2015, so today is 12/30/2014. So the data after 36 hours is 2015.")
        y.append("2015")

        x.append(
            "\"The first day of 'X' is a Tuesday, and today is the first Monday of 2019. Today is 01/07/2019.\" What is the answer of 'X'?")
        z.append(
            "If today is the first Monday of 2019 and today is 01/07/2019. So The first day of 2019 is a Tuesday.")
        y.append("2019")

        x.append(
            "\"The concert was scheduled to be on 'X'/01/1943, but was delayed by one day to today. 10 days ago is 05/23/1943.\" What is the answer of 'X'?")
        z.append(
            "10 days ago is 05/23/1943, and the concert was delayed by one day to today, so today is 06/02/1943. So the concert was scheduled to be on 06/01/1943")
        y.append("06")

        x.append(
            "\"It is ’X'/19/1969 today. 24 hours later is 04/20/1969.\" What is the answer of 'X'?")
        z.append(
            "24 hours later is 04/20/1969. So today is 04/19/1969.")
        y.append("04")

        x.append(
            "\"Jane thought today is 'X'/12/2002, but today is in fact Mar 12, which is 1 day later. 24 hours later is 03/13/2002.\" What is the answer of 'X'?")
        z.append(
            "24 hours later is 03/13/2002. So today is 03/12/2002.")
        y.append("03")

        x.append(
            "\"Jane was born on the last day of Feburary in 'X'. Today is her 16-year-old birthday. Yesterday is 02/27/2017\" What is the answer of 'X'?")
        z.append(
            "Yesterday is 02/27/2017, so today is 02/28/2017, Jane was born on 02/28/2001.")
        y.append("2001")

    else:
        raise ValueError("dataset is not properly defined ...")

    # randomize order of the examples ...
    index_list = list(range(len(x)))
    random.shuffle(index_list)
    if args.FN != 0:
        index_list = index_list[:args.FN]
    # Concatenate demonstration examples ...
    demo_text = ""
    for i in index_list:
        if cot_flag:
            demo_text += "Q: " + x[i] + "\nA: " + z[i] + " " + \
                         args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
        else:
            demo_text += "Q: " + x[i] + "\nA: " + \
                         args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"

    return demo_text


def question_turn_decalrative(args, text, answer, answers_0, function, declarative):
    global new_question
    if 'Answer Choices' in text:
        text = text.split('Answer Choices')[0]
    try:
        if args.dataset in ("commonsensqa"):
            text = text.replace(',', '.')
            position_fullstop = text[::-1].find('.')

            question = text[len(text) - position_fullstop:]
            ts = text[:len(text) - position_fullstop]

            if ts[0] == ' ':
                ts = ts[1:]
            if ts[-1] != ' ':
                ts += ' '
            ts = ts.replace(' .', '.')
            if args.model == 'UL2':
                return ts, 'yes', "'{} The answer is {}' If the question and answer are changed into fluent declarative sentences: ".format(
                    question, answer)
            else:
                return ts, 'yes', "Q: Please change the questions and answers into a complete declarative sentences '{} The answer is {}'\nA: ".format(
                    question, answer)

        text = text.replace(',', '.')
        position_fullstop = text[::-1].find('.')

        question = text[len(text) - position_fullstop:]
        ts = text[:len(text) - position_fullstop]
        if args.dataset in ('bigbench_date'):
            declarative = question[17:-15] + ' is ' + answer + '.'
        else:
            if declarative == '':
                try:
                    declarative = function(args,
                                           "Q: Please change the questions and answers into a complete declarative sentences '{} The answer is {}'\nA: ".format(
                                               question, answer), args.max_length_cot, 0, 0, 1,'\n',is_turn_to_declarative=True)[0]
                except:

                    declarative = function(args,
                                               "Q: Please change the questions and answers into a complete declarative sentences '{} The answer is {}'\nA: ".format(
                                                   question, answer), args.max_length_cot, 0, 0, 1,'\n',is_turn_to_declarative=True)[0]

            else:
                if answers_0 in declarative:
                    declarative = declarative[:len(declarative) - declarative[::-1].find(answers_0[::-1]) - len(
                        answers_0)] + answer + declarative[len(declarative) - declarative[::-1].find(answers_0[::-1]):]
                else:
                    try:
                        declarative = function(args,
                                               "Q: Please change the questions and answers into a complete declarative sentences '{} The answer is {}'\nA: ".format(
                                                   question, answer), args.max_length_cot, 0, 0, 1,'\n',is_turn_to_declarative=True)[0]
                    except:
                        declarative = "{} The answer is {}.".format(question, answer)

        new_question_number = [s for s in re.findall(r'-?\d+\.?\d*', ts)]

        sentences, ans = [], []
        for nqn in range(len(new_question_number)):
            new_ts = ''
            number_find = False
            for i in ts.split('.'):
                if new_question_number[nqn] in i and number_find == False:
                    new_question = [p for p in i]
                    new_question[
                    i.find(new_question_number[nqn]):i.find(new_question_number[nqn]) + len(new_question_number[nqn])] = "'X'"
                    new_question = ''.join(new_question) + '.'
                    new_question.replace(' .', '.')
                    new_ts += new_question
                else:
                    new_ts += i + '.'
            new_ts = new_ts.replace('..', '.')

            if new_ts[0] == ' ':
                new_ts = new_ts[1:]
            if new_ts[-1] != ' ':
                new_ts += ' '
            new_ts = new_ts.replace(' .', '.')

            sentences.append('"' + new_ts + declarative + '"' + args.verifier_text)
            ans.append(new_question_number[nqn])
        return sentences[:3], ans[:3],declarative

    except:

        return '', '', ''


def create_verifier_demo_text_TF(args, cot_flag):
    x, z, y = [], [], []

    # example sentences ...
    if args.dataset in ("multiarith", "gsm8k", "addsub", "svamp", "singleeq"):

        x.append(
            "\'There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. The grove workers planted 4 trees today.\' Do it is correct (True or False)?")
        z.append(
            "If the Grove workers will plant 4 trees today and there will be 21 trees after they are done. 21 - 4 = 17, there are 17 trees in the grove, but actually there are 15 trees, 17 != 15, which is different from the theme.")
        y.append("False")

        x.append(
            "\'If there are 3 cars in the parking lot and 2 more cars arrive, There are 5 cars in the parking lot.\' Do it is correct (True or False)?")
        z.append(
            "If there will be 5 cars in the parking lot, subtract 2 cars that will arrive, 5 - 2 = 3, so there are 2 cars in the parking lot, which is consistent with the theme.")
        y.append("True")

        x.append(
            "\'Leah had 32 chocolates and her sister had 42. If they ate 35, they have 39 pieces left in total.\' Do it is correct (True or False)?")
        z.append(
            "If there are 39 pieces of chocolates and 35 pieces of chocolate are eaten, Leah and her sister have 39 + 35 = 74 in total. Her sister's had 42, so Leah had 74 - 42 = 32, which is consistent with the theme.")
        y.append("True")

        x.append(
            "\'Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. Jason gave Denny 6 lollipops.\' Do it is correct (True or False)?")
        z.append(
            "If Jason gave Denny 6 lollipops, and Jason now has 12 lollipops, so Jason originally had 6+12=18 lollipops, 18 != 20, which is different from the theme.")
        y.append("False")

        x.append(
            "\'Shawn has five toys. For Christmas, he got two toys each from his mom and dad. He has 9 toys now.\' Do it is correct (True or False)?")
        z.append(
            "If Shawn now has 9 toys and his parents gaven him two each, then he originally had 9 - 2 - 2 = 5, which is consistent with the theme.")
        y.append("True")

        x.append(
            "\'There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. There are 18 computers in the server room.\' Do it is correct (True or False)?")
        z.append(
            "Now there are 18 computers in the server room. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. So there were 18 - 20= -2 in the server room originally, -2 != 9, which is different from the theme.")
        y.append("False")

        x.append(
            "\'Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. He had 40 golf balls at the end of Wednesday.\' Do it is correct (True or False)?")
        z.append(
            "If Michael had 40 golf balls on Wednesday, he had 40+2=42 on Tuesday because he lost 2 golf balls on Wednesday. Due to lost 23 balls on Tuesday, he should have 42+23=65 on Monday, but in fact Michael has 58 golf balls original, which is different from the theme.")
        y.append("False")

        x.append(
            "\'Olivia has $23. She bought five bagels for $3 each.  She has 8 dollars left.\' Do it is correct (True or False)?")
        z.append(
            "If Olivia had $8 left and she bought five bagels for $3 each, so costs 5 * 3 = 15, so there was 8 + 15 = 23, which is consistent with the theme.")
        y.append("True")

    elif args.dataset in ("aqua"):
        x.append(
            "\"John found that the average of 15 numbers is 40. If 10 is added to each number then the mean of the numbers is 50.\" Do it is correct (True or False)?")
        z.append(
            "The new mean would be 50. The average of 15 numbers is 4, if 10 is added to each number, then the mean of the numbers also increases by 10. 50 - 40 = 10.")
        y.append("True")

        x.append("\"If a / b = 3/4 and 8a + 5b = 22, then the value of a is 3.\" Do it is correct (True or False)?")
        z.append(
            "If a is 3, a / b = 3/4, so b = 4. then 8a + 5b = 8 * 2 + 5 * 4 = 36, but 8a + 5b = 22")
        y.append("False")

        x.append(
            "\"A person is traveling at 20 km/hr and reached his destiny in 2.5 hr then find the distance is 65km.\" Do it is correct (True or False)?")
        z.append("If 65km is driven at 20km/hr, so the driving time is 65km / 20km/hr = 3.25h, but he destiny in 2.5 hr.")
        y.append("False")

        x.append(
            "\"There were 9 computers in the server room. Five more computers were installed each day, from monday to thursday. There are 29 computers in the server room.\" Do it is correct (True or False)?")
        z.append(
            "There are 29 computers in the server room. For each of 4 days, 5 more computers were added. 5 * 4 = 20 computers were added. So there were originally 9 computers. ")
        y.append("True")
    else:
        raise ValueError("dataset is not properly defined ...")

    # randomize order of the examples ...
    index_list = list(range(len(x)))
    random.shuffle(index_list)

    # Concatenate demonstration examples ...
    demo_text = ""
    for i in index_list:
        if cot_flag:
            demo_text += "Q: " + x[i] + "\nA: " + z[i] + " " + \
                         args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
        else:
            demo_text += "Q: " + x[i] + "\nA: " + \
                         args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"

    return demo_text
