import argparse
import logging
import torch
import random
import time
from tqdm import tqdm
import os
from utils import *


def log_data(text, path):
    with open(path + '/loggings.txt', 'a', encoding='utf-8') as f:
        f.write(text)
        print(text)
        f.write('\n')


def log_data_self(text, path):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(text)
        print(text)
        f.write('\n')


def log_start(MODEL, DATA, N, K, FN):
    log_name = MODEL + "_" + DATA + "_" + str(N) + "_" + str(K) + "_" + str(FN)
    try:
        os.mkdir('log/' + log_name)
    except:
        log_name += time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        os.mkdir('log/' + log_name)

    with open('log/' + log_name + '/python_file.py', 'a', encoding='utf-8') as f:
        with open(os.path.basename(__file__), 'r', encoding='utf-8') as f2:
            file = f2.read()
        f.write(file)

    path = 'log/' + log_name
    return path


class AverageMeter:  # 为了tqdm实时显示loss和acc
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = []
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum.append(val * n)
        self.count += n
        self.avg = sum(self.sum) / self.count


def main():
    args = parse_arguments()
    path = log_start(args.model, args.dataset, args.N, args.K, args.method)
    log_data('*****************************', path)
    print(args)
    log_data('*****************************', path)
    fix_seed(args.random_seed)

    # Initialize decoder class (load model and tokenizer) ...
    decoder = Decoder(args)

    log_data("setup data loader ...", path)
    dataloader = setup_data_loader(args)
    print_now()

    if args.method == "few_shot":
        demo = create_demo_text(args, cot_flag=False)
    elif args.method == 'verifier_cot':
        demo_F = create_demo_text(args, cot_flag=True)
        demo_B = create_verifier_demo_text(args, cot_flag=True)
        demo_F = demo_F.split('\n\n')[:-1]
        demo_B = demo_B.split('\n\n')[:-1]
    else:
        pass

    total = 0
    correct_list = []
    tk = tqdm(dataloader)
    accs = AverageMeter()
    accs_sc = AverageMeter()
    accs_avg = AverageMeter()
    accs_verifier = AverageMeter()
    loggings = []
    for i, data in enumerate(tk):
        log_data('*************************', path)
        log_data("{}st data".format(i + 1), path)

        # Prepare question template ...
        x, y = data

        max_length = args.max_length_cot if "cot" in args.method else args.max_length_direct
        if args.method == "few_shot_cot":
            x = "Q: " + x[0] + "\n" + "A:"
            y = y[0].strip()
            x = demo + x

            max_length = args.max_length_cot if "cot" in args.method else args.max_length_direct
            z = decoder.decode(args, x, max_length, i, 0, 1, '\n')[0]

            # Answer extraction for zero-shot-cot ...
            pred = z
            log_data(x + pred,path)

            # Clensing of predicted answer ...
            pred = answer_cleansing(args, pred)

            # Choose the most frequent answer from the list ...
            log_data("pred : {}".format(pred),path)
            log_data("GT : " + y,path)
            log_data('*************************',path)

            # Checking answer ...
            correct = (np.array([pred]) == np.array([y])).sum().item()
            correct_list.append(correct)
            total += 1  # np.array([y]).size(0)
        elif args.method == 'verifier_cot':
            declarative = ''
            answers = []
            verifier_acc_self_now = []
            log_data('Q: {}'.format(x[0]), path)
            log_data("GT : " + y[0], path)
            y = y[0].strip()
            x_F = "Q: " + x[0] + "\n" + "A:"
            x_F = '\n\n'.join(demo_F) + '\n\n' + x_F

            pred = decoder.decode(args, x_F, max_length, i, args.K, args.N, '\n')

            if len(pred) == 0:
                continue
            # Clensing of predicted answer ...

            for p_it in pred:
                p_item = answer_cleansing(args, p_it)
                try:
                    p_item = str(float(p_item))
                    y = str(float(y))
                except:
                    pass

                is_true = (np.array([p_item]) == np.array([y])).sum().item()
                log_data("{}: {}".format({0: False, 1: True}[is_true], p_it), path)
                if p_item != '':
                    if args.dataset in ('aqua', 'commonsensqa'):
                        choice = x[0].split('Answer Choices: ')[1].split('(')[1:]
                        choice = {i[0]: i[2:] for i in choice}
                        p_item = choice[p_item]

                    if p_item not in answers:
                        answers.append(p_item)
            if len(answers) == 0:
                accs.update(0)
                accs_verifier.update(0)
                correct_list.append(0)
                total += 1  # np.array([y]).size(0)
                tk.set_postfix(accs=accs.avg, verifier_acc=accs_verifier.avg)
                log_data('verifier:{} accs:{} '.format(0, 0), path)
                log_data('*************************', path)
                loggings.append({'question': data[0][0],
                                 'target': data[1][0],
                                 'CoT': '',
                                 'is_true': 0,
                                 'verifier': 0})

            else:
                if len(answers) == 1:
                    scores = {0: 1}
                    pred_verifier = []
                else:
                    scores = {i: 0 for i in range(len(answers))}
                    pred_verifier = {i: [] for i in range(len(answers))}
                    for A in range(len(answers)):

                        decl, answer, declarative = question_turn_decalrative(args, x[0], answers[A], answers[0],
                                                                              decoder.decode, declarative)
                        for d in range(len(decl)):
                            random.shuffle(demo_B)
                            x_B = '\n\n'.join(demo_B) + 'Q: ' + decl[d] + '\nA: '
                            try:
                                pred_v = decoder.decode(args, x_B, max_length, i, 0.2, 10, '\n\n')
                            except:
                                pred_v = [''] * 10
                            answers_verifier = []
                            for p in range(len(pred_v)):
                                p_item_v = answer_cleansing_verifier(args, pred_v[p])
                                try:
                                    answers_verifier.append(float(p_item_v))
                                except:
                                    try:
                                        answers_verifier.append(p_item_v)
                                    except:
                                        pass
                            try:
                                score = sum(np.array(answers_verifier) == np.array(float(answer[d])))
                            except:
                                try:
                                    score = sum(np.array(answers_verifier) == np.array(answer[d]))
                                except:
                                    score = 0
                            pred_verifier[A].append(pred_v)
                            scores[A] += score
                        try:
                            log_data('{} - {}: {}'.format(answers[A], scores[A], pred_v[0].replace('\n', ' ')), path)
                        except:
                            pass
                verifier_is_ture = list(scores.values())

                if args.dataset in ("aqua", "commonsensqa"):
                    answers_is_ture = (np.array(answers) == np.array([choice[y]])).tolist()
                    accs.update((np.array([answers[0]]) == np.array([choice[y]])).sum().item())
                    accs_verifier.update(
                        (np.array([answers[np.argmax(np.array(scores)).item()]]) == np.array([choice[y]])).sum().item())
                    correct = (np.array([answers[np.argmax(np.array(scores)).item()]]) == np.array(
                        [choice[y]])).sum().item()
                    log_data('verifier:{} accs:{} '.format(
                        (np.array([answers[np.argmax(np.array(scores)).item()]]) == np.array([choice[y]])).sum().item(),
                        (np.array([answers[0]]) == np.array([choice[y]])).sum().item()),
                        path)
                    log_data('*************************', path)
                    loggings.append({'question': data[0][0],
                                     'target': data[1][0],
                                     'CoT': pred_verifier,
                                     'is_true': (np.array(answers) == np.array([choice[y]] * len(answers))).tolist(),
                                     'verifier': (np.array([answers[np.argmax(np.array(scores)).item()]]) == np.array(
                                         [choice[y]])).sum().item()})

                else:
                    answers_is_ture = (np.array(answers) == np.array(y)).tolist()
                    accs.update((np.array([answers[0]]) == np.array([y])).sum().item())
                    accs_verifier.update((np.array([answers[np.argmax(np.array(verifier_is_ture)).item()]]) == np.array(
                        [y])).sum().item())
                    correct = (np.array([answers[np.argmax(np.array(verifier_is_ture)).item()]]) == np.array(
                        [y])).sum().item()
                    log_data('verifier:{} accs:{} '.format(
                        (np.array([answers[np.argmax(np.array(verifier_is_ture)).item()]]) == np.array(
                            [y])).sum().item(),
                        (np.array([answers[0]]) == np.array([y])).sum().item()),
                        path)
                    log_data('*************************', path)
                    loggings.append({'question': data[0][0],
                                     'target': data[1][0],
                                     'CoT': pred_verifier,
                                     'is_true': (np.array(answers) == np.array([y] * len(answers))).tolist(),
                                     'verifier': (np.array([answers[np.argmax(np.array(scores)).item()]]) == np.array(
                                         [y])).sum().item()})
                log_data_self(
                    '{}'.format(','.join([str(int(x)) for x in answers_is_ture])),
                    path + '/answers_acc.txt')
                log_data_self(
                    '{}'.format(','.join([str(int(x)) for x in verifier_is_ture])),
                    path + '/verifier_acc.txt')
                correct_list.append(correct)
                total += 1  # np.array([y]).size(0)
                tk.set_postfix(accs=accs.avg, verifier_acc=accs_verifier.avg)


        else:
            raise ValueError("method is not properly defined ...")

    # Calculate accuracy ...
    accuracy = (sum(correct_list) * 1.0 / total) * 100
    log_data("accuracy : {}".format(accuracy), path)
    if args.method == 'verifier_cot':
        log_data('accs:{}  self_consistency:{}  Top_N_acc:{}'.format(accs.avg, accs_sc.avg, accs_avg.avg), path)
        loggings = pd.DataFrame(loggings)
        loggings.to_csv(path + '/ls.csv')




def parse_arguments():
    parser = argparse.ArgumentParser(description="Reason with self-verification")

    parser.add_argument(
        "--api_log_file_name", type=str, default=None,
        help="mandatory argument ! json['i>=1']['j==1']['k={1,2}'][{'request', response'}]"
    )

    parser.add_argument("--random_seed", type=int, default=1, help="random seed")

    parser.add_argument(
        "--dataset", type=str, default="gsm8k",
        choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith", "strategyqa", "svamp", "singleeq",
                 "bigbench_date", "object_tracking", "coin_flip", "last_letters", "meddialog"],
        help="dataset used for experiment"
    )

    parser.add_argument("--minibatch_size", type=int, default=1, choices=[1],
                        help="minibatch size should be 1 because GPT-3 API takes only 1 input for each request")

    parser.add_argument("--max_num_worker", type=int, default=0, help="maximum number of workers for dataloader")

    parser.add_argument(
        "--model", type=str, default="codex",
        choices=["gpt3", "gpt3-medium", "gpt3-large", "gpt3-xl", "codex", "codex-001"],
        help="model used for decoding. Note that 'gpt3' are the smallest models."
    )

    parser.add_argument(
        "--method", type=str, default="verifier_cot",
        choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "verifier_cot", "verifier"], help="method"
    )
    parser.add_argument(
        "--cot_trigger_no", type=int, default=1,
        help="A trigger sentence that elicits a model to execute chain of thought"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=168,
        help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--max_length_direct", type=int, default=32,
        help="maximum length of output tokens by model for answer extraction"
    )
    parser.add_argument(
        "--limit_dataset_size", type=int, default=0,
        help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=4.0, help=""
    )
    parser.add_argument(
        "--log_dir", type=str, default="./log/", help="log directory"
    )
    parser.add_argument(
        "--N", type=int, default=5
    )
    parser.add_argument(
        "--K", type=int, default=0.3
    )
    parser.add_argument(
        "--FN", type=int, default=0, help="few-shot number"
    )

    args = parser.parse_args()

    if args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/test.json"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
    elif args.dataset == "gsm8k":
        args.dataset_path = "./dataset/grade-school-math/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "commonsensqa":
        args.dataset_path = "./dataset/CommonsenseQA/dev_rand_split.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
        args.plausible_answer_trigger = "Choose the most plausible answer from among choices A through E."
    elif args.dataset == "addsub":
        args.dataset_path = "./dataset/AddSub/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "multiarith":
        args.dataset_path = "./dataset/MultiArith/MultiArith.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/StrategyQA/task.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "svamp":
        args.dataset_path = "./dataset/SVAMP/SVAMP.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "singleeq":
        args.dataset_path = "./dataset/SingleEq/questions.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "bigbench_date":
        args.dataset_path = "./dataset/Bigbench_Date/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through F, the answer is"
    elif args.dataset == "object_tracking":
        args.dataset_path = "./dataset/Bigbench_object_tracking/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through C, the answer is"
    elif args.dataset == "coin_flip":
        args.dataset_path = "./dataset/coin_flip/coin_flip.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset == "meddialog":
        args.dataset_path = "./dataset/MedDialog/english-test.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"

    else:
        raise ValueError("dataset is not properly defined ...")

    # "Therefore, the answer ..." -> "The answer ..."
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger

    args.direct_answer_trigger_for_fewshot = "The answer is"
    if args.dataset in ("commonsensqa"):
        args.verifier_text = ' Judge whether this statement is normal (yes or no)'
    else:
        args.verifier_text = " What is the answer of 'X'?"
    if args.cot_trigger_no == 1:
        args.cot_trigger = "Let's think step by step."
    elif args.cot_trigger_no == 2:
        args.cot_trigger = "We should think about this step by step."
    elif args.cot_trigger_no == 3:
        args.cot_trigger = "First,"
    elif args.cot_trigger_no == 4:
        args.cot_trigger = "Before we dive into the answer,"
    elif args.cot_trigger_no == 5:
        args.cot_trigger = "Proof followed by the answer."
    elif args.cot_trigger_no == 6:
        args.cot_trigger = "Let's think step by step in a realistic way."
    elif args.cot_trigger_no == 7:
        args.cot_trigger = "Let's think step by step using common sense and knowledge."
    elif args.cot_trigger_no == 8:
        args.cot_trigger = "Let's think like a detective step by step."
    elif args.cot_trigger_no == 9:
        args.cot_trigger = "Let's think about this logically."
    elif args.cot_trigger_no == 10:
        args.cot_trigger = "Let's think step by step. First,"
    elif args.cot_trigger_no == 11:
        args.cot_trigger = "Let's think"
    elif args.cot_trigger_no == 12:
        args.cot_trigger = "Let's solve this problem by splitting it into steps."
    elif args.cot_trigger_no == 13:
        args.cot_trigger = "The answer is after the proof."
    elif args.cot_trigger_no == 14:
        args.cot_trigger = "Let's be realistic and think step by step."
    else:
        raise ValueError("cot_trigger_no is not properly defined ...")

    return args


if __name__ == "__main__":
    main()
