import argparse
import logging
import torch
import random
import time
from tqdm import tqdm
import os
from utils import *
import openai

import time
def print_stream(text):
    for i in text:
        print(i,end='',flush=True)
        time.sleep(0.02)
    print('\n',end='')

def main():
    args = parse_arguments()
    fix_seed(args.random_seed)

    # Initialize decoder class (load model and tokenizer) ...
    decoder = Decoder(args)

    demo_F = create_demo_text(args, cot_flag=True)
    demo_B = create_verifier_demo_text(args, cot_flag=True)
    demo_F = demo_F.split('\n\n')[:-1]
    demo_B = demo_B.split('\n\n')[:-1]


    while True:
        x = input("\033[0;34mWhat do you want to ask me? :)\n[QUESTION] \033[0m")
        if 'break' == x.lower():
            break
        # Prepare question template ...
    
        max_length = args.max_length_cot
        declarative = ''
        answers = []
    
        x_F = "Q: " + x + "\n" + "A:"
        x_F = '\n\n'.join(demo_F) + '\n\n' + x_F
    
        print_stream("\033[0;33m[INFO] Setting Temperature = \033[0m{}".format(args.K))
        print_stream("\033[0;33m[INFO] Setting Candidate Answer Number = \033[0m{}".format(args.N))
        print('')
        print_stream("\033[0;33m[INFO] Now Model is Generating Candidate Answers...\033[0m")
    

        pred = decoder.decode(args, x_F, max_length, 0, args.K, args.N, '\n')
    
        for p_it in pred:
            p_item = answer_cleansing(args, p_it)
            try:
                p_item = str(float(p_item))
            except:
                pass
    
            if p_item != '':
                if p_item not in answers:
                    answers.append(p_item)

        if len(answers) == 0:
            print_stream("\033[0;31m[ERROR] The Candidate Answer is None!!!\033[0m")
            print_stream("\033[0;31m[ERROR] Break Now! /(ㄒoㄒ)/~~\033[0m")
            print('')
    
        else:
    
            if len(answers) == 1:
                print_stream("\033[0;32m[ACCEPT] The Candidate Answer's number is 1, so the answer is:\033[0m")
                print_stream("\033[0;32m[ACCEPT] {}\033[0m".format(answers[0]))
                print('')
    
            else:
                print_stream("\033[0;33m[INFO] Now Model is Self-Verificating the Candidate Answers...\033[0m")
                scores = {i: 0 for i in range(len(answers))}
                pred_verifier = {i: [] for i in range(len(answers))}
                for A in range(len(answers)):
    
                    decl, answer, declarative = question_turn_decalrative(args, x, answers[A], answers[0],
                                                                            decoder.decode, declarative)
                    for d in range(len(decl)):
                        random.shuffle(demo_B)
                        x_B = '\n\n'.join(demo_B) + 'Q: ' + decl[d] + '\nA: '
                        try:
                            pred_v = decoder.decode(args, x_B, max_length, 0, 0.4, 10, '\n\n')
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
                        print_stream(f"\033[0;36m[Self-Verification SCORE] The Candidate Answer “{answers[A]}” :\033[0m")
                        print_stream(f"\033[0;36m[Self-Verification SCORE] {'█'*scores[A]}: {str(scores[A])}\033[0m")
    
                    except:
                        pass
                verifier_scores = list(scores.values())
            
                for i in range(len(verifier_scores)):
                    if verifier_scores[i] == max(verifier_scores):
                        print_stream(f"\033[0;32m[ACCEPT] The Best Answer is:\033[0m")
                        print_stream(f"\033[0;32m[ACCEPT] {answers[i]}\033[0m")
                        print('')
                        break

            





def parse_arguments():
    parser = argparse.ArgumentParser(description="Reason with self-verification")

    parser.add_argument("--random_seed", type=int, default=1, help="random seed")

    parser.add_argument(
        "--model", type=str, default="text-003",
        choices=["gpt3", "gpt3-medium", "gpt3-large", "gpt3-xl", "codex", "codex-001","text-003"],
        help="model used for decoding. Note that 'gpt3' are the smallest models."
    )

    parser.add_argument(
        "--max_length_cot", type=int, default=168,
        help="maximum length of output tokens by model for reasoning extraction"
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
        "--K", type=int, default=0.6
    )
    parser.add_argument(
        "--FN", type=int, default=0, help="few-shot number"
    )

    args = parser.parse_args()
    args.dataset = 'gsm8k'
    args.direct_answer_trigger_for_fewshot = "The answer is"
    args.method = 'verifier_cot'
    args.verifier_text = " What is the answer of 'X'?"
    return args


if __name__ == "__main__":
    main()
