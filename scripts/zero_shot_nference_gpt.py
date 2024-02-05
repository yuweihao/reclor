import argparse
import openai
import json
import re
import os
from tqdm import tqdm
import numpy as np
from openai import OpenAI
client = OpenAI()

openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.api_key = "sk-xxxxxx"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--model", type=str, default="gpt-4-0613", help="OpenAI GPT model name"
    )
    parser.add_argument(
    "--results_file", type=str, default=None, help="results file name to save"
    )
    args = parser.parse_args()

    model = args.model
    if args.results_file:
        results_file = args.results_file
    else:
        results_file = f"reclor_test_{model}_zero_shot_results"
    
    results_file_json = results_file + ".json"
    if os.path.exists(results_file_json):
        with open(results_file_json, "r") as f:
            zero_shot_results = json.load(f)
    else:
        zero_shot_results = []

    with open("reclor_data/test.json", "r") as f:
        data = json.load(f)


    start = len(zero_shot_results)

    for i in tqdm(range(start, len(data))):
        question = data[i]['context'] + " " + data[i]['question'] + \
            "\nA. " + data[i]['answers'][0] + \
            "\nB. " + data[i]['answers'][1] + \
            "\nC. " + data[i]['answers'][2] + \
            "\nD. " + data[i]['answers'][3] + \
            "\nAmong A to D, the answer is: "
        
        messages = [
            {"role": "user", "content": question},
        ]

        response = client.chat.completions.create(
            model=model,
            max_tokens=32,
            temperature=0,
            seed=0,
            messages=messages)

        content = response.choices[0].message.content
        try: 
            pred = re.findall(r'A|B|C|D', content)[0]
        except:
            try_time = 1
            while try_time < 5:
                response = client.chat.completions.create(
                    model=model,
                    max_tokens=32,
                    temperature=1,
                    messages=messages)
                content = response.choices[0].message.content
                pred = re.findall(r'A|B|C|D', content)
                if len(pred) > 0:
                    pred = pred[0]
                    break
                else:
                    try_time += 1
                    print(f"Try {try_time} times for {i}th question")

        results = json.loads(response.model_dump_json())
        results['pred'] = pred
        zero_shot_results.append(results)

        with open(results_file_json, "w") as f:
            json.dump(zero_shot_results, f, indent=4)

    assert len(zero_shot_results) == len(data)
    map_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    results_npy = []
    for i in range(len(zero_shot_results)):
        results_npy.append(map_dict[zero_shot_results[i]['pred']])
    results_npy = np.array(results_npy)
    results_file_npy = results_file + ".npy"
    np.save(results_file_npy, results_npy)


if __name__ == "__main__":
    main()