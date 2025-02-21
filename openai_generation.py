from openai import AzureOpenAI
import json

json_data = json.load(open("OpenAI.json"))
endpoint = json_data["api_base"]
api_key = json_data["api_key"]
api_version = json_data["api_version"]
api_type = json_data["deployment_name"]
gpt35 = api_type["GPT3.5"]

# gets the API Key from environment variable AZURE_OPENAI_API_KEY
client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
deployment_name = gpt35

def prompt_template(doc):
    prompt = "Document: {}\nParaphrase of the document:".format(doc)
    return prompt

def dp_paraphrase(input, **kwargs):
    prompt = prompt_template(input)
    response = generate(prompt, **kwargs)
    return response
    

def generate(
    content,
    temperature=0.0,
    logits_dict=None,
    max_tokens=100,
    top_p=None,
    frequency_penalty=None,
    presence_penalty=None,
    stop=None,
    print_output=True,
):
    if print_output:
        print(content)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    ans = None
    try:
        args = {
            "model": deployment_name,
            "prompt": content,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if logits_dict != None:
            args["logit_bias"] = logits_dict
        if top_p != None:
            args["top_p"] = top_p
        if frequency_penalty != None:
            args["frequency_penalty"] = frequency_penalty
        if presence_penalty != None:
            args["presence_penalty"] = presence_penalty
        if stop != None:
            args["stop"] = stop

        response = client.completions.create(**args)
        ans = response.choices[0].text
        if print_output:
            print(f"{response.choices[0].text}")
    except Exception as e:
        ans = ""
        print(f"Error: {e}")
    if print_output:
        print("========================================")
    return ans
