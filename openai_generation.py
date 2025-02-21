from openai import AzureOpenAI
import json

json_data = json.load(open("OpenAI.json"))
endpoint = json_data["api_base"]
api_key = json_data["api_key"]
api_version = json_data["api_version"]
deployment_name = json_data["deployment_name"]["GPT3.5"]

client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)

def prompt_template(doc):
    return f"Document: {doc}\nParaphrase of the document:"

def dp_paraphrase(text, **kwargs):
    prompt = prompt_template(text)
    return generate(prompt, **kwargs)
     
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
    try:
        args = {
            "model": deployment_name,  
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if logits_dict is not None:
            args["logit_bias"] = logits_dict
        if top_p is not None:
            args["top_p"] = top_p
        if frequency_penalty is not None:
            args["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            args["presence_penalty"] = presence_penalty
        if stop is not None:
            args["stop"] = stop

        response = client.chat.completions.create(**args)
        ans = response.choices[0].message.content
        if print_output:
            print(ans)
    except Exception as e:
        ans = ""
        print(f"Error: {e}")
    if print_output:
        print("========================================")
    return ans

if __name__ == "__main__":
    input_text = "In which year, john f. kennedy was assassinated?"
    rewrites = dp_paraphrase(input_text, temperature=1.0)
    print(rewrites)
