from openai import AzureOpenAI
import json

# from pandarallel import pandarallel


# pandarallel.initialize(progress_bar = True, nb_workers=2)

json_data = json.load(open("OpenAI.json"))

endpoint = json_data["api_base"]
api_key = json_data["api_key"]
api_version = json_data["api_version"]
api_type = json_data["deployment_name"]
gpt35 = api_type["GPT3.5"]

# gets the API Key from environment variable AZURE_OPENAI_API_KEY
client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
deployment_name = gpt35


def openAI_BT_generate(content, from_language, to_language, temperature):
    prompt = (
        f"Content: {content}\nTranslate Content from {from_language} to {to_language}:"
    )
    print(f"Prompt: {prompt}")
    ans = None
    try:
        response = client.completions.create(
            model=deployment_name,
            prompt=prompt,
            temperature=temperature,
            max_tokens=2500,
        )
        ans = response.choices[0].text
        print(f"Response: {response.choices[0].text}")
    except Exception as e:
        ans = ""
        print(f"Error: {e}")
    print("========================================")
    return ans


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


def openAI_generate_parallel_df(df, df_field, temperature, save_path=None):
    # Review: {review}\nParaphrase of the review:
    translated_df = df[df_field].parallel_apply(
        lambda x: generate(
            content=f"Review: {x}\nParaphrase of the review:", temperature=temperature
        )
    )
    if save_path != None:
        translated_df.to_csv(save_path, index=False)
    return translated_df


def openAI_generate_df(
    df, df_field, from_language, to_language, temperature, save_path=None
):
    translated_df = df[df_field].apply(
        lambda x: openAI_BT_generate(
            content=x,
            from_language=from_language,
            to_language=to_language,
            temperature=temperature,
        )
    )
    if save_path != None:
        translated_df.to_csv(save_path, index=False)
    return translated_df


if __name__ == "__main__":
    # positive: 31587, negative: 43324, world: 14957, sports: 84660, business: 27243, technology: 59342
    # logits_bias_dict = {"14957": 100, "84660": 100, "27243": 100, "59342": 100}
    logits_bias_dict = {"31587": -100}
    # news = "Wall St. Bears Claw Back Into the Black (Reuters) Reuters - Short-sellers, Wall Street's dwindling\band of ultra-cynics, are seeing green again."
    news = "You are good. The world is beautiful."
    content = f"Article: {news} Answer: "
    response = generate(content, 0.0, logits_bias_dict, 100)
    # response = generate(content, 0.0)

    # content = "The discussion of the merits of DP-MLM must also be met with its remaining limitations. As our rewriting mechanism leveraging DP-MLM relies on token-level DP replacements, the primary limitation comes with the initial inability to rewrite sentences with differing lengths from the original texts."
    # print(f"Oringinal Content: {content}")
    # response_en_to_ch = openAI_BT_generate(content, "English", "Chinese", 0.5)
    # print(f"English to Chinese: {response_en_to_ch.choices[0].text}")
    # response_ch_to_en = openAI_BT_generate(
    #     response_en_to_ch.choices[0].text, "Chinese", "English", 1.75
    # )
    # print(f"Chinese to English: {response_ch_to_en.choices[0].text}")
