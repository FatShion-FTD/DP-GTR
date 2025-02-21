from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch


class HaS:
    def __init__(self, device):
        self.tokenizer = AutoTokenizer.from_pretrained("SecurityXuanwuLab/HaS-820m")
        self.model = AutoModelForCausalLM.from_pretrained("SecurityXuanwuLab/HaS-820m").to(device)
        self.device = device
        self.is_debug = False
        
    def hide(self, original_input):
        hide_model = self.model
        tokenizer = self.tokenizer
        
        hide_template = """<s>Paraphrase the text:%s\n\n"""
        input_text = hide_template % original_input
        inputs = tokenizer(input_text, return_tensors='pt').to(hide_model.device)
        pred = hide_model.generate(
            **inputs, 
            generation_config=GenerationConfig(
                max_new_tokens = int(len(inputs['input_ids'][0]) * 1.3),
                do_sample=False,
                num_beams=3,
                repetition_penalty=5.0,
                ),
            )
        pred = pred.cpu()[0][len(inputs['input_ids'][0]):]
        hide_input = tokenizer.decode(pred, skip_special_tokens=True)
        return hide_input
    
    def custom_hide(self, prompt, context, hide_template=None):
        hide_model = self.model
        tokenizer = self.tokenizer
        
        if hide_template is None:
            hide_template = """
            <s>Paraphrase the following text:
            [QUESTION]
            %s
            [/QUESTION]
            [OPTIONS]
            %s
            [/OPTIONS]
            </s>
            """
        input_text = hide_template % (prompt, context)
        inputs = tokenizer(input_text, return_tensors='pt').to(hide_model.device)
        pred = hide_model.generate(
            **inputs, 
            generation_config=GenerationConfig(
                max_new_tokens = int(len(inputs['input_ids'][0]) * 1.3),
                do_sample=False,
                num_beams=3,
                repetition_penalty=5.0,
                ),
            )
        pred = pred.cpu()[0][len(inputs['input_ids'][0]):]
        hide_input = tokenizer.decode(pred, skip_special_tokens=True)
        return hide_input
    

    def seek(self, hide_input, hide_output, original_input):
        seek_model = self.model
        tokenizer = self.tokenizer
        
        seek_template = """<s>Convert the text:\n%s\n\n%s\n\nConvert the text:\n%s\n\n"""
        input_text = seek_template % (hide_input, hide_output, original_input)
        inputs = tokenizer(input_text, return_tensors='pt').to(seek_model.device)
        pred = seek_model.generate(
            **inputs, 
            generation_config=GenerationConfig(
                max_new_tokens = int(len(inputs['input_ids'][0]) * 1.3),
                do_sample=False,
                num_beams=3,
                ),
            )
        pred = pred.cpu()[0][len(inputs['input_ids'][0]):]
        original_output = tokenizer.decode(pred, skip_special_tokens=True)
        return original_output



if __name__ == "__main__":
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    has = HaS(device)
    test_question = "what is the date mentioned in this letter?"
    test_list = [ "Confidential", "..", "..", "RJRT", "PR", "APPROVAL", "DATE", ":", "1/8/13", "Ru", "alAs", "PROPOSED", "RELEASE", "DATE:", "for", "response", "FOR", "RELEASE", "TO:", "CONTACT:", "P.", "CARTER", "ROUTE", "TO", "Initials", "pate", "Peggy", "Carter", "Ac", "Maura", "Payne", "David", "Fishel", "Tom", "GRISCom", "Diane", "Barrows", "Ed", "Blackmer", "Tow", "Rucker", "TR", "Return", "to", "Peggy", "Carter,", "PR,", "16", "Reynolds", "Building", "51142", "3977", ".", ".", "Source:", "https://www.industrydocuments.ucsf.edu/docs/xnb10037" ]
    prompt = ""
    for word in test_list:
        prompt += f"{word}, "
        
    hide_input = has.hide(prompt[:-2])
    print(f"Hide Input Prompt1: {hide_input}")

    hide_input = has.hide(test_question)
    print(f"Hide Input Prompt2: {hide_input}")

    hide_input = has.custom_hide(test_question, prompt)

    print(f"Hide Input Prompt3: {hide_input}")
    
    hide_input = has.hide(f"{test_question}\n{prompt}")
    
    print(f"Hide Input Prompt4: {hide_input}")
    
    
    