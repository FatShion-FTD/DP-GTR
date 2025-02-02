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
    original_input = "The capital of France is Paris."
    print('original input:', original_input)
    has = HaS('cuda')
    hide_input = has.hide(original_input)
    print('hide input:', hide_input)
    # prompt = "Translate the following text into English.\n %s\n" % hide_input
    hide_output = "英国首都是伦敦。"
    print('hide output:', hide_output)
    original_output = has.seek(hide_input, hide_output, original_input)
    print('original output:', original_output)
    