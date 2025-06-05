from transformers import StoppingCriteriaList, StoppingCriteria
import openai
import os
import requests
if "OPENAI_API_KEY" in os.environ:
    openai.api_key = os.environ["OPENAI_API_KEY"]
def generate_from_bloom(model, tokenizer, query, max_tokens):
    encoded_input = tokenizer(query, return_tensors='pt')
    stop = tokenizer("[PLAN END]", return_tensors='pt')
    stoplist = StoppingCriteriaList([stop])
    output_sequences = model.generate(input_ids=encoded_input['input_ids'].cuda(), max_new_tokens=max_tokens,
                                      temperature=0, top_p=1)
    return tokenizer.decode(output_sequences[0], skip_special_tokes=True)


def send_query(query, engine, max_tokens, model=None, stop="[STATEMENT]"):
    max_token_err_flag = False
    if engine == 'bloom':

        if model:
            response = generate_from_bloom(model['model'], model['tokenizer'], query, max_tokens)
            response = response.replace(query, '')
            resp_string = ""
            for line in response.split('\n'):
                if '[PLAN END]' in line:
                    break
                else:
                    resp_string += f'{line}\n'
            return resp_string
        else:
            assert model is not None
    elif engine == 'finetuned':
        if model:
            try:
                response = openai.Completion.create(
                    model=model['model'],
                    prompt=query,
                    temperature=0,
                    max_tokens=max_tokens,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=["[PLAN END]"])
            except Exception as e:
                max_token_err_flag = True
                print("[-]: Failed GPT3 query execution: {}".format(e))
            text_response = response["choices"][0]["text"] if not max_token_err_flag else ""
            return text_response.strip()
        else:
            assert model is not None
    # elif engine == 'ollama':
    #     try:
    #         response = requests.post(
    #             "http://localhost:11434/api/generate",
    #             json={
    #                 "model": "qwen:7b",  # fixed as per your directive
    #                 "prompt": query,
    #                 "stream": False,
    #                 "options": {
    #                     "num_predict": max_tokens  # conform to `max_tokens` use
    #                 }
    #             }
    #         )
    #         response.raise_for_status()
    #         json_response = response.json()

    #         if "response" not in json_response:
    #             raise ValueError("Missing 'response' key in Ollama output.")

    #         raw = json_response["response"]

    #         # Extract between delimiters
    #         if "[PLAN BEGIN]" in raw and "[PLAN END]" in raw:
    #             raw = raw.split("[PLAN BEGIN]", 1)[1].split("[PLAN END]", 1)[0].strip()
    #         else:
    #             # print("[-]: PLAN delimiters not found in Ollama response.")
    #             # raw = ""
    #             raw = raw.strip()

    #         # Remove query echo if present
    #         if raw.startswith(query):
    #             raw = raw[len(query):].strip()

    #         # Clean lines
    #         cleaned_lines = []
    #         for line in raw.split("\n"):
    #             line = line.strip().strip(":.")
    #             if line:
    #                 cleaned_lines.append(line)

    #         text_response = "\n".join(cleaned_lines)

    #     except Exception as e:
    #         max_token_err_flag = True
    #         print("[-]: Failed Ollama query execution: {}".format(e))
    #         text_response = ""

    #     return text_response

    elif engine == 'ollama':
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen:7b",  # or other model as needed
                    "prompt": query,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens  # equivalent to OpenAI's max_tokens
                    }
                }
            )
            response.raise_for_status()
            json_response = response.json()

            if "response" not in json_response:
                raise ValueError("Missing 'response' key in Ollama output.")

            raw = json_response["response"]

            # Extract text between [PLAN BEGIN] and [PLAN END] if available
            if "[PLAN BEGIN]" in raw and "[PLAN END]" in raw:
                raw = raw.split("[PLAN BEGIN]", 1)[1].split("[PLAN END]", 1)[0].strip()
            else:
                print("[-]: PLAN delimiters not found in Ollama response.")
                raw = raw.strip()

            # Remove query echo if present
            if raw.startswith(query):
                raw = raw[len(query):].strip()

            # Clean each line
            cleaned_lines = []
            for line in raw.split("\n"):
                line = line.strip().strip(":").strip(".")
                if line:
                    cleaned_lines.append(line)

            text_response = "\n".join(cleaned_lines)

        except Exception as e:
            max_token_err_flag = True
            print("[-]: Failed Ollama query execution: {}".format(e))
            text_response = ""

        return text_response




    elif '_chat' in engine:
        
        eng = engine.split('_')[0]
        # print('chatmodels', eng)
        messages=[
        {"role": "system", "content": "You are the planner assistant who comes up with correct plans."},
        {"role": "user", "content": query}
        ]
        try:
            response = openai.ChatCompletion.create(model=eng, messages=messages, temperature=0)
        except Exception as e:
            max_token_err_flag = True
            print("[-]: Failed GPT3 query execution: {}".format(e))
        text_response = response['choices'][0]['message']['content'] if not max_token_err_flag else ""
        return text_response.strip()        
    else:
        try:
            response = openai.Completion.create(
                model=engine,
                prompt=query,
                temperature=0,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop)
        except Exception as e:
            max_token_err_flag = True
            print("[-]: Failed GPT3 query execution: {}".format(e))

        text_response = response["choices"][0]["text"] if not max_token_err_flag else ""
        return text_response.strip()