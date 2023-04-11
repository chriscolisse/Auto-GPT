import os
import re

from tiktoken import get_encoding
import openai
from dotenv import load_dotenv
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from nltk.tokenize import sent_tokenize

# Modify all of this to compress on a file-per-file basis.
# group files together based on size

path = Path(os.getcwd())
if '.env' not in os.listdir(path):
    path = Path(path.parent)

path = path.as_uri()
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_key

# make a task dispatcher GPT helper that takes the initial prompt and breaks it down into smaller parts, and then saves each prompt along with detailed yet concise steps into text files in the directory "tasks"
# then the manager GPT (Jippitty) will purge its short term memory each time a subtask is successfully completed, notes that Jippitty has taken on the task, the process, relevant info, key points, will be condensed by the summarizer team, and fed into the context for the next task.
#

import tensorflow_hub as hub
import tensorflow

import nltk

nltk.download('punkt')


def sentence_embeddings_USE(sentences, model_url="https://tfhub.dev/google/universal-sentence-encoder/4"):
    embed = hub.load(model_url)
    embeddings = embed(sentences).numpy()
    return embeddings


import torch
import numpy as np
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import AgglomerativeClustering


def check_tokens(text):
    enc = get_encoding("cl100k_base")
    return len(enc.encode(text))


def sentence_embeddings(sentences, model_name="allenai/longformer-base-4096"):
    # if not torch.cuda.is_available():
    #     return sentence_embeddings_USE(sentences)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=4096)

        # Move the inputs to the same device as the model
        inputs = {key: tensor.to(device) for key, tensor in inputs.items()}

        outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy())

    return np.vstack(embeddings)


def group_text(text, token_limit=1750):
    sentences = sent_tokenize(text)

    # Create sentence embeddings
    embeddings = sentence_embeddings(sentences)

    print('1234')
    # Create a similarity matrix
    similarity_matrix = np.inner(embeddings, embeddings)
    print('5678')
    # Perform clustering based on the similarity matrix
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.1, affinity="precomputed",
                                         linkage="average")
    labels = clustering.fit_predict(1 - similarity_matrix)

    # Group sentences into coherent chunks
    chunks = {label: [] for label in np.unique(labels)}
    for i, sentence in enumerate(sentences):
        chunks[labels[i]].append(sentence)

    # Ensure that chunks are within the token limit
    final_chunks = []
    for chunk in chunks.values():
        current_chunk = []
        current_tokens = 0
        for sentence in chunk:
            sentence_tokens = check_tokens(sentence)
            if current_tokens + sentence_tokens <= token_limit:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
            else:
                final_chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens

        if current_chunk:
            final_chunks.append(" ".join(current_chunk))

    return final_chunks


def gpt_3_5_compress(text):
    print("\n\n\n ...COMPRESSING...\n\n\n")
    compression_prompt = "FUNCTION GOAL: Max compress. Use symbols/lang/code/shorthand. Short concepts. Minimum words. Replace and " \
                         "group. No redundancy or repeat. Preserve all info. Lossless. Self-recover only. " \
                         "CONSIDERATIONS:\n" \
                         "- Apply the previous compression instructions to all input text, and return compression as output." \
                         "-  only with the output compressed text, nothing more. " \
                         "- If you need to compress code, start by removing any comments." \
                         "- A GPT-4 LLM should be able to reconstruct, at minimum, 95% of the meaning of the original text. It is important to maximize conceptual and general meaning." \
                         "OUTPUT FORMAT (replace <...> brackets and contents with real file info):" \
                         "filename: <filename.ext>\n" \
                         "compressed text: <compressed input text in bullet point form>\n" \
                         "*- It is possible that you will summarize some of your previous summaries. In such a case, only apply the above format formalities on a once per file basis." \
                         "*if, for some reason you cannot compress the text, return the text as-is. If the imput is empty, respond with \"\".\n \*"

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": compression_prompt},
            {"role": "user", "content": text}
        ]
    )
    print("\n\n\n ...COMPRESSING...\n\n\n")
    print(completion['choices'][0]['message']['content'])
    return completion['choices'][0]['message']['content']


def gpt_3_5_stitch(text_list: list):
    print("\n\n\n ...STITCHING...\n\n\n")
    stitch_prompt = "You are a python function that takes in a list of texts, and glues them together to form a " \
                    "seamless output. You must not change the content of the texts - you must only put them " \
                    "together in a way that makes sense. Your typehints are: def stitch(text:list[str])->str."
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": stitch_prompt},
            {"role": "user", "content": str(text_list)}
        ]
    )
    print("\n\n\n ...STITCHING...\n\n\n")
    print(completion['choices'][0]['message']['content'])
    return completion['choices'][0]['message']['content']


# def gpt_3_5_validator(text: str, initial_text: str, splitter=None) -> str:
#     splitter = "**" if not splitter else splitter
#
#     while not end_loop:
#         start_bot = False
#
#         # implement"" these in classes so that I can get information about the class when passing as objects. Makes it
#         # easier to pass in agents in a function and have the function detect what type of bot it is. This can reduce
#         # need to pass in many arguments and avoid writing the same code many times.
#         print("\n\n\n ..VALIDATING...\n\n\n")
#         validate_prompt = "Validate compression/stitching GPT-3.5 agent output by comparing input to initial text. " \
#                           "Preserve meaning, return VALIDATED OUTPUT, or use NEW_AGENT if adjustments needed. Response formats: **VALIDATED " \
#                           f"OUTPUT: <output> {splitter} FINAL ANSWER {splitter} OR {splitter}COMMAND: NEW_AGENT {splitter} ARGS: <initial text>" \
#                           f"{splitter}. Respond with response format only. Preserve filenames in output."
#
#
#         messages = [
#             {"role": "system", "content": validate_prompt},
#             {"role": "system", "content": "INITIAL TEXT (FROM BOT): \n" + initial_text + "\nBOT TEXT:\n" + str(text)}
#         ]
#
#         if extra_mem:
#             messages = [*messages, *extra_mem]
#
#         completion = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=messages
#         )
#         completion = completion['choices'][0]['message']['content']
#         end_loop = "FINAL ANSWER" in completion or "VALIDATED OUTPUT:" in completion
#         start_bot = True if (
#                     (("COMMAND" in completion) and ("ARGS" in completion)) or ("START BOT" in completion)) else False
#         print(end_loop, start_bot)
#         print(completion)
#         if end_loop and start_bot:
#             x_mem = [{"role": "assistant", "content": completion}, {"role": "system",
#                                                                     "content": f"Your output is confusing. You used both the COMMAND format, and the FINAL ANSWER format. Remember, you must respect ONE of the two following formats: {splitter}VALIDATED OUTPUT: <final validated output> {splitter} FINAL ANSWER {splitter}\n **OR**\n{splitter}COMMAND: NEW_COMPRESSION_AGENT {splitter} ARGS: <enter the initial text here>{splitter}\n"}]
#             extra_mem = x_mem if not extra_mem else extra_mem + x_mem
#
#         if end_loop:
#             print(completion.split(splitter))
#             try:
#                 return completion.split(splitter)[0].split("VALIDATED OUTPUT:")[1]
#             except:
#                 try:
#                     return completion.split(splitter)[1].split("VALIDATED OUTPUT:")[1]
#                 except Exception as e:
#                     raise e
#
#         x_mem = None
#
#         if start_bot:
#             while True:
#                 agent_command = None
#                 agent_kwargs = {}
#                 text_val = completion.split(splitter)
#                 # ... rest of the code
#
#                 if not (kwargs["args"] and agent_command):
#                     completion = gpt_3_5_validator(completion, initial_text=initial_text, extra_mem=x_mem,
#                                                    args=agent_kwargs)
#                     continue
#                 else:
#                     compress = gpt_3_5_compress(initial_text)
#                     text = compress
#                     initial_text = initial_text
#                     extra_mem = None
#                     break
#         else:
#             x_mem = [{"role": "assistant", "content": completion}, {"role": "system",
#                                                                     "content": "Your output could not be parsed. Please make sure you have responded in the proper format."}]
#             text = completion
#             initial_text = initial_text
#             extra_mem = x_mem
#
#
#     return text


def split_by_2(text):
    output = text.split(" ")
    if abs((len(output) / 2) - (round(len(output) / 2))) != 0:
        return output[:int(len(output) - 1 / 2)], output[int(len(output) - 1 / 2):]
    return output[:int(len(output) / 2)], output[int(len(output) / 2):]


def token_check_and_compress(text, compression=False, token_limit=1750):
    try:
        tokens = check_tokens(text)
        if re.match("COMPRESSED TEXT FILES:\n", text):
            text = text.replace("COMPRESSED TEXT FILES:\n", "")
    except Exception as e:
        if isinstance(text, list):
            # text = str(text)
            text = gpt_3_5_stitch(text)
            tokens = check_tokens(text)
        else:
            return "invalid " + str(text)

    if tokens > token_limit:
        split_text = split_by_2(text)
        split_1, split_2 = " ".join(split_text[0]), " ".join(split_text[1])
        list_compress = [token_check_and_compress(split_1, compression=True),
                         token_check_and_compress(split_2, compression=True)]
        return token_check_and_compress(list_compress)
    # if compression:
    #    # print('\n\n>>>>>\n', text, '\n')
    #     text = gpt_3_5_validator(gpt_3_5_compress(text), initial_text=text)
    # return text


def get_cwd_file_contents(path=None, delimiter="<>", workspace=True):
    print(path)
    output = ''
    if not path or path == "auto_gpt_workspace":
        path = os.getcwd() + "\\auto_gpt_workspace"

    if path and workspace:
        if os.getcwd() not in path and "auto_gpt_workspace" not in path:
            path = os.getcwd() + "\\auto_gpt_workspace\\" + path
        elif os.getcwd() not in path:
            path = os.getcwd() + f"\\{path}"
        else:
            path = path

    print('\n', path, '\n')

    files = os.listdir(path)
    if not files:
        return "No files files or folders present."
    for file in files:
        file = path + "\\" + file
        if os.path.isfile(file):
            with open(file, 'r', encoding="utf-8") as f:
                name = f.name.split('\\')[-1]
                output += f"\n{delimiter}\nFile Name:\n{name}\\nContents:\n{f.read()}\n{delimiter}\n"

    return token_check_and_compress(output, files)


if __name__ == "__main__":  # Example usage
    import os

    path = Path(os.getcwd())


    def run_in_docker(file, *args, **kwargs):
        import docker

        client = docker.from_env()

        # You can replace 'python:3.8' with the desired Python image/version
        # You can find available Python images on Docker Hub:
        # https://hub.docker.com/_/python
        workspace_folder = path.parent
        container = client.containers.run(
            'python:3.10',
            f'pip install  -r requirements.txt&&pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118&&python {file}',
            volumes={
                os.path.abspath(workspace_folder): {
                    'bind': '/workspace',
                    'mode': 'ro'}},
            working_dir='/workspace',
            stderr=True,
            stdout=True,
            detach=True,
        )

        output = container.wait()
        logs = container.logs().decode('utf-8')
        container.remove()

        # print(f"Execution complete. Output: {output}")
        # print(f"Logs: {logs}")

        return logs


    print(check_tokens("""‘It would be enough to turn any boy’s
head. Famous before he can walk and talk! Famous for something
he won’t even remember! Can’t you see how much better off he’ll
be, growing up away from all that until he’s ready to take it?’
Professor McGonagall opened her mouth, changed her mind,
swallowed and then said, ‘Yes – yes, you’re right, of course. But
how is the boy getting here, Dumbledore?’ She eyed his cloak
suddenly as though she thought he might be hiding Harry
underneath it. ‘Hagrid’s bringing him.’
‘You think it – wise – to trust Hagrid with something as important as this?’
‘I would trust Hagrid with my life,’ said Dumbledore. ‘I’m not saying his heart isn’t in the right place,’ said Professor
McGonagall grudgingly, ‘but you can’t pretend he’s not careless. He does tend to – what was that?’
A low rumbling sound had broken the silence around them. It
grew steadily louder as they looked up and down the street for
some sign of a headlight; it swelled to a roar as they both looked
up at the sky – and a huge motorbike fell out of the air and landed
on the road in front of them. If the motorbike was huge, it was nothing to the man sitting
astride it. He was almost twice as tall as a normal man and at least
five times as wide. He looked simply too big to be allowed, and so
wild – long tangles of bushy black hair and beard hid most of his
face, he had hands the size of dustbin lids and his feet in their
leather boots were like baby dolphins. In his vast, muscular arms
he was holding a bundle of blankets. ‘Hagrid,’ said Dumbledore, sounding relieved. ‘At last. And
where did you get that motorbike?’
‘Borrowed it, Professor Dumbledore, sir,’ said the giant, climbing
carefully off the motorbike as he spoke. ‘Young Sirius Black lent it
me. I’ve got him, sir.’
‘No problems, were there?’
‘No, sir – house was almost destroyed but I got him out all
right before the Muggles started swarmin’ around. He fell asleep
as we was flyin’ over Bristol.’
Dumbledore and Professor McGonagall bent forward over the
bundle of blankets. Inside, just visible, was a baby boy, fast asleep. Under a tuft of jet-black hair over his forehead they could see a
 THE BOY WHO LIVED 17
curiously shaped cut, like a bolt of lightning. ‘Is that where –?’ whispered Professor McGonagall. ‘Yes,’ said Dumbledore. ‘He’ll have that scar for ever.’
‘Couldn’t you do something about it, Dumbledore?’
‘Even if I could, I wouldn’t. Scars can come in useful. I have
one myself above my left knee which is a perfect map of the
London Underground. Well – give him here, Hagrid – we’d better
get this over with.’
Dumbledore took Harry in his arms and turned towards the
Dursleys’ house. ‘Could I – could I say goodbye to him, sir?’ asked Hagrid. He bent his great, shaggy head over Harry and gave him what
must have been a very scratchy, whiskery kiss. Then, suddenly,
Hagrid let out a howl like a wounded dog. ‘Shhh!’ hissed Professor McGonagall. ‘You’ll wake the Muggles!’
‘S-s-sorry,’ sobbed Hagrid, taking out a large spotted handkerchief and burying his face in it. ‘But I c-c-can’t stand it – Lily an’
James dead – an’ poor little Harry off ter live with Muggles –’
‘Yes, yes, it’s all very sad, but get a grip on yourself, Hagrid, or
we’ll be found,’ Professor McGonagall whispered, patting Hagrid
gingerly on the arm as Dumbledore stepped over the low garden
wall and walked to the front door. He laid Harry gently on the
doorstep, took a letter out of his cloak, tucked it inside Harry’s
blankets and then came back to the other two. For a full minute
the three of them stood and looked at the little bundle; Hagrid’s
shoulders shook, Professor McGonagall blinked furiously and the
twinkling light that usually shone from Dumbledore’s eyes seemed
to have gone out. ‘Well,’ said Dumbledore finally, ‘that’s that. We’ve no business
staying here. We may as well go and join the celebrations.’
‘Yeah,’ said Hagrid in a very muffled voice. ‘I’d best get
this bike away. G’night, Professor McGonagall – Professor
Dumbledore, sir.’
Wiping his streaming eyes on his jacket sleeve, Hagrid swung
himself on to the motorbike and kicked the engine into life; with
a roar it rose into the air and off into the night. ‘I shall see you soon, I expect, Professor McGonagall,’ said
Dumbledore, nodding to her. Professor McGonagall blew her nose
in reply. Dumbledore turned and walked back down the street. On the
18 HARRY POTTER
corner he stopped and took out the silver Put-Outer. He clicked it
once and twelve balls of light sped back to their street lamps so
that Privet Drive glowed suddenly orange and he could make out
a tabby cat slinking around the corner at the other end of the
street. He could just see the bundle of blankets on the step of
number four. ‘Good luck, Harry,’ he murmured. He turned on his heel and
with a swish of his cloak he was gone. A breeze ruffled the neat hedges of Privet Drive, which lay
silent and tidy under the inky sky, the very last place you would
expect astonishing things to happen. Harry Potter rolled over
inside his blankets without waking up. One small hand closed on
the letter beside him and he slept on, not knowing he was special,
not knowing he was famous, not knowing he would be woken in
a few hours’ time by Mrs Dursley’s scream as she opened the front
door to put out the milk bottles, nor that he would spend the next
few weeks being prodded and pinched by his cousin Dudley ... He
couldn’t know that at this very moment, people meeting in secret
all over the country were holding up their glasses and saying in
hushed voices: ‘To Harry Potter – the boy who lived!’"""))
