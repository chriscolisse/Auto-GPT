import os
import re

from tiktoken import get_encoding
import openai
from dotenv import load_dotenv
from pathlib import Path

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


def gpt_3_5_validator(text: str, initial_text: str, splitter=None, extra_mem: list | None = None, **kwargs) -> str:
    print(text)
    start_bot = False
    end_loop = False
    splitter = "**"
    if kwargs:
        if "end_loop" in kwargs:
            return text

    # implement"" these in classes so that I can get information about the class when passing as objects. Makes it
    # easier to pass in agents in a function and have the function detect what type of bot it is. This can reduce
    # need to pass in many arguments and avoid writing the same code many times.
    print("\n\n\n ..VALIDATING...\n\n\n")
    validate_prompt = "You are a python function that validates output from a compression GPT-3.5 Agent or stitching GPT-3.5 agent by comparing " \
                      "input text to the initial text, and determining whether the meaning of the initial text is entirely conveyed in the input text.\n" \
                      "If the near-entirety of the meaning of the initial text can be recovered from the input text, pass the exact input text as VALIDATED OUTPUT, unadulterated.\n" \
                      "If the Bot input text was empty, return \"\" for VALIDATED OUTPUT" \
                      "If there is an issue with the input text, or the bot is confused or uncertain in any way, use NEW AGENT.\n" \
                      "Do not speak, comment, or provide any other information than the proper response format\n"\
                      "YOUR RESPONSE MUST BE IN ONE OF TWO FORMATS, and must match EXACTLY:\n" \
                      f"{splitter}VALIDATED OUTPUT: <final validated output> {splitter} FINAL ANSWER {splitter}\n" \
                      "**OR**\n" \
                      f"{splitter}COMMAND: NEW_AGENT {splitter} ARGS: <enter the initial text here>{splitter}\n" \
                      "**You must keep your responses equal in length or shorter than the input text**\n" \
                      "**If filenames are present in the input text, you must preserve them in your final answer.**"


    messages = [
        {"role": "system", "content": validate_prompt},
        {"role": "system", "content": "INITIAL TEXT (FROM BOT): \n" + initial_text + "\nBOT TEXT:\n" + str(text)}
    ]

    if extra_mem:
        messages = [*messages, *extra_mem]

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    completion = completion['choices'][0]['message']['content']
    end_loop = "FINAL ANSWER" in completion or "VALIDATED OUTPUT:" in completion
    start_bot = True if (
                (("COMMAND" in completion) and ("ARGS" in completion)) or ("START BOT" in completion)) else False
    print(end_loop, start_bot)
    print(completion)
    if end_loop and start_bot:
        x_mem = [{"role": "assistant", "content": completion}, {"role": "system",
                                                                "content": f"Your output is confusing. You used both the COMMAND format, and the FINAL ANSWER format. Remember, you must respect ONE of the two following formats: {splitter}VALIDATED OUTPUT: <final validated output> {splitter} FINAL ANSWER {splitter}\n **OR**\n{splitter}COMMAND: NEW_COMPRESSION_AGENT {splitter} ARGS: <enter the initial text here>{splitter}\n"}]
        return gpt_3_5_validator(completion, initial_text=initial_text, extra_mem=x_mem)

    if end_loop:
        print(completion.split(splitter))
        try:
            return completion.split(splitter)[0].split("VALIDATED OUTPUT:")[1]
        except:
            try:
                return completion.split(splitter)[1].split("VALIDATED OUTPUT:")[1]
            except Exception as e:
                raise e


    x_mem = None

    if start_bot:
        kwargs["args"] = None
        agent_command = None
        agent_kwargs = {}
        text_val = completion.split(splitter)
        print(text_val)
        for s in text_val:
            if "COMMAND:" in s:
                agent_command = s.split("COMMAND:")[1].strip()
                if "NEW_AGENT" not in agent_command:
                    x_mem = [{"role": "assistant", "content": completion}, {"role": "system",
                                                                            "content": "Your output contains the 'COMMAND' and 'ARGS' commands, yet no valid command has been found by system. Did you want to start a bot? If so, respond with 'START BOT' and nothing else. Did you want to provide a final answer? If so respond only with \"FINAL ANSWER\", anf nothing else."}]
            elif "ARGS:" in s:
                kwargs["args"] = s.split("ARGS: ")[1]
        if not (kwargs["args"] and agent_command):
            insight = gpt_3_5_validator(completion, initial_text=initial_text, extra_mem=x_mem, args=agent_kwargs)
            return gpt_3_5_validator(insight, initial_text=initial_text)
        else:
            compress = gpt_3_5_compress(initial_text)
            return gpt_3_5_validator(compress, initial_text=initial_text)
    else:
        x_mem = [{"role": "assistant", "content": completion}, {"role": "system",
                                                                "content": "Your output could not be parsed. Please make sure you have responded in the proper format."}]
        return gpt_3_5_validator(completion, initial_text=initial_text, extra_mem=x_mem)


def split_by_2(text):
    output = text.split(" ")
    if abs((len(output) / 2) - (round(len(output) / 2))) != 0:
        return output[:int(len(output) - 1 / 2)], output[int(len(output) - 1 / 2):]
    return output[:int(len(output) / 2)], output[int(len(output) / 2):]


def check_tokens(text):
    enc = get_encoding("cl100k_base")
    return len(enc.encode(text))


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
    if compression:
        # print('\n\n>>>>>\n', text, '\n')
        text = gpt_3_5_validator(gpt_3_5_compress(text), initial_text=text)
    return text


def get_cwd_file_contents(path=None, delimiter="<>", workspace=True):
    output = ''
    if not path or path == "auto_gpt_workspace":
        path = os.getcwd() + "\\auto_gpt_workspace"
    if path and workspace:
        path = os.getcwd() + "\\auto_gpt_workspace\\" + path

    files = os.listdir(path)
    if files not in path:
        return "No files files or folders present."
    for file in files:
        file = path + "\\" + file
        if os.path.isfile(file):
            with open(file, 'r', encoding="utf-8") as f:
                name = f.name.split('\\')[-1]
                output += f"\n{delimiter}\nFile Name:\n{name}\\nContents:\n{f.read()}\n{delimiter}\n"

    return token_check_and_compress(output, files)


if __name__ == "__main__":
    delimiter = "<>"
    newt = get_cwd_file_contents(path=r"C:\Users\PC\Desktop\Writing\New folder", delimiter=delimiter)
    print(check_tokens(newt))


    def test():
        names = []
        split = newt.split("File Name:\n")
        for s in split:
            if s.split("\nContents:")[0].strip() != delimiter:
                names.append(s.split("Contents:")[0].strip('\\n').strip('\n'))
        print(names)
        print(check_tokens(get_cwd_file_contents(delimiter=delimiter)))


    test()
    print(newt)
