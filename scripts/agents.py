import os

from llm_utils import create_chat_completion
from config import Config
from ai_config import AIConfig
import chat
import json
import token_counter
from pathlib import Path

cfg = Config()

path = Path(os.getcwd())


class BaseAgent:
    def __init__(self, model, messages, temperature, max_tokens):
        self.messages = messages
        self.temperature = temperature
        self._max_tokens = max_tokens
        self.model = model

    @property
    def max_tokens(self):
        return self._max_tokens

    @max_tokens.setter
    def max_tokens(self, val):
        msg_tokens = token_counter.count_message_tokens(self.messages)
        self._max_tokens = val - msg_tokens

    def execute(self):
        print('\n\n---\n\n', self.max_tokens, '\n\n---\n\n')
        output = create_chat_completion(
            model=self.model,
            messages=self.messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        return output

    @staticmethod
    def construct_tool_prompt(tools, prompt_start=None):
        if prompt_start is None:
            prompt_start = """These are the tools you have access to (AVAILABLE COMMANDS):\n"""

        # example tool command: format  1. Google Search: "google", args: "input": "<search>"
        prompt = ""

        if tools:
            for n, tool in enumerate(tools):
                tool_name = tools[tool].get('name')
                description = tools[tool].get('description')
                get_args = tools[tool].get('args')
                args = ""
                if isinstance(get_args, dict):
                    for arg, val in get_args.items():
                        args += f"\"{arg}\": \"{val}\""

                prompt += f"{n}. {tool}: \"{tool_name}\", description: {description},  args: {args}\n"
            prompt = prompt_start + prompt + '\n'

        return prompt

    def construct_full_prompt(self, tools=None, base_prompt=False):
        prompt_start = """Your decisions must always be made independently without seeking user assistance. Use your reasoning capabilities to break tasks down into smaller parts and achieve your goals."""

        # Construct full prompt
        if not base_prompt:
            full_prompt = f"You are {self.ai_name}, {self.ai_role}\n{prompt_start}\n\nGOALS:\n\n"
            for i, goal in enumerate(self.ai_goals):
                full_prompt += f"{i + 1}. {goal}\n"
        else:
            return base_prompt

        if tools is None:
            tools = self.tools
        else:
            tools = tools

        file_prompt = self.load_prompt()
        file_prompt = file_prompt.split("AVAILABLE COMMANDS:")
        tool_prompt = self.construct_tool_prompt(tools)
        file_prompt.insert(1, tool_prompt)
        file_prompt = " ".join(file_prompt)

        full_prompt += file_prompt
        return full_prompt

    # @property
    def load_prompt(self):
        try:
            # get directory of this file:
            file_dir = Path(__file__).parent
            prompt_file_path = file_dir / "data" / "prompt.txt"

            # Load the prompt from data/prompt.txt
            with open(prompt_file_path, "r") as prompt_file:
                prompt = prompt_file.read()

            return prompt
        except FileNotFoundError:
            print("Error: Prompt file not found", flush=True)
            return ""


class ValidatorAgent(BaseAgent):
    def __init__(self, temperature=0, max_tokens=cfg.fast_token_limit, model=cfg.fast_llm_model):
        self.prompt = None  # need to change so prompt here is validator prompt
        self.model = model
        self.terminate = False
        self.messages = []
        self.ai_config = AIConfig()
        self.temperature = temperature
        tokens = token_counter.count_message_tokens(self.messages, model=self.model)

        self.max_tokens -= int(tokens + self.max_tokens / 2)
        self.load_base_prompt()
        self.max_tokens = max_tokens

        super().__init__(model=self.model, messages=self.messages, temperature=self.temperature,
                         max_tokens=self.max_tokens)


    def load_base_prompt(self, prompt_file="data/validator_prompt.txt"):
        import os
        try:
            with open(prompt_file, "r") as file:
                contents = file.read()
        except FileNotFoundError:
            raise FileNotFoundError("No file found for Validator prompt.")

        with open("available_agent_tools.json", "r") as file:
            available_tools = json.loads(file.read())
        prompt_start = "VALID COMMANDS:"
        full_prompt = ""
        file_prompt = contents.split(prompt_start)
        tool_prompt = self.ai_config.construct_tool_prompt(available_tools, prompt_start=prompt_start)
        file_prompt.insert(1, tool_prompt)
        file_prompt.insert(1, "VALID COMMANDS:")
        file_prompt = " ".join(file_prompt)
        full_prompt += file_prompt
        self.prompt = self.ai_config.construct_full_prompt(tools={}, base_prompt=full_prompt)
        self.messages.append(chat.create_chat_message(role="system", content=full_prompt))

    def validate(self):
        # print("\n\n-------------\nVALIDATING OUTPUT\n")
        output = self.execute()
        # print("\nVALIDED OUTPUT: \n")
        # print(output, "\n\n---------------\n\n")
        return output


# all the prompt info should be handled in these agent classes, and not hard coded into chat.py, AIconfig.py etc..
# those files should instead become parsers, and message management should happen directly in the agent classes.
class ManagerAgent(BaseAgent):
    def __init__(self, messages, temperature, max_tokens=cfg.fast_token_limit, model=cfg.fast_llm_model,
                 main_agent=False):
        self.messages = messages
        self.temperature = 0.5  # temperature
        self.validate_output = False
        self.model = model
        self.ai_config = AIConfig()
        self.main_agent = main_agent

        try:
            main_path = path
            if "main.py" not in os.listdir(path):
                if "scripts" in os.listdir(path):
                    main_path = f"{main_path}/scripts/"

            with open(f"{main_path}/available_agent_tools.json", "r") as f:
                self.tools = json.loads(f.read())
        except Exception as e:
            raise (Exception(e))
            # self.tools = {}

        with open(f"{main_path}/data/prompt.txt", "r") as f:
            raw_base_prompt = f.read()

        self.base_prompt = self.load_base_prompt(base_prompt=raw_base_prompt)
        self.messages.append(chat.create_chat_message(role="system", content=self.base_prompt))
        self.max_tokens = cfg.fast_token_limit
        super().__init__(self.model, self.messages, self.temperature, self.max_tokens)

    def load_base_prompt(self, base_prompt: bool | str = False, prompt_config="../ai_settings.yaml"):
        prompt_start = """Your decisions must always be made independently without seeking user assistance. Use your reasoning capabilities to break tasks down into smaller parts and achieve your goals."""
        full_prompt = ""
        if self.main_agent:
            import yaml

            try:
                with open(prompt_config) as file:
                    config_params = yaml.load(file, Loader=yaml.FullLoader)
            except FileNotFoundError:
                ValueError(
                    "NO VALID AGENT CONFIG HAS BEEN FOUND FOR MANAGER AGENT (make sure there is an valid ai_settings.yaml file)")

            ai_name = config_params["ai_name"]
            ai_goals = config_params["ai_goals"]
            ai_role = config_params["ai_role"]

            full_prompt = f"You are {ai_name}, {ai_role}\n{prompt_start}\n\nGOALS:\n\n"
            for i, goal in enumerate(ai_goals):
                full_prompt += f"{i + 1}. {goal}\n"

            full_prompt = f"You are {ai_name}, {ai_role}\n{prompt_start}\n\nGOALS:\n\n"
            for i, goal in enumerate(ai_goals):
                full_prompt += f"{i + 1}. {goal}\n"

        file_prompt = base_prompt
        file_prompt = file_prompt.split("AVAILABLE COMMANDS:")
        tool_prompt = self.construct_tool_prompt(self.tools)
        file_prompt.insert(1, tool_prompt)
        file_prompt = " ".join(file_prompt)

        full_prompt += file_prompt
        return full_prompt

    def start(self):
        output = self.execute()
        if self.validate_output:
            validator_agent = ValidatorAgent()
            validator_agent.messages.append(chat.create_chat_message(role="user", content=output))
            output = validator_agent.validate()
            try:
                json.loads(output)
            except:
                pass

        return output
