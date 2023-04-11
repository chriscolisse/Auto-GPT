import yaml
import data
import json


class AIConfig:
    def __init__(self, ai_name="", ai_role="", ai_goals=[], tools={}):
        self.ai_name = ai_name
        self.ai_role = ai_role
        self.ai_goals = ai_goals
        self.tools = tools

    # Soon this will go in a folder where it remembers more stuff about the run(s)
    SAVE_FILE = "../ai_settings.yaml"

    @classmethod
    def load(cls, config_file=SAVE_FILE):
        # Load variables from yaml file if it exists
        try:
            with open(config_file) as file:
                config_params = yaml.load(file, Loader=yaml.FullLoader)
        except FileNotFoundError:
            config_params = {}
        try:
            with open("tool_list.json", "r") as file:
                tool_params = json.loads(file.read())
        except:
            tool_params = {}

        ai_name = config_params.get("ai_name", "")
        ai_role = config_params.get("ai_role", "")
        ai_goals = config_params.get("ai_goals", [])
        tools = tool_params

        return cls(ai_name, ai_role, ai_goals, tools)

    def save(self, config_file=SAVE_FILE):
        config = {"ai_name": self.ai_name, "ai_role": self.ai_role, "ai_goals": self.ai_goals}
        with open(config_file, "w") as file:
            yaml.dump(config, file)

