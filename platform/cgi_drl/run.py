"""CGI DRL Framework (Video Game)"""
import argparse
import importlib
import json
import multiprocessing
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-t", "--template", help="What's the config template you want to use?", default="cgi_drl.default_config")
    parser.add_argument("-tc", "--template_class", help="What's the config template class you want to use?", default="Default")
    parser.add_argument("-f", "--config_file", help="What's the config file you want to load?", default="elo_rcc_jsai.yaml")
    parser.add_argument("-k", "--config_key", help="What's the config key you want to load?", default="default")
    parser.add_argument("-i", "--run_id", help="What's the id of this run?", default="")
    parser.add_argument("-j", "--json_parameter_string", help="What's the extra parameters with json format?", type=str, default="{}")
    return parser.parse_args()

def run(template, template_class, config_file, config_key, run_id, json_parameter_string):
    def load(_template, _template_class, _filename, _key, _info=""):
        print("load {} from {} : {} with template {}".format(_info, _filename, _key, _template))
        with open(_filename,encoding="utf-8") as f:
            config = yaml.safe_load(f)[_key]
        return getattr(importlib.import_module(_template), _template_class)(config)
        
    config = load(template, template_class, config_file, config_key)
    problem_config = load(*config["config"], _info="problem")
    problem_config["load_function"] = load
    extra_parameters = json.loads(json_parameter_string)
    problem_config["run_id"] = run_id
    problem_config["extra_parameters"] = extra_parameters

    launch = importlib.import_module(config["solver"]).launch
    launch(problem_config)

def main(args):
    process = multiprocessing.Process(target=run, args=(args.template, args.template_class, args.config_file, args.config_key, args.run_id, args.json_parameter_string))
    process.start()
    process.join()

if __name__ == "__main__":
    args = parse_args()
    main(args)
