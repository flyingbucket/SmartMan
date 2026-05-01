import argparse
import yaml
from .pipeline.trainer import train
from .pipeline.evaluator import evaluate
from .utils import load_data


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, help="Path to config file")
    p.add_argument("--mode", choices=["train", "eval", "all"], default="all")
    args = p.parse_args()
    return args


def main():
    args = get_args()
    # load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    model_id = config["model_id"]
    run_name = config["run_name"]
    data_dirs = config["data_dirs"]
    recipe = config["recipe"]
    eval_conf = config.get("eval_conf")

    # load data
    data_dict = load_data(data_dirs)

    if args.mode == "train":
        train(model_id, run_name, recipe, data_dict)
    elif args.mode == "eval":
        assert eval_conf is not None
        evaluate(model_id, eval_conf, data_dict, data_dirs)
    elif args.mode == "all":
        output_dir = train(model_id, run_name, recipe, data_dict)
        assert eval_conf is not None
        eval_conf["lora_dir"].append(output_dir)
        evaluate(model_id, eval_conf, data_dict, data_dirs)


if __name__ == "__main__":
    main()
