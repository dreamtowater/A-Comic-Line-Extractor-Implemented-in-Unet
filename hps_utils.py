import os
import argparse
import json


get_config_path = lambda model_dir: os.path.join(model_dir, "config.json")
get_log_dir = lambda model_dir: os.path.join(model_dir, "logs")
get_ckpt_dir = lambda model_dir, epoch: os.path.join(model_dir, str(epoch))



def init_model_dir(model_dir, config_path):
    # create model_dir
    assert not os.path.isdir(model_dir), "不可重名,继续训练也请创建新文件夹"
    os.makedirs(model_dir)

    # copy config file
    with open(config_path, "r") as f: data = f.read()
    config_path = get_config_path(model_dir)
    with open(config_path, "w") as f: f.write(data)

    # create log dir
    log_dir = get_log_dir(model_dir)
    os.makedirs(log_dir)



def get_hparams_from_parser():
    """
    从命令行获取超参数。
    train: True表示调用该函数用于训练,须给定config_path,会创建保存模型的文件夹;\
           Fasle表示调用该函数用于推理,无需给定config_path,不创建任何文件(夹)。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', type=int, default=1,
                        help='Train or eval')
    parser.add_argument('-c', '--config_path', type=str, default='./config.json',
                        help='JSON file for configuration')
    parser.add_argument('-m', '--model_name', type=str, required=True,
                        help='Model name')
    parser.add_argument('-d', '--Models_dir', type=str, default="./Models",
                        help='Directory for saving model')
    args = parser.parse_args()

    model_dir = os.path.join(args.Models_dir, args.model_name)

    if args.train == 1: init_model_dir(model_dir, args.config_path)
    # else: args.config_path = get_config_path(model_dir)

    with open(args.config_path, "r") as f: hps = HParams(**json.loads(f.read()))
  
    hps.model_dir = model_dir
    hps.log_dir = get_log_dir(model_dir)
    hps.model_name = args.model_name
    return hps



def get_hparams_from_dir(model_dir):
    config_path = get_config_path(model_dir)
    with open(config_path, "r") as f: hps = HParams(**json.loads(f.read()))
    return hps



def get_hparams_from_file(config_path):
    with open(config_path, "r") as f: hps = HParams(**json.loads(f.read()))
    return hps



class HParams():
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      if type(v) == dict:
        v = HParams(**v)
      self[k] = v
    
  def keys(self):
    return self.__dict__.keys()

  def items(self):
    return self.__dict__.items()

  def values(self):
    return self.__dict__.values()

  def __len__(self):
    return len(self.__dict__)

  def __getitem__(self, key):
    return getattr(self, key)

  def __setitem__(self, key, value):
    return setattr(self, key, value)

  def __contains__(self, key):
    return key in self.__dict__

  def __repr__(self):
    return self.__dict__.__repr__()

  def __add__(self, rhs):
    for k, v in self.items():
      if k not in rhs.keys():
        rhs[k] = v
      elif type(v) != type(rhs[k]):
        raise NotImplementedError(f"Don't merge {type(v)} and {type(rhs[k])}!!")
      elif type(v) == type(self):
        rhs[k] = v + rhs[k]
      else:
        rhs[k] = v
    return rhs