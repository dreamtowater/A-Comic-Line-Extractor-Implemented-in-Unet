# 简介

## 训练

在当前工程目录下使用 `
!python train.py -c config_flie -d your_model_dir -m your_model_name
` 命令进行训练。

 `config_file` 是训练时的超参数配置文件（必须为 json 格式），详细配置项如项目中的 `config.json` 文件中所示。

`your_model_dir` 是你存放模型的文件夹。

`your_model_name` 是你给训练的模型起的名字，训练时会在 `your_model_dir` 文件下建立一个名字为 `your_model_name` 的子文件夹，日志、模型参数等文件会存放在该子文件夹下。

## 生成

在当前工程目录下使用 `!python generate_sketch.py -m pretrained_model_path -i the_image_need_to_be_converted -c config_file` 生成线稿，线稿会直接保存在当前工程目录下。

`pretrained_model_path` 是训练好的模型的 state_dict 的保存路径。

`the_image_need_to_be_converted` 是需要生成线稿的图片的路径。

`config_file` 同训练命令中的 `config_file` 。

# 效果

<img src="./resource/comparison.png" style="zoom:100%;" />
