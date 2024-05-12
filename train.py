import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler, ModelEma

from data_tools import ImageSet
from modules import Unet
from generate_sketch import generate_one_sketch
from hps_utils import get_hparams_from_parser
from log_utils import create_logger


def main():

    args = get_hparams_from_parser()
    logger = create_logger("log", os.path.join(args.log_dir, "log.txt"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("use", device)

    random.seed(args.seed.random_seed)
    np.random.seed(args.seed.np_random_seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed.os_environ_PYTHONHASHSEED)
    torch.manual_seed(args.seed.torch_manual_seed)
    torch.cuda.manual_seed(args.seed.torch_cuda_manual_seed)
    torch.cuda.manual_seed_all(args.seed.torch_cuda_manual_seed_all)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.use_deterministic_algorithms(False)

    dataset_train = ImageSet(args.train_data_path, args.img_Size, args.num_classes)
    data_loader_train = DataLoader(dataset_train, args.batch_size, True,
                                    num_workers=args.num_workers,
                                    pin_memory=args.pin_mem,
                                    drop_last=True)
    logger.info(f"数据集总数为:{len(dataset_train)}, 总批次数为:{len(data_loader_train)}")
    
    model = Unet(args.image_channels, args.start_channels, args.num_classes, args.n_steps).to(device)
    print("模型参数量为:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = create_optimizer(args, model)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    criterion = nn.CrossEntropyLoss()
    loss_scaler = NativeScaler()

    print("开始训练...")
    for epoch in range(args.epochs):

        total_loss = 0.        
        for batch_idx, (images, tgts) in enumerate(data_loader_train):
            images = images.to(device, non_blocking=True)
            tgts = tgts.flatten().to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                preds = model(images).permute(0, 2, 3, 1).flatten(0, 2)
                loss = criterion(preds, tgts)

            optimizer.zero_grad()

            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = getattr(optimizer, 'is_second_order', False)
            loss_scaler(loss, optimizer, clip_grad=None,
                        parameters=model.parameters(), create_graph=is_second_order)

            total_loss += loss.item()
            if batch_idx % 50 == 49:
                print(f"epoch:{epoch} ===> {batch_idx/len(data_loader_train)*100:.2f}%")

        lr_scheduler.step(epoch)

        if epoch % args.save_interval == args.save_interval-1:
            torch.save(model.state_dict(), os.path.join(args.model_dir, f"epoch{epoch}.pth"))
        
        if epoch % args.eval_interval == args.eval_interval-1:
            generate_one_sketch(model, args.eval_image, True, args.img_Size, f"result/epoch{epoch}.png")

        logger.info(f"epoch:{epoch:3d} finished, total_loss = {total_loss}")


if __name__ == "__main__":
    main()