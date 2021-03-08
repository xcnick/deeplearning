import argparse
import logging
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from data.coco import get_coco
from data import data_transforms as T
from models.yolov3 import YoloV3
from models.head import YoloLoss

logger = logging.getLogger(__name__)


anchors = [
    [[116, 90], [156, 198], [373, 326]],
    [[30, 61], [62, 45], [59, 119]],
    [[10, 13], [16, 30], [33, 23]],
]

image_size = (416, 416)

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    transforms.append(T.Resize(image_size[0]))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def train(args):
    train_dataset = get_coco(root=args.data_dir, image_set="val", transforms=get_transform("True"))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    )
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=T.collater
    )
    t_total = len(train_dataloader) // args.num_train_epochs

    model = YoloV3().to(args.device)

    logger.info("Using SGD optimizer.")
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters()], "weight_decay": 4e-05},
    ]
    optimizer = optim.SGD(optimizer_grouped_parameters, lr=0.001, momentum=0.9, weight_decay=4e-05)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Total optimization steps = %d", t_total)

    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YoloLoss(anchors[i], 80, image_size))

    model.zero_grad()
    for epoch in range(int(args.num_train_epochs)):
        for step, samples in enumerate(train_dataloader):
            start_time = time.time()
            model.train()
            images, labels = tuple(t.to(args.device) for t in samples)

            optimizer.zero_grad()
            outputs = model(images)
            losses_name = ["total_loss", "x", "y", "w", "h", "conf", "cls"]
            losses = []
            for _ in range(len(losses_name)):
                losses.append([])
            for i in range(3):
                _loss_item = yolo_losses[i](outputs[i], labels)
                for j, l in enumerate(_loss_item):
                    losses[j].append(l)
            losses = [sum(l) for l in losses]
            loss = losses[0]
            loss.backward()
            optimizer.step()

            model.zero_grad()
            if step > 0 and step % 10 == 0:
                _loss = loss.item()
                duration = float(time.time() - start_time)
                example_per_second = args.train_batch_size / duration
                lr = optimizer.param_groups[0]['lr']
                logger.info(
                    "epoch [%.3d] iter = %d loss = %.2f example/sec = %.3f lr = %.5f "%
                    (epoch, step, _loss, example_per_second, lr)
                )
                # config["tensorboard_writer"].add_scalar("lr",
                #                                         lr,
                #                                         config["global_step"])
                # config["tensorboard_writer"].add_scalar("example/sec",
                #                                         example_per_second,
                #                                         config["global_step"])
                for i, name in enumerate(losses_name):
                    value = _loss if i == 0 else losses[i]
                    # config["tensorboard_writer"].add_scalar(name,
                    #                                         value,
                    #                                         config["global_step"])


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=False,
        help="Model type selected in the list:",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=2,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="For distributed training: local_rank"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision instead of 32-bit",
    )
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    train(args)


if __name__ == "__main__":
    main()
