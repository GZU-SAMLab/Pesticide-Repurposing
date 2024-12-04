import os
import warnings

from openke.config import Tester, Trainer
from openke.data import TestDataLoader, TrainDataLoader
from openke.module import model
from openke.module.loss import SoftplusLoss
from openke.module.model import ComplEx, RotatE, TransD, TransE, TransH, TransR
from openke.module.strategy import NegativeSampling
from utils import get_model, parse_arguments

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    args = parse_arguments()

    # data parameters
    data_path = args.data_path
    batch_size = args.batch_size
    threads = args.threads
    sampling_mode = args.sampling_mode
    bern_flag = args.bern_flag
    filter_flag = args.filter_flag
    neg_ent = args.neg_ent
    neg_rel = args.neg_rel
    # model parameters
    model_name: str = args.model_name
    dim = args.dim
    margin = args.margin
    epsilon = args.epsilon

    # optimizer parameters
    train_times = args.train_times
    alpha = args.alpha
    use_gpu = args.use_gpu
    opt_method = args.opt_method

    # sampler parameters
    regul_rate = args.regul_rate

    # Train/Test
    is_train = args.train
    is_test = args.test
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path=data_path,
        batch_size=batch_size,
        threads=threads,
        sampling_mode=sampling_mode,
        bern_flag=bern_flag,
        filter_flag=filter_flag,
        neg_ent=neg_ent,
        neg_rel=neg_rel,
    )

    # dataloader for test
    test_dataloader = TestDataLoader(data_path, "link")

    # define the model
    the_model = get_model(
        model_name.lower(), train_dataloader=train_dataloader, dim=dim, margin=margin, epsilon=epsilon
    )

    if is_train:
    # define sampler and loss function
        sampler = NegativeSampling(
            model=the_model,
            loss=SoftplusLoss(),
            batch_size=train_dataloader.get_batch_size(),
            regul_rate=regul_rate,
        )

    # train the model
        trainer = Trainer(
            model=sampler,
            data_loader=train_dataloader,
            train_times=train_times,
            alpha=alpha,
            use_gpu=use_gpu,
            opt_method=opt_method,
        )
        trainer.run()

    # model save
    if args.save_path:
        model_save_path = f"{args.save_path}/{model_name}.ckpt"
        os.makedirs(model_save_path, exist_ok=True)
        the_model.save_checkpoint(model_save_path)

    # test the model
    if not is_train and is_test and args.load_path:
        # test ckpt
        the_model.load_checkpoint(model_save_path)
    tester = Tester(model=the_model, data_loader=test_dataloader, use_gpu=use_gpu)
    mrr, mr, hit10, hit3, hit1, tail_mrr, tail_mr, tail_hit10, tail_hit3, tail_hit1 = tester.run_link_prediction(
        type_constrain=True
    )
