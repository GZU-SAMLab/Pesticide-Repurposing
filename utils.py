import argparse

from openke.module.model import ComplEx, RotatE, TransD, TransE, TransH, TransR


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train/Test a knowledge graph embedding model.")

    # Model parameters
    parser.add_argument(
        "--model_name",
        type=str,
        choices=[
            "TransE",
            "TransH",
            "TransR",
            "TransD",
            "ComplEx",
            "RotatE",
            "transe",
            "transh",
            "transr",
            "transd",
            "complex",
            "rotate",
        ],
        default="TransE",
        help="The model name. Choices are: TransE, Trans, TransR, TransD, ComplEx, RotatE. Default is TransE.",
    )
    parser.add_argument("--dim", type=int, default=200, help="The dimension of the embedding matrix. Default is 200.")
    parser.add_argument(
        "--margin",
        type=int,
        default=10,
        help="The margin for loss function. Default is 10. This parameter does not take effect if model_name is ComplEx.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="The range for initializing the matrix. If None, Xavier initialization is used. Otherwise, uniform initialization in the range [-epsilon, epsilon].",
    )

    # Data parameters
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="The path to the dataset. Refer to the Openke dataset documentation for dataset format.",
    )

    # Optimizer parameters
    parser.add_argument(
        "--opt_method",
        type=str,
        choices=["Adam", "Adadelta", "Adagrad", "SGD"],
        default="Adam",
        help="The optimizer method. Choices are: Adam, Adadelta, Adagrad, SGD. Default is Adam.",
    )
    parser.add_argument("--alpha", type=float, default=0.01, help="The learning rate. Default is 0.01.")
    parser.add_argument("--train_times", type=int, default=1000, help="The number of training epochs. Default is 1000.")
    parser.add_argument("--batch_size", type=int, default=4096, help="The batch size. Default is 4096.")
    parser.add_argument("--use_gpu", type=bool, default=True, help="Whether to use GPU for training. Default is True.")

    # Sampler parameters
    parser.add_argument(
        "--sampling_mode",
        type=str,
        choices=["normal", "cross_sampling"],
        default="normal",
        help="Sampling mode. 'normal' for head and tail entity sampling together, 'cross_sampling' for head and tail entity cross-sampling. Default is 'normal'.",
    )
    parser.add_argument(
        "--bern_flag",
        type=int,
        choices=[0, 1],
        default=1,
        help="Negative sample sampling strategy. 1 for Bernoulli sampling, 0 for uniform sampling. Default is 1.",
    )
    parser.add_argument(
        "--filter_flag",
        type=int,
        choices=[0, 1],
        default=1,
        help="Whether to filter the sampled negative entities. Default is 1.",
    )
    parser.add_argument(
        "--neg_ent", type=int, default=1, help="The ratio of negative samples for entities. Default is 1."
    )
    parser.add_argument(
        "--neg_rel", type=int, default=0, help="The ratio of negative samples for relations. Default is 0."
    )
    parser.add_argument("--regul_rate", type=float, default=0, help="The regularization coefficient. Default is 0.")

    parser.add_argument(
        "--threads", type=int, default=8, help="The num of theads that uesd for train and test. Default is 8"
    )

    parser.add_argument("--train", type=bool, default=False, help="Train Model ?")

    parser.add_argument("--test", type=bool, default=True, help="Test Model ?")

    parser.add_argument("--save_path", type=str, default="ckpt", help="the path to save checkpoint")

    parser.add_argument("--load_path", type=str, default=None, help="the path to loading model just for test")
    return parser.parse_args()


def get_model(model_name: str, train_dataloader, dim, margin, epsilon=None):
    match model_name.lower():
        case "transe":
            the_model = TransE(
                ent_tot=train_dataloader.get_ent_tot(),
                rel_tot=train_dataloader.get_rel_tot(),
                dim=dim,
                margin=margin,
                epsilon=epsilon,
            )
        case "transr":
            the_model = TransR(
                ent_tot=train_dataloader.get_ent_tot(),
                rel_tot=train_dataloader.get_rel_tot(),
                dim_e=dim,
                dim_r=dim,
                margin=margin,
            )
        case "transh":
            the_model = TransH(
                ent_tot=train_dataloader.get_ent_tot(),
                rel_tot=train_dataloader.get_rel_tot(),
                dim=dim,
                margin=margin,
            )
        case "transd":
            the_model = TransD(
                ent_tot=train_dataloader.get_ent_tot(),
                rel_tot=train_dataloader.get_rel_tot(),
                dim_e=dim,
                dim_r=dim,
                margin=margin,
                epsilon=epsilon,
            )
        case "complex":
            the_model = ComplEx(
                ent_tot=train_dataloader.get_ent_tot(),
                rel_tot=train_dataloader.get_rel_tot(),
                dim=dim,
            )
        case "rotate":
            the_model = RotatE(
                ent_tot=train_dataloader.get_ent_tot(),
                rel_tot=train_dataloader.get_rel_tot(),
                dim=dim,
                margin=margin,
            )
    return the_model
