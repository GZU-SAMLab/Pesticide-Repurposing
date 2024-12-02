import os
import warnings

from openke.config import Tester, Trainer
from openke.data import TestDataLoader, TrainDataLoader
from openke.module import model
from openke.module.loss import SoftplusLoss
from openke.module.model import RotatE
from openke.module.strategy import NegativeSampling

warnings.filterwarnings("ignore")
# data parameters
data_path = "data/"
batch_size = 4096
threads = 64
sampling_mode = "normal"
bern_flag = 1
filter_flag = 1
neg_ent = 1
neg_rel = 0

# model parameters
model_name = "RotatE"
dim = 20
margin = 10

# optimizer parameters
train_times = 1000
alpha = 0.01
use_gpu = True
opt_method = "Adam"

# sampler parameters
regul_rate = 0


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
rotate = RotatE(
    ent_tot=train_dataloader.get_ent_tot(),
    rel_tot=train_dataloader.get_rel_tot(),
    dim=dim,
    margin=margin,
)

# define the loss function
model = NegativeSampling(
    model=rotate,
    loss=SoftplusLoss(),
    batch_size=train_dataloader.get_batch_size(),
    regul_rate=regul_rate,
)

# train the model
trainer = Trainer(
    model=model,
    data_loader=train_dataloader,
    train_times=train_times,
    alpha=alpha,
    use_gpu=use_gpu,
    opt_method=opt_method,
)
# trainer.run()

# tensorboardX save
model_save_path = f"ckpt/{model_name}.ckpt"
# os.makedirs(model_save_path, exist_ok=True)
# rotate.save_checkpoint(model_save_path)

# test the model
rotate.load_checkpoint(model_save_path)
tester = Tester(model=rotate, data_loader=test_dataloader, use_gpu=True)
mrr, mr, hit10, hit3, hit1, tail_mrr, tail_mr, tail_hit10, tail_hit3, tail_hit1 = tester.run_link_prediction(
    type_constrain=True
)
