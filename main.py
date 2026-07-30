'''Base main script for easy working'''

import os
import argparse
import logging
from util_modules import utils
import pipelines


def Learning_RL(args, logger):
    logger.info(f"********************* Representation learning *********************")
    if args["net"] != "ocir":
        raise ValueError(f"Unsupported net for this reproduction workspace: {args['net']}")
    learning = pipelines.RlPipeline(args, logger)
    model = learning()
    return model


def Learning_RUL(args, logger, encoder = None, shared_layer = None):
    logger.info(f"********************* RUL estimation Pipeline *********************")
    learning = pipelines.RulPipeline(args, logger, encoder, shared_layer)
    model = learning()
    return model


def main(args, logger):
    if args["task"] == "RL":
        Learning_RL(args, logger)
    elif args["task"] == "rul":
        rl_model = Learning_RL(args, logger)
        Learning_RUL(args, logger, rl_model, None)
    else:
        raise ValueError(f"Unsupported task for this reproduction workspace: {args['task']}")
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default='./Logss/logs_', help="path to save all the products from each trainging")
    parser.add_argument("--id_", type=int, default=0, help="Run id")
    parser.add_argument("--data_path", type=str, default='./data/processed/main_protocol_v1', help="path to prepared main_protocol_v1 data")
    parser.add_argument("--description", type=str, default='test_run', help="optional")
    
    # Task & data args
    parser.add_argument("--task", type=str, default="RL", choices = ["RL", "rul"])
    parser.add_argument("--dataset", type=str, default="FD001", choices = {"FD001", "FD002", "FD003", "FD004"})
    parser.add_argument("--c_type", type=str, default="discrete", help = "latent operating-condition code type")
    parser.add_argument("--net", type=str, default="ocir", choices = ["ocir"])
    parser.add_argument("--valid_split", type=float, default=0., help= "learning rate")
    
    # Training args
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--gpu_dev", type=str, default="6")
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--patience", type=int, default=10)
    
    parser.add_argument("--rul_epochs", type=int, default=50)
    parser.add_argument("--rul_validate_every_epochs", type=int, default=1, help="RUL validation frequency in epochs")
    # optimizer
    parser.add_argument("--lr_", type=float, default=5e-4, help= "learning rate")
    parser.add_argument("--rul_lr", type=float, default=5e-4, help= "learning rate for rul")
    parser.add_argument("--scheduler", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--kl_annealing", type=float, default=0.15)

    ## Scheduler params
    parser.add_argument("--warm_up", type=float, default=0.2, help="portion of warm up given number of epoches, e.g., 20 percent by defualt")
    parser.add_argument("--start_lr", type=float, default=1e-5, help="starting learning rate")
    parser.add_argument("--ref_lr", type=float, default=1.5e-4, help= "peak learning rate")
    parser.add_argument("--final_lr", type=float, default=1e-5, help = "final learning rate")
    parser.add_argument("--start_wd", type=float, default=0.01, help = "starting weight decay, setting it to 0. means no decay and decay scheduler")
    parser.add_argument("--final_wd", type=float, default=0.0001, help = "fianl weight decay")
    # TODO consider specifying separate lr 
    # Save path
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--plots_save_path', type=str, default='plots')
    parser.add_argument('--his_save_path', type=str, default='hist')
    
    # Model args
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--H", type=int, default=2)
    parser.add_argument("--hyper_lookback", type=int, default=2)
    parser.add_argument("--time_embedding", type=bool, default=False)
    parser.add_argument("--c_kl", type=int, default=0, help = "kl div for f_C")
    
    parser.add_argument("--dx", type=int, default=14)
    parser.add_argument("--dz", type=int, default=20)
    parser.add_argument("--dc", type=int, default=6, help = "if discrete, it is the expected number of ocs")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=2)
    
    ## Latent encoder
    parser.add_argument("--encoder_E", type=str, default="transformer", choices= ["transformer", "TCN"])
    parser.add_argument("--z_projection", type=str, default="spc", 
                        choices = ["aggregation", "spc", "rnn", "seq", "aggregation_all"])
    ## Discriminator
    parser.add_argument("--D_projection", type=str, default="spc", 
                    choices = ["aggregation", "spc",  "rnn", "None"])
    ## Q and coder encoder
    parser.add_argument("--c_posterior_param", type=str, default="soft", choices = ["soft", "hard"])
    
    # Downstream task hyper params
    parser.add_argument("--extractor", type=str, default="ocir", choices= ["ocir"])
    
    # Baseline hyperparams
    parser.add_argument("--conditional", type=int, default=1)
    
    
    config = parser.parse_args()
    
    log_path = config.log_path + config.dataset + "_" + f"{config.id_}" + "_" + f"{config.description}" 
    
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    os.environ["log_loc"] = f"{log_path}"
    root_dir = os.getcwd() 
    logging.basicConfig(filename=os.path.join(root_dir, f'{log_path}/log_all.txt'), level=logging.INFO,
                        format = '%(asctime)s - %(name)s - %(message)s')
    logger = logging.getLogger('In main')
    
    config.model_save_path = os.path.join(log_path,"checkpoints") 
    utils.mkdir(config.model_save_path)
    config.plots_save_path = os.path.join(log_path,"plots") 
    utils.mkdir(config.plots_save_path)
    config.his_save_path = os.path.join(log_path,"hist") 
    utils.mkdir(config.his_save_path)
    
    config.time_embedding = False
    # Check args
    if (config.dataset == "FD001") or (config.dataset == "FD003"):
        config.c_type = "continuous"
    else:
        config.c_type = "discrete"
        # config.dc = 6
    
    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        logger.info('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    
    main(args, logger)
