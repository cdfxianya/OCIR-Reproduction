import torch
import numpy as np
import src

from ocir_reproduction.protocols import MAIN_PROTOCOL_V1, validate_main_protocol_v1
import util_modules as ut

class DotDict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value

class Solver(object):
    DEFAULTS = {}
    def __init__(self, config, logger):
        super(Solver, self).__init__()
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.logger = logger
        
        seed = self.seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = torch.device(f'cuda:{self.gpu_dev}' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device("cpu")
        self.logger.info(f"GPU (device: {self.device}) used" if torch.cuda.is_available() else 'cpu used')
        
        self.evaluation = ut.Evaluation(self.plots_save_path,
                                        self.his_save_path,
                                        self.logger)
        
    def build_dataset(self):
        protocol_errors = validate_main_protocol_v1(MAIN_PROTOCOL_V1)
        if protocol_errors:
            raise ValueError("Invalid main_protocol_v1: " + "; ".join(protocol_errors))

        try:
            from ocir_reproduction.data.cmapss_protocol import load_main_protocol_cmapss
        except ModuleNotFoundError as exc:
            raise NotImplementedError(
                "main_protocol_v1 data layer is not implemented yet. "
                "Next step: create ocir_reproduction.data.cmapss_protocol.load_main_protocol_cmapss."
            ) from exc

        self.training_data, self.val_data, self.testing_data, self.full_test_set = load_main_protocol_cmapss(
            dataset=self.dataset,
            data_path=self.data_path,
            window_length=self.window,
            rul_cap=MAIN_PROTOCOL_V1["labels"]["rul_cap"],
            batch_size=self.batch,
            split_seed=MAIN_PROTOCOL_V1["splits"]["split_seed"],
            model_seed=self.seed,
            logger=self.logger,
        )
        self.ipe = len(self.training_data)

    def set_training_sampler_epoch(self, epoch):
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")

        sampler = getattr(getattr(self, "training_data", None), "sampler", None)
        if sampler is None:
            self.logger.warning("training_data has no sampler; epoch-aware train sampling is not active")
            return
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
            return

        self.logger.warning(
            "training_data sampler %s does not support set_epoch(); epoch-aware train sampling is not active",
            type(sampler).__name__,
        )

    def build_model(self):
        def print_model(m, model_name):
            self.logger.info(f"Model: {model_name}")
            total_param = 0
            for name, param in m.named_parameters():
                num_params = param.numel()
                total_param += num_params
                self.logger.info(f"{name}: {num_params} parameters")
            
            self.logger.info(f"Total parameters in {model_name}: {total_param}")
            self.logger.info("")
            return total_param
        
        ocir = src.OCIR(dx=self.dx, dz=self.dz, dc=self.dc, window=self.window, 
                        d_model=self.d_model, num_heads=self.num_heads, z_projection=self.z_projection, 
                        D_projection=self.D_projection, time_emb=self.time_embedding, c_type=self.c_type, 
                        c_posterior_param=self.c_posterior_param, encoder_E=self.encoder_E, c_kl= self.c_kl, device=self.device)
        
        print_model(ocir, "OCIR")
        self.ocir, self.required_training = ut.load_model(ocir, self.model_save_path, "OCIR")
        self.ocir.to(self.device)
        
    def get_optimizers(self):
        raise NotImplementedError("")
    
    def early_stop(self):
        raise NotImplementedError("")
    
