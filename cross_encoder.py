# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import logging, sys, shutil, time, argparse, os, importlib.util
import inspect
from fvcore.nn import FlopCountAnalysis, flop_count_table
import config
from Datasets.openEDS2020GazeDataModule import openEDS2020GazeDataModule
#from Datasets.openEDS2020BinocularGazeDataModule import openEDS2020BinocularDataModule
from Datasets.TSSinglePersonDataModule import TSSinglePersonDataModule
from Datasets.VRGazeDataModuleUnsupervised import TSDataModuleUnsupervised
from Datasets.NVGazeMonocularDataModuleIntegratedCalib import NVGazeMonocularDataModuleIntegratedCalib
from Datasets.NVGazeMonocularSinglePersonDataModule import NVGazeMonocularSinglePersonDataModule
from models.MonocularNVGazeModelIntegratedCalib import MonocularNVGazeModelIntegratedCalib
from models.Binocular3DGazeModel import Binocular3DGazeModel
from models.TSPersonAgnosticModel import TSPersonAgnosticModel
from pytorch_lightning.loggers import NeptuneLogger
from models.CrossEncoder import CrossEncoder
from pytorch_lightning import seed_everything

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
import pytorch_lightning as pl
from datetime import datetime
import torch
import numpy as np



def isdebugging():
    for frame in inspect.stack():
        if frame[1].endswith("pydevd.py"):
            return True
    return False

from torch.utils.mobile_optimizer import optimize_for_mobile

def prepare_model_for_android(model, input_tensor, out_path):
    # https://pytorch.org/mobile/android/

    model = model.to_torchscript(method="trace",  example_inputs=(input_tensor, input_tensor))

    # prepare model for Andeoid
    out_path = os.path.join(out_path, 'androd_model_torch_mobile.ptl')
    print('out_path: {}'.format(out_path))
    # input_tensor = torch.randn(1, 1, int(args.image_width), int(args.image_height))
    traced_script_module = torch.jit.trace(model, input_tensor)
    traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    traced_script_module_optimized._save_for_lite_interpreter(out_path)

    # Load the model, Just to test if it is loadable
    model_mobile = torch.jit.load(out_path)


    # # You can also check for mobile support:
    # mobile_support = torch.jit.check(torch.jit.script(model))
    # if mobile_support:
    #     print("Model is compatible with PyTorch Mobile for Android.")
    # else:
    #     print("Model may not be fully compatible with PyTorch Mobile for Android.")

    return model_mobile


def write_predictions(predictions, output_path):

    for gaze_dir, frame_id in predictions:
        output_file = os.path.join(output_path, frame_id[0] + '.txt')
        np.savetxt(output_file, gaze_dir.cpu().numpy())


def main(args):

    if args.seed is not None:
        seed_everything(args.seed)

    use_wandb = args.wandb and not isdebugging()
    #use_wandb = True

    # Check whether dump_path exists and create one if not existing.
    output_path = os.path.join(args.experiment, '-'.join([args.run,datetime.now().strftime("%Y%m%d-%H%M%S")]))
    #output_path = os.path.join(args.experiment, args.run)
    args.output_path = output_path
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    model = CrossEncoder(args)
    monitor = 'val_error_after_calib'
    # input = torch.rand(1, args.channels, args.input_height, args.input_width)
    # prepare_model_for_android(model, input, '/tmp/')
    # if 1:
    #     input = torch.rand(1, args.channels, args.input_height, args.input_width)
    #     flops = FlopCountAnalysis(model, (input, input))
    #     table_both_stage = flop_count_table(flops)
    #     with open(os.path.join(output_path, 'flop_count.txt'), 'w') as f:
    #         f.write(table_both_stage)
    #
    #     onnx_file_path = os.path.join(output_path, 'model.onnx')
    #     torch.onnx.export(model, (input, input), onnx_file_path, verbose=True,
    #                       opset_version=10)


    if args.dataset_type == 'openeds_2020':
        pass
        # if args.binocular_model:
        #     data = openEDS2020BinocularDataModule(args)
        # else:
        #     data = openEDS2020GazeDataModule(args)
    elif args.dataset_type == 'nvgaze_real_vr':
        if args.batch_per_person:
            data = NVGazeMonocularDataModuleIntegratedCalib(args)
        else:
            data = NVGazeMonocularSinglePersonDataModule(args)
    elif args.dataset_type == 'ts':
        data = TSDataModuleUnsupervised(args)
    else:
        raise ValueError


    mode = 'min'
    save_top_k = 3
    # checkpoint_callback = ModelCheckpoint(dirpath=output_path, save_last=True,
    #                                       save_top_k=save_top_k, monitor=monitor, mode=mode,
    #                                       verbose=True)

    checkpoint_callback = ModelCheckpoint(dirpath=output_path, save_last=True,verbose=True, monitor=monitor,
                                          mode=mode, save_top_k=save_top_k)

    # create logger
    csv_logger = CSVLogger(save_dir=output_path, name='result')
    my_loggers = [csv_logger]
    use_neptune = True
    exp_name = os.path.basename(output_path)

    if use_wandb:

        wandb_logger = WandbLogger(project='sgaze', entity=args.wandb_entity, config=args, name=exp_name,
                               id=exp_name,save_code=True)
        my_loggers.append(wandb_logger)

    ckpt_path = args.ckpt_path if args.ckpt_path else None
    if ckpt_path and args.person_id_calib:
        ckpt_path = None

    trainer = pl.Trainer(    #resume_from_checkpoint=ckpt_path,
                             deterministic=True,
                             default_root_dir=output_path,
                             logger=my_loggers,
                             max_epochs=args.max_epochs,
                             accelerator=args.accelerator,
                             strategy=args.strategy,
                             precision=args.precision,
                             fast_dev_run=args.fast_dev_run,
                             callbacks=[checkpoint_callback],
                             num_sanity_val_steps=args.num_sanity_val_steps,
                             val_check_interval=args.val_check_interval,
                             accumulate_grad_batches=args.accumulate_grad_batches,
                             log_every_n_steps=args.log_every_n_steps,
                             limit_train_batches=args.limit_train_batches,
                             check_val_every_n_epoch=args.check_val_every_n_epoch,
                             limit_val_batches=args.limit_val_batches)

    if not args.validate and not args.predict:
        # train / val
        print('start training')
        trainer.fit(model, data)
        print('start evaluating')
    elif args.validate:
        # eval only
        #model = model.load_from_checkpoint(ckpt_path, hparams=args, channels=args.channels)
        print('start evaluating')
        trainer.validate(model, data)
    else:
        predictions = trainer.predict(model, data)
        write_predictions(predictions, output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = config.get_args(parser)

    main(args)

