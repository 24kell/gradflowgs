#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import time
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from scene.gaussian_model import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, views, gaussians, background, pipeline):
    render_path = os.path.join(model_path, name, args.text, "images")
    makedirs(render_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx)))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, opt : OptimizationParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        if not opt.include_mask:
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
            print("choice 1. ")
        else:
            scene = Scene(dataset, gaussians, shuffle=False)
            checkpoint_iteration = opt.identify_iter 
            checkpoint = os.path.join(scene.model_path,
                                      'chkpnt' + str(checkpoint_iteration) + '.pth')
            print('choice 2. checkpoint :', checkpoint)
            (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
            gaussians.restore(model_params, args)
            start_time = time.perf_counter()
            gaussians.segment()
            end_time = (time.perf_counter() - start_time) * 1000  # 秒 → 毫秒
            print(f"{end_time:.3f} ms")


        if not dataset.white_background:
            print("Rendering with black background")
        bg_color = [1,1,1] # if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            views = scene.getTrainCameras()
            render_set(dataset.model_path, "train", views, gaussians, background, pipeline)
        if not skip_test:
            views = scene.getTestCameras()
            render_set(dataset.model_path, "test", views, gaussians, background, pipeline)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--text", type=str, default=None)
    args = get_combined_args(parser)
    print("Rendering segmentation, " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), opt.extract(args), args.skip_train, args.skip_test)