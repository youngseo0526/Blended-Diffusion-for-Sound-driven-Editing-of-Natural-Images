import os
from pathlib import Path
from optimization.constants import ASSETS_DIR_NAME, RANKED_RESULTS_DIR

from utils.metrics_accumulator import MetricsAccumulator
from utils.video import save_video

from numpy import random
from optimization.augmentations import ImageAugmentations

from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import functional as TF
from torch.nn.functional import mse_loss
from optimization.losses import range_loss, d_clip_loss
import lpips
import numpy as np

from CLIP import clip
from guided_diffusion.guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from utils.visualization import show_tensor_image, show_editied_masked_image

from soundclip_loss import AudioEncoder, SoundCLIPLoss
import librosa
from librosa.core import audio
import cv2
import timm

class ImageEditor:
    def __init__(self, args) -> None:
        self.args = args
        os.makedirs(self.args.output_path, exist_ok=True)

        self.ranked_results_path = Path(os.path.join(self.args.output_path, RANKED_RESULTS_DIR))
        os.makedirs(self.ranked_results_path, exist_ok=True)

        if self.args.export_assets:
            self.assets_path = Path(os.path.join(self.args.output_path, ASSETS_DIR_NAME))
            os.makedirs(self.assets_path, exist_ok=True)
        if self.args.seed is not None:
            torch.manual_seed(self.args.seed)
            np.random.seed(self.args.seed)
            random.seed(self.args.seed)

        self.model_config = model_and_diffusion_defaults()
        self.model_config.update(
            {
                "attention_resolutions": "32, 16, 8",
                "class_cond": self.args.model_output_size == 512,
                "diffusion_steps": 1000,
                "rescale_timesteps": True,
                "timestep_respacing": self.args.timestep_respacing,
                "image_size": self.args.model_output_size,
                "learn_sigma": True,
                "noise_schedule": "linear",
                "num_channels": 256,
                "num_head_channels": 64,
                "num_res_blocks": 2,
                "resblock_updown": True,
                "use_fp16": True,
                "use_scale_shift_norm": True,
            }
        )
        # Load models
        self.device = torch.device(
            f"cuda:{self.args.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        print("Using device:", self.device)

        self.model, self.diffusion = create_model_and_diffusion(**self.model_config)
        self.model.load_state_dict(
            torch.load(
                "checkpoints/256x256_diffusion_uncond.pt"
                if self.args.model_output_size == 256
                else "checkpoints/512x512_diffusion.pt",
                map_location="cpu",
            )
        )
        self.model.requires_grad_(False).eval().to(self.device)
        for name, param in self.model.named_parameters():
            if "qkv" in name or "norm" in name or "proj" in name:
                param.requires_grad_()
        if self.model_config["use_fp16"]:
            self.model.convert_to_fp16()

        self.clip_model = (
            clip.load("ViT-B/16", device=self.device, jit=False)[0].eval().requires_grad_(False)
        )
        self.clip_size = self.clip_model.visual.input_resolution
        self.clip_normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        )
        self.lpips_model = lpips.LPIPS(net="vgg").to(self.device)

        self.image_augmentations = ImageAugmentations(self.clip_size, self.args.aug_num)
        self.metrics_accumulator = MetricsAccumulator()
        
        self.soundclip_loss = SoundCLIPLoss(self.args)
        self.audio_encoder = AudioEncoder()
        

    def unscale_timestep(self, t):
        unscaled_timestep = (t * (self.diffusion.num_timesteps / 1000)).long()

        return unscaled_timestep

    def clip_loss(self, x_in, audio_embed):
        clip_loss = torch.tensor(0)

        if self.mask is not None:
            masked_input = x_in * self.mask
        else:
            masked_input = x_in
        augmented_input = self.image_augmentations(masked_input).add(1).div(2)
        clip_in = self.clip_normalize(augmented_input)
        image_embeds = self.clip_model.encode_image(clip_in).float()
        dists = d_clip_loss(image_embeds, audio_embed)  # image_embeds: ([8, 512]), text_embed: ([1, 512])

        # We want to sum over the averages
        for i in range(self.args.batch_size):
            # We want to average at the "augmentations level"
            clip_loss = clip_loss + dists[i :: self.args.batch_size].mean()

        return clip_loss

    def unaugmented_clip_distance(self, x, audio_embed):
        #print(1)
        x = F.resize(x, [self.clip_size, self.clip_size])
        image_embeds = self.clip_model.encode_image(x).float()
        dists = d_clip_loss(image_embeds, audio_embed)
        print("dists:", dists)

        return dists.item()

    def edit_image_by_audio(self):
        self.image_size = (self.model_config["image_size"], self.model_config["image_size"])
        self.init_image_pil = Image.open(self.args.init_image).convert("RGB")
        self.init_image_pil = self.init_image_pil.resize(self.image_size, Image.LANCZOS)  # type: ignore
        self.init_image = (
            TF.to_tensor(self.init_image_pil).to(self.device).unsqueeze(0).mul(2).sub(1)
        )

        if self.args.export_assets:
            img_path = self.assets_path / Path(self.args.output_file)
            self.init_image_pil.save(img_path)

        self.mask = torch.ones_like(self.init_image, device=self.device)
        self.mask_pil = None
        if self.args.mask is not None:
            self.mask_pil = Image.open(self.args.mask).convert("RGB")
            if self.mask_pil.size != self.image_size:
                self.mask_pil = self.mask_pil.resize(self.image_size, Image.NEAREST)  # type: ignore
            image_mask_pil_binarized = ((np.array(self.mask_pil) > 0.5) * 255).astype(np.uint8)
            if self.args.invert_mask:
                image_mask_pil_binarized = 255 - image_mask_pil_binarized
                self.mask_pil = TF.to_pil_image(image_mask_pil_binarized)
            self.mask = TF.to_tensor(Image.fromarray(image_mask_pil_binarized))
            self.mask = self.mask[0, ...].unsqueeze(0).unsqueeze(0).to(self.device)

            if self.args.export_assets:
                mask_path = self.assets_path / Path(
                    self.args.output_file.replace(".png", "_mask.png")
                )
                self.mask_pil.save(mask_path)

        # audio
        y, sr = librosa.load(self.args.audio_path, sr=44100)
        n_mels = 128
        time_length = 864
        audio_inputs = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        audio_inputs = librosa.power_to_db(audio_inputs, ref=np.max) / 80.0 + 1  # convert log scale

        audio_inputs = audio_inputs

        zero = np.zeros((n_mels, time_length))
        resize_resolution = 512
        h, w = audio_inputs.shape
        if w >= time_length:
            j = 0
            j = random.randint(0, w-time_length)
            audio_inputs = audio_inputs[:,j:j+time_length]
        else:
            zero[:,:w] = audio_inputs[:,:w]
            audio_inputs = zero

        audio_inputs = cv2.resize(audio_inputs, (n_mels, resize_resolution))
        audio_inputs = np.array([audio_inputs])
        audio_inputs = torch.from_numpy(audio_inputs.reshape((1, 1, n_mels, resize_resolution))).float().cuda()
        

        def cond_fn(x, t, y=None):
            if self.args.prompt == "":
                return torch.zeros_like(x)

            with torch.enable_grad():
                x = x.detach().requires_grad_()
                t = self.unscale_timestep(t)

                out = self.diffusion.p_mean_variance(
                    self.model, x, t, clip_denoised=False, model_kwargs={"y": y}
                )
                
                soundclip_loss = SoundCLIPLoss(self.args)
                # audio_inputs: torch.Size([1, 1, 128, 512])
                # init_image: torch.Size([1, 3, 256, 256])
                #cosine_distance_loss = soundclip_loss(image=self.init_image, audio=audio_inputs)
                
                #print("cosine_distance_loss: ", cosine_distance_loss.size)

                fac = self.diffusion.sqrt_one_minus_alphas_cumprod[t[0].item()]
                x_in = out["pred_xstart"] * fac + x * (1 - fac)
                #print("x_in:", x_in.shape)

                #pred_image = F.resize(x, [self.args.model_output_size, self.args.model_output_size])
                #print("pred_image: ", pred_image.shape)
                #pred_image = self.clip_model.encode_image(x).float()
                #pred_image = d_clip_loss
                #cosine_distance_loss = soundclip_loss(image=x_in, audio=audio_inputs)
                #cosine_distance_loss = soundclip_loss(image=self.init_image, audio=audio_inputs)
                
                # x_in = out["pred_xstart"]
                loss = torch.tensor(0)
                if self.args.clip_guidance_lambda != 0:
                    cosine_distance_loss = soundclip_loss(image=x_in, audio=audio_inputs)
                    cosine_distance_loss = cosine_distance_loss[1] * self.args.clip_guidance_lambda
                    loss = loss + cosine_distance_loss
                    self.metrics_accumulator.update_metric("cosine_distance_loss", cosine_distance_loss.item())

                if self.args.range_lambda != 0:
                    r_loss = range_loss(out["pred_xstart"]).sum() * self.args.range_lambda
                    loss = loss + r_loss
                    self.metrics_accumulator.update_metric("range_loss", r_loss.item())

                if self.args.background_preservation_loss:
                    if self.mask is not None:
                        masked_background = x_in * (1 - self.mask)
                    else:
                        masked_background = x_in

                    if self.args.lpips_sim_lambda:
                        loss = (
                            loss
                            + self.lpips_model(masked_background, self.init_image).sum()
                            * self.args.lpips_sim_lambda
                        )
                    if self.args.l2_sim_lambda:
                        loss = (
                            loss
                            + mse_loss(masked_background, self.init_image) * self.args.l2_sim_lambda
                        )

                return -torch.autograd.grad(loss, x)[0]

        @torch.no_grad()
        def postprocess_fn(out, t):
            if self.mask is not None:
                background_stage_t = self.diffusion.q_sample(self.init_image, t[0])
                background_stage_t = torch.tile(
                    background_stage_t, dims=(self.args.batch_size, 1, 1, 1)
                )
                out["sample"] = out["sample"] * self.mask + background_stage_t * (1 - self.mask)

            return out

        save_image_interval = self.diffusion.num_timesteps // 5
        for iteration_number in range(self.args.iterations_num):
            print(f"Start iterations {iteration_number}")

            sample_func = (
                self.diffusion.ddim_sample_loop_progressive
                if self.args.ddim
                else self.diffusion.p_sample_loop_progressive
            )
            samples = sample_func(
                self.model,
                (
                    self.args.batch_size,
                    3,
                    self.model_config["image_size"],
                    self.model_config["image_size"],
                ),
                clip_denoised=False,
                model_kwargs={}
                if self.args.model_output_size == 256
                else {
                    "y": torch.zeros([self.args.batch_size], device=self.device, dtype=torch.long)
                },
                cond_fn=cond_fn,
                progress=True,
                skip_timesteps=self.args.skip_timesteps,
                init_image=self.init_image,
                postprocess_fn=None if self.args.local_clip_guided_diffusion else postprocess_fn,
                randomize_class=True,
            )

            intermediate_samples = [[] for i in range(self.args.batch_size)]
            total_steps = self.diffusion.num_timesteps - self.args.skip_timesteps - 1

            #print("1 audio_inputs: ", audio_inputs.shape)
            
            for j, sample in enumerate(samples):
                self.image_size = (self.model_config["image_size"], self.model_config["image_size"])
                self.pred_image_pil = Image.open(self.args.init_image).convert("RGB")
                self.pred_image_pil = self.pred_image_pil.resize(self.image_size, Image.LANCZOS)  # type: ignore
                self.pred_image = (
                    TF.to_tensor(self.pred_image_pil).to(self.device).unsqueeze(0).mul(2).sub(1)
                )
                should_save_image = j % save_image_interval == 0 or j == total_steps
                if should_save_image or self.args.save_video:
                    self.metrics_accumulator.print_average_metric()

                    for b in range(self.args.batch_size):
                        pred_image = sample["pred_xstart"][b]
                        visualization_path = Path(
                            os.path.join(self.args.output_path, self.args.output_file)
                        )
                        visualization_path = visualization_path.with_stem(
                            f"{visualization_path.stem}_i_{iteration_number}_b_{b}"
                        )

                        if (
                            self.mask is not None
                            and self.args.enforce_background
                            and j == total_steps
                            and not self.args.local_clip_guided_diffusion
                        ):
                            pred_image = (
                                self.init_image[0] * (1 - self.mask[0]) + pred_image * self.mask[0]
                            )
                        pred_image = pred_image.add(1).div(2).clamp(0, 1)
                        pred_image_pil = TF.to_pil_image(pred_image)
                        masked_pred_image = self.mask * pred_image.unsqueeze(0)
                        #print("masked_pred_image: ", masked_pred_image.shape)

                        #pred_image = F.resize(pred_image_pil, [self.args.model_output_size, self.args.model_output_size])

                        
                        audio_embeding = SoundCLIPLoss(self.args)
                        #audio_embed, cosine_distance_loss = audio_embeding(image=self.init_image, audio=audio_inputs)
                        audio_embed, cosine_distance_loss = audio_embeding(image=masked_pred_image, audio=audio_inputs)

                        #print("3 audio_embed: ", audio_embed.shape)
                        final_distance = self.unaugmented_clip_distance(
                            masked_pred_image, audio_embed
                        )
                        formatted_distance = f"{final_distance:.4f}"

                        if self.args.export_assets:
                            pred_path = self.assets_path / visualization_path.name
                            pred_image_pil.save(pred_path)

                        if j == total_steps:
                            path_friendly_distance = formatted_distance.replace(".", "")
                            ranked_pred_path = self.ranked_results_path / (
                                path_friendly_distance + "_" + visualization_path.name
                            )
                            pred_image_pil.save(ranked_pred_path)

                        intermediate_samples[b].append(pred_image_pil)
                        if should_save_image:
                            show_editied_masked_image(
                                #title=self.args.prompt,
                                title=self.args.audio_path,
                                source_image=self.init_image_pil,
                                edited_image=pred_image_pil,
                                mask=self.mask_pil,
                                path=visualization_path,
                                distance=formatted_distance,
                            )

            if self.args.save_video:
                for b in range(self.args.batch_size):
                    video_name = self.args.output_file.replace(
                        ".png", f"_i_{iteration_number}_b_{b}.avi"
                    )
                    video_path = os.path.join(self.args.output_path, video_name)
                    save_video(intermediate_samples[b], video_path)

    ##########################################################################                

    def reconstruct_image(self):
        init = Image.open(self.args.init_image).convert("RGB")
        init = init.resize(
            self.image_size,  # type: ignore
            Image.LANCZOS,
        )
        init = TF.to_tensor(init).to(self.device).unsqueeze(0).mul(2).sub(1)

        samples = self.diffusion.p_sample_loop_progressive(
            self.model,
            (1, 3, self.model_config["image_size"], self.model_config["image_size"],),
            clip_denoised=False,
            model_kwargs={}
            if self.args.model_output_size == 256
            else {"y": torch.zeros([self.args.batch_size], device=self.device, dtype=torch.long)},
            cond_fn=None,
            progress=True,
            skip_timesteps=self.args.skip_timesteps,
            init_image=init,
            randomize_class=True,
        )
        save_image_interval = self.diffusion.num_timesteps // 5
        max_iterations = self.diffusion.num_timesteps - self.args.skip_timesteps - 1

        for j, sample in enumerate(samples):
            if j % save_image_interval == 0 or j == max_iterations:
                print()
                filename = os.path.join(self.args.output_path, self.args.output_file)
                TF.to_pil_image(sample["pred_xstart"][0].add(1).div(2).clamp(0, 1)).save(filename)
