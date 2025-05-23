import pdb
import copy
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .imagenet_templates import IMAGENET_TEMPLATES

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg, zero_shot_model=False):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    if not zero_shot_model:
        design_details = {
            "trainer": "IVLP",
            "vision_depth": cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_VISION,
            "language_depth": cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_TEXT,
            "vision_ctx": cfg.TRAINER.PROMPTSRC.N_CTX_VISION,
            "language_ctx": cfg.TRAINER.PROMPTSRC.N_CTX_TEXT,
        }
        model = clip.build_model(state_dict or model.state_dict(), design_details)
    else:
        # Return original CLIP model for generating frozen VL features
        design_details = {
            "trainer": "IVLP",
            "vision_depth": 0,
            "language_depth": 0,
            "vision_ctx": 0,
            "language_ctx": 0,
        }
        model = clip.build_model(state_dict or model.state_dict(), design_details)
        return model
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = (
            x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
            @ self.text_projection
        )

        return x


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        # Make sure Language depth >= 1
        assert cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_TEXT >= 1, (
            "In Independent VL prompting, Language prompt depth should be >=1"
            "\nPlease use VPT trainer if you want to learn only vision "
            "branch"
        )
        n_ctx = cfg.TRAINER.PROMPTSRC.N_CTX_TEXT # 4
        ctx_init = cfg.TRAINER.PROMPTSRC.CTX_INIT # "a photo of a"
        dtype = clip_model.dtype # torch.float16
        ctx_dim = clip_model.ln_final.weight.shape[0] # 512
        clip_imsize = clip_model.visual.input_resolution # 224
        cfg_imsize = cfg.INPUT.SIZE[0] # 224
        assert (
            cfg_imsize == clip_imsize
        ), f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and n_ctx <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")  # "a photo of a"
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init) # torch.Size([1, 77])
            # prompt: tensor([[49406,   320,  1125,   539,   320, 49407,     0,     0,     0,     0,
            #  0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            #  0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            #  0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            #  0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            #  0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            #  0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            #  0,     0,     0,     0,     0,     0,     0]])
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype) # torch.Size([1, 77, 512]) [bs, seq_len, dim]
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :] # torch.Size([4, 512]) 开头是sos token
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype) # torch.Size([4, 512])
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print(f"Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(
            f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.PROMPTSRC.N_CTX_VISION}"
        )
        self.ctx = nn.Parameter(ctx_vectors) # torch.Size([4, 512])
        # 至此，self.ctx 是初始化好的 context vectors
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat(
            [clip.tokenize(p) for p in prompts]
        )  # (n_cls, n_tkn)
        # Also create frozen CLIP
        clip_model_temp = load_clip_to_cpu(cfg, True).float().cuda()
        clip_model_temp_image = load_clip_to_cpu(cfg, True)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            self.ZS_image_encoder = clip_model_temp_image.visual
            # Now pre-compute the frozen VL embeddings
            all_teacher_features = []
            # Using multiple text templates to ensure textual diversity during training
            for single_template in IMAGENET_TEMPLATES:
                x = [single_template.replace("{}", name) for name in classnames]
                x_tokenized = torch.cat([clip.tokenize(p) for p in x])
                text_features = clip_model_temp.encode_text(x_tokenized.cuda())
                all_teacher_features.append(text_features.unsqueeze(1))

        self.fixed_embeddings = torch.cat(all_teacher_features, dim=1).mean(dim=1)
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        
        self.classnames = classnames
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx # torch.Size([4, 512])
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1) # torch.Size([127, 4, 512])

        prefix = self.token_prefix 
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts
        


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)

    def forward(self, image, label=None, return_features=False):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts = self.prompt_learner()
        # Compute the prompted image and text features
        text_features = self.text_encoder(prompts, tokenized_prompts)
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # Compute the prompted logits
        logits = logit_scale * image_features @ text_features.t()
        if self.prompt_learner.training:
            # Now calculate the frozen pre-trained features
            fixed_embeddings = (
                self.prompt_learner.fixed_embeddings
            )  # precomputed pre-trained frozen textual features
            fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(
                dim=-1, keepdim=True
            )
            with torch.no_grad():
                zero_shot_features = self.prompt_learner.ZS_image_encoder(
                    image.type(self.dtype)
                )
                zero_shot_features = zero_shot_features / zero_shot_features.norm(
                    dim=-1, keepdim=True
                )
                # Compute pre-trained frozen visual features
                zero_shot_logits = (
                    logit_scale
                    * zero_shot_features.cuda()
                    @ fixed_embeddings.half().cuda().t()
                )

            return (
                F.cross_entropy(logits, label),
                text_features,
                fixed_embeddings,
                zero_shot_features,
                image_features,
                zero_shot_logits,
                logits,
            )
        else:
            if return_features:
                return logits, image_features
            return logits


@TRAINER_REGISTRY.register()
class PromptSRC(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.PROMPTSRC.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.PROMPTSRC.PREC == "fp32" or cfg.TRAINER.PROMPTSRC.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            else:
                if "ZS_image_encoder" in name:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}")
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)
        # Cosine scheduler
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1
        N = cfg.OPTIM.MAX_EPOCH
        mean = cfg.TRAINER.PROMPTSRC.GPA_MEAN
        stdev = cfg.TRAINER.PROMPTSRC.GPA_STD
        gauss = self.get_gauss(mean, stdev)
        self.gauss = np.array([gauss(a) for a in range(1, N + 1)])
        self.gauss = self.gauss / sum(self.gauss)
        self.scaler = GradScaler() if cfg.TRAINER.PROMPTSRC.PREC == "amp" else None
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)
        # Keep model with GPA
        self.previous_model_gpa = None

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.PROMPTSRC.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            (
                loss_ce,
                normalized_text_features,
                zs_clip_text_embeddings,
                zs_image_embedd,
                image_ft,
                zero_shot_logits,
                logits,
            ) = model(image, label)
            # Calculate the L_SCL_text loss
            loss_scl_text = (
                F.l1_loss(
                    normalized_text_features,
                    zs_clip_text_embeddings.cuda(),
                    reduction="mean",
                )
                * self.cfg.TRAINER.PROMPTSRC.TEXT_LOSS_WEIGHT
            )
            # Calculate the L_SCL_image loss
            loss_scl_image = (
                F.l1_loss(image_ft, zs_image_embedd.cuda(), reduction="mean")
                * self.cfg.TRAINER.PROMPTSRC.IMAGE_LOSS_WEIGHT
            )
            # Now calculate L_SCL_logits
            L_SCL_logits = (
                F.kl_div(
                    F.log_softmax(logits / 1, dim=1),
                    F.log_softmax(zero_shot_logits / 1, dim=1),
                    reduction="sum",
                    log_target=True,
                )
                * (1 * 1)
                / logits.numel()
            )
            L_SCL = L_SCL_logits + loss_scl_text + loss_scl_image
            loss = loss_ce + L_SCL
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
            # Means one epoch is completed, perform GPA
            self.step_counter = self.step_counter + 1
            current_epoch_weight = self.gauss[self.step_counter - 2]
            current_model_weights = copy.deepcopy(model.state_dict())
            weighted_state_dict = self.state_dict_weighting(
                current_model_weights, current_epoch_weight
            )
            if self.previous_model_gpa is None:
                self.previous_model_gpa = weighted_state_dict
            else:
                self.previous_model_gpa = self.state_dict_add(
                    weighted_state_dict, self.previous_model_gpa
                )

        if self.step_counter == self.model.total_epochs + 1:
            print("Using GPA model for final inference...")
            model.load_state_dict(self.previous_model_gpa)
            self.model.load_state_dict(self.previous_model_gpa)
        return loss_summary

    def state_dict_weighting(self, main_dict, weightage, prompt_only=False):
        # Average all parameters
        updated_dict = copy.deepcopy(main_dict)
        if not prompt_only:
            for key in main_dict:
                updated_dict[key] = main_dict[key] * weightage
            return updated_dict
        else:
            return main_dict * weightage

    def state_dict_add(self, dict1, dict2, prompt_only=False):
        # Average all parameters
        if not prompt_only:
            modified_dict = dict2
            for key in dict1:
                modified_dict[key] = modified_dict[key] + dict1[key]
            return modified_dict
        else:
            return dict1 + dict2

    def get_gauss(self, mu, sigma):
        gauss = lambda x: (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x - mu) / sigma) ** 2
        )
        return gauss

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print(
                "Loading weights to {} "
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def test(self, split=None):
        import torch.nn as nn
        from tqdm import tqdm
        import numpy as np
        from sklearn.metrics import accuracy_score, adjusted_rand_score
        from sklearn.cluster import KMeans

        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        visual_features = []
        all_labels = []  # 用于保存真实标签
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output, image_features = self.model_inference(input, return_features=True)
            self.evaluator.process(output, label)
            visual_features.append(image_features)
            all_labels.append(label.cpu().numpy())  # 保存标签

        # 聚类代码
        visual_features = [
            vf.cpu().numpy() if isinstance(vf, torch.Tensor) else vf
            for vf in visual_features
        ]
        visual_features = np.vstack(visual_features)
        all_labels = np.concatenate(all_labels, axis=0)
        print("all labels shape:", all_labels.shape)

        # 使用KMeans 进行聚类
        print("kmeans clustering...")
        kmeans = KMeans(n_clusters=self.num_classes, random_state=0).fit(
            visual_features
        )
        print("kmeans clustering done")

        # 计算聚类效果
        cluster_labels = kmeans.labels_
        # print(f'Cluster Labels: {cluster_labels}')

        # 使用匈牙利算法对齐标签
        accuracy = self.cluster_accuracy(all_labels, cluster_labels)
        print(f"Clustering Accuracy (Hungarian): {accuracy * 100:.2f}%")

        # 计算调整兰德指数（ARI）
        ari_score = adjusted_rand_score(all_labels, cluster_labels)
        print(f"Adjusted Rand Index (ARI): {ari_score:.4f}")

        metrics = self.calculate_intra_class_metrics(visual_features, all_labels)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    @staticmethod
    # 对齐聚类标签和真实标签
    def cluster_accuracy(true_labels, cluster_labels):
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        """
        将聚类标签重新排列，使其与真实标签尽可能匹配
        """
        D = max(cluster_labels.max(), true_labels.max()) + 1
        cost_matrix = np.zeros((D, D), dtype=np.int64)

        for i in range(len(cluster_labels)):
            cost_matrix[cluster_labels[i], true_labels[i]] += 1

        row_ind, col_ind = linear_sum_assignment(cost_matrix.max() - cost_matrix)
        return cost_matrix[row_ind, col_ind].sum() / len(cluster_labels)
