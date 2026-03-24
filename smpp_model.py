"""
Main SMPP Model  —  论文整体架构

两阶段流水线：
    Stage 1（分类训练）：CLIP + Prototype + Prompt + CrossModal -> L_g + L_o + L_c
    Stage 2（回归推理）：冻结模型，提取特征向量 F -> LightGBM / CatBoost

特征向量 F（论文公式 11）：
    F = [f_I(x), f_T(t), w_i^G, w_i^L, x_tilde, V_tilde, T_tilde, s]
    w_i^G / w_i^L 取预测类别对应的 Prompt 向量（非全类均值）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import clip

from custom_clip_encoder import CustomCLIPTextEncoder, create_custom_text_encoder
from dual_prompt_learner  import DualGrainedPromptLearner, create_prompt_learner
from cross_modal_fusion   import CrossModalFusion
from smpp_loss            import SMPPLoss


class SMPPModel(nn.Module):

    def __init__(
        self,
        num_classes: int = 77,
        clip_model_name: str = "ViT-B/32",
        global_ctx_length: int = 16,
        local_ctx_length: int = 8,
        fusion_hidden_dim: int = 512,
        fusion_num_heads: int = 8,
        fusion_num_layers: int = 2,
        prompt_dropout: float = 0.1,
        class_names: list = None,
        device: str = "cuda",
        freeze_image: bool = True,
        freeze_text: bool = True,
        freeze_text_embedding: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.device      = device

        # 1. CLIP
        self.clip_model, self.preprocess = clip.load(clip_model_name, device=device)
        self.clip_model = self.clip_model.float()
        self.feature_dim = self.clip_model.visual.output_dim

        # 2. 自定义文本编码器
        self.text_encoder = CustomCLIPTextEncoder(self.clip_model)

        # 3. 双粒度 Prompt Learner
        self.prompt_learner = create_prompt_learner(
            clip_model=self.clip_model,
            num_classes=num_classes,
            class_names=class_names,
            global_ctx_length=global_ctx_length,
            local_ctx_length=local_ctx_length,
            prompt_dropout=prompt_dropout,
            device=device,
        )

        # 4. 跨模态融合
        self.cross_modal_fusion = CrossModalFusion(
            image_dim=self.feature_dim,
            text_dim=self.feature_dim,
            hidden_dim=fusion_hidden_dim,
            num_heads=fusion_num_heads,
            num_layers=fusion_num_layers,
        )

        # 5. 原型（训练前加载）
        self.register_buffer("visual_prototypes",
                             torch.zeros(num_classes, self.feature_dim))
        self.register_buffer("textual_prototypes",
                             torch.zeros(num_classes, self.feature_dim))

        # 6. 损失函数（超参数对齐论文 4.1.3）
        self.criterion = SMPPLoss(
            temperature=0.07,
            spatial_temperature=0.1,
            temperature_visual=0.07,
            temperature_textual=0.07,
        )

        # 7. 冻结 CLIP 主干
        self._freeze_clip(freeze_image, freeze_text, freeze_text_embedding)

    # ------------------------------------------------------------------

    def _freeze_clip(self, freeze_image, freeze_text, freeze_text_embedding):
        if freeze_image:
            for p in self.clip_model.visual.parameters():
                p.requires_grad = False
        if freeze_text:
            for p in self.clip_model.transformer.parameters():
                p.requires_grad = False
        if freeze_text_embedding:
            for attr in ("token_embedding", "ln_final"):
                mod = getattr(self.clip_model, attr, None)
                if mod is not None:
                    for p in mod.parameters():
                        p.requires_grad = False
            for attr in ("positional_embedding", "text_projection"):
                t = getattr(self.clip_model, attr, None)
                if t is not None and isinstance(t, torch.Tensor):
                    t.requires_grad = False

    def load_prototypes(self, visual_prototypes, textual_prototypes):
        self.visual_prototypes  = visual_prototypes.to(self.device)
        self.textual_prototypes = textual_prototypes.to(self.device)

    def encode_image(self, images):
        feats = self.clip_model.encode_image(images).float()
        return feats / feats.norm(dim=-1, keepdim=True)

    def encode_text(self, text_tokens):
        return self.text_encoder(text_tokens)

    # ------------------------------------------------------------------

    def forward(self, images, text_tokens, labels=None, return_features=False):
        B = images.shape[0]

        # Step 1-3
        x    = self.encode_image(images)
        h, H = self.encode_text(text_tokens)
        G, L = self.prompt_learner()

        # Step 4: 加权原型
        V = self.visual_prototypes.to(dtype=x.dtype)
        T = self.textual_prototypes.to(dtype=x.dtype)
        weights = F.softmax(torch.matmul(x, V.T), dim=1)
        V_w = torch.matmul(weights, V)
        T_w = torch.matmul(weights, T)

        # Step 5: 融合（加权原型版，产生 x_tilde）
        x_tilde, V_w_tilde, T_w_tilde = self.cross_modal_fusion(x, V_w, T_w)

        output = {
            "image_features":      x,
            "text_global":         h,
            "text_sequence":       H,
            "enhanced_image":      x_tilde,
            "enhanced_V_weighted": V_w_tilde,
            "enhanced_T_weighted": T_w_tilde,
            "global_prompt_features": G,
            "local_prompt_features":  L,
        }

        # Step 6: 损失（对全部 K 个原型融合）
        if labels is not None:
            all_eV, all_eT = [], []
            for i in range(self.num_classes):
                _, eV, eT = self.cross_modal_fusion(
                    x,
                    V[i:i+1].expand(B, -1),
                    T[i:i+1].expand(B, -1),
                )
                all_eV.append(eV)
                all_eT.append(eT)
            eV_all = torch.stack(all_eV, dim=1)   # [B, K, hidden]
            eT_all = torch.stack(all_eT, dim=1)

            loss_dict = self.criterion(
                text_global_features=h,
                text_sequence_features=H,
                global_prompt_features=G,
                local_prompt_features=L,
                enhanced_image_features=x_tilde,
                enhanced_visual_prototypes=eV_all,
                enhanced_textual_prototypes=eT_all,
                labels=labels,
            )
            output.update(loss_dict)

        # Step 7: 特征提取
        if return_features:
            pred_labels = self._get_pred_labels(output)
            output["all_features"] = self.extract_all_features(
                image_features=x,
                text_global=h,
                enhanced_image=x_tilde,
                enhanced_V=V_w_tilde,
                enhanced_T=T_w_tilde,
                global_prompts=G,
                local_prompts=L,
                pred_labels=pred_labels,
            )

        return output

    def _get_pred_labels(self, output):
        if all(k in output for k in
               ("logits_global", "logits_local", "logits_visual")):
            avg = (output["logits_global"]
                   + output["logits_local"]
                   + output["logits_visual"]) / 3
            return avg.argmax(dim=-1)
        return None

    # ------------------------------------------------------------------
    # F = [f_I(x), f_T(t), w^G_i, w^L_i, x_tilde, V_tilde, T_tilde, s]
    # ------------------------------------------------------------------

    def extract_all_features(
        self,
        image_features,
        text_global,
        enhanced_image,
        enhanced_V,
        enhanced_T,
        global_prompts,
        local_prompts,
        pred_labels=None,
        user_features=None,
    ):
        B = image_features.shape[0]
        if pred_labels is not None:
            pred_labels = pred_labels.to(global_prompts.device)
            w_G = global_prompts[pred_labels]
            w_L = local_prompts[pred_labels]
        else:
            w_G = global_prompts.mean(0).unsqueeze(0).expand(B, -1)
            w_L = local_prompts.mean(0).unsqueeze(0).expand(B, -1)

        parts = [image_features, text_global, w_G, w_L,
                 enhanced_image, enhanced_V, enhanced_T]
        if user_features is not None:
            parts.append(user_features.to(image_features.device))
        return torch.cat(parts, dim=-1)

    # ------------------------------------------------------------------

    def predict_class(self, images, text_tokens):
        with torch.no_grad():
            out = self.forward(images, text_tokens)
            avg = (out["logits_global"]
                   + out["logits_local"]
                   + out["logits_visual"]) / 3
            return avg.argmax(dim=-1)

    def extract_features_for_regression(self, images, text_tokens,
                                         user_features=None):
        with torch.no_grad():
            # 第一次 forward：获取 pred_labels
            out = self.forward(images, text_tokens, labels=None,
                               return_features=False)
            pred_labels = self._get_pred_labels(out)
            G, L = self.prompt_learner()
            feats = self.extract_all_features(
                image_features=out["image_features"],
                text_global=out["text_global"],
                enhanced_image=out["enhanced_image"],
                enhanced_V=out["enhanced_V_weighted"],
                enhanced_T=out["enhanced_T_weighted"],
                global_prompts=G,
                local_prompts=L,
                pred_labels=pred_labels,
                user_features=user_features,
            )
        return feats


# ======================================================================
# Trainer
# ======================================================================

class SMPPTrainer:

    def __init__(self, model, optimizer, device="cuda"):
        self.model     = model
        self.optimizer = optimizer
        self.device    = device
        self.model.to(device)

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = total_g = total_o = total_c = 0.0
        correct = total = 0

        for step, batch in enumerate(dataloader):
            if len(batch) >= 4:
                images, text_tokens, labels, *_ = batch
            else:
                images, text_tokens, labels = batch

            images      = images.to(self.device)
            text_tokens = text_tokens.to(self.device)
            labels      = labels.to(self.device)

            out  = self.model(images, text_tokens, labels)
            loss = out["loss"]
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_g    += out["loss_global"].item()
            total_o    += out["loss_local"].item()
            total_c    += out["loss_visual"].item()

            avg = (out["logits_global"] + out["logits_local"]
                   + out["logits_visual"]) / 3
            correct += (avg.argmax(-1) == labels).sum().item()
            total   += labels.size(0)

            if (step + 1) % 100 == 0:
                n = step + 1
                print(f"Epoch {epoch}  Step {n}/{len(dataloader)}  "
                      f"Loss={total_loss/n:.4f}  Acc={100*correct/total:.2f}%")

        n = len(dataloader)
        print(f"\nEpoch {epoch}  Loss={total_loss/n:.4f}  "
              f"Lg={total_g/n:.4f}  Lo={total_o/n:.4f}  Lc={total_c/n:.4f}  "
              f"Acc={100*correct/total:.2f}%")
        return total_loss / n, 100 * correct / total

    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval()
        total_loss = correct = total = 0

        for batch in dataloader:
            if len(batch) >= 4:
                images, text_tokens, labels, *_ = batch
            else:
                images, text_tokens, labels = batch

            images      = images.to(self.device)
            text_tokens = text_tokens.to(self.device)
            labels      = labels.to(self.device)

            out = self.model(images, text_tokens, labels)
            total_loss += out["loss"].item()
            avg = (out["logits_global"] + out["logits_local"]
                   + out["logits_visual"]) / 3
            correct += (avg.argmax(-1) == labels).sum().item()
            total   += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        avg_acc  = 100 * correct / total
        print(f"Validation — Loss={avg_loss:.4f}  Acc={avg_acc:.2f}%")
        return avg_loss, avg_acc

    def save_checkpoint(self, path, epoch, loss):
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss":                 loss,
        }, path)
        print(f"[Checkpoint] Saved -> {path}")

    def load_checkpoint(self, path):
        ckpt       = torch.load(path, map_location=self.device)
        state_dict = ckpt["model_state_dict"]
        is_dp      = isinstance(self.model, torch.nn.DataParallel)
        has_pfx    = any(k.startswith("module.") for k in state_dict)
        if is_dp and not has_pfx:
            state_dict = {f"module.{k}": v for k, v in state_dict.items()}
        elif (not is_dp) and has_pfx:
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=False)
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"[Checkpoint] Loaded <- {path}  "
              f"Epoch={ckpt['epoch']}  Loss={ckpt['loss']:.4f}")
        return ckpt["epoch"], ckpt["loss"]