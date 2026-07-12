import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter

from models.segment_anything.modeling import Sam
from models.segment_anything import sam_model_registry
from models.common.adapter import Adapter, FLRA_Adapter, SW_Adapter
from models.common.adapter import _adapter_attn, _adapter_mlp


class DetailEnhancement(nn.Module):
    """Edge Guided Refiner — multi-scale edge enhancement decoder.

    Extracts multi-scale edges via parallel depth-wise convolutions (3×3, 5×5, 7×7),
    then fuses them with upsampled decoder features for refined segmentation.
    """

    def __init__(self, img_dim=64, feature_dim=64, num_classes=6, norm=nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        self.num_classes = num_classes

        # Extract local features from input image: Fl = C3x3(I)
        self.img_in_conv = nn.Sequential(
            nn.Conv2d(3, img_dim, 3, padding=1, bias=False),
            norm(img_dim),
            act()
        )

        # Multi-scale depth-wise convolutions (parallel paths, Eq.8)
        self.dconv_3x3 = nn.Conv2d(img_dim, img_dim, 3, padding=1, groups=img_dim, bias=False)
        self.dconv_5x5 = nn.Conv2d(img_dim, img_dim, 5, padding=2, groups=img_dim, bias=False)
        self.dconv_7x7 = nn.Conv2d(img_dim, img_dim, 7, padding=3, groups=img_dim, bias=False)

        # Edge enhancement: C1x1 for each scale (Eq.9)
        self.edge_conv_3x3 = nn.Conv2d(img_dim, img_dim, 1, bias=False)
        self.edge_conv_5x5 = nn.Conv2d(img_dim, img_dim, 1, bias=False)
        self.edge_conv_7x7 = nn.Conv2d(img_dim, img_dim, 1, bias=False)

        # Average pooling for edge extraction
        self.ap = nn.AvgPool2d(3, stride=1, padding=1)

        # Multi-scale edge fusion: F'ee = C3x3(Concat(F1_ee, F2_ee, F3_ee)) (Eq.10)
        self.edge_fusion = nn.Sequential(
            nn.Conv2d(img_dim * 3, img_dim, 3, padding=1, bias=False),
            norm(img_dim),
            act()
        )

        # Channel adjustment for decoder features
        self.channel_adjust = nn.Conv2d(256, feature_dim, 1, bias=False)

        # Decoder feature upsampling: F's = C3x3(UP(C3x3(UP(C3x3(Fs))))) (Eq.7)
        self.feature_upsample = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            norm(feature_dim),
            act(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            norm(feature_dim),
            act(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            norm(feature_dim),
            act(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )

        # C3x3 applied to F'ee before element-wise addition with Fl (Eq.11)
        self.final_edge_conv = nn.Sequential(
            nn.Conv2d(img_dim, img_dim, 3, padding=1, bias=False),
            norm(img_dim),
            act()
        )

        # MLP: integrated feature has 64 channels, projected by 1x1 conv to N classes (Eq.11)
        self.out_conv = nn.Conv2d(feature_dim + img_dim, num_classes, 1)

        # Handle 768-dimensional features (ViT-H)
        self.feature_proj_768 = nn.Sequential(
            nn.Conv2d(768, 256, 1, bias=False),
            norm(256),
            act(),
            nn.Conv2d(256, feature_dim, 1, bias=False),
            norm(feature_dim),
            act()
        )

    def forward(self, img, decoder_feature):
        """
        Args:
            img: Original input image [B, 3, H, W]
            decoder_feature: Encoder/decoder feature [B, C, 64, 64]
        Returns:
            Se: Enhanced segmentation mask [B, num_classes, H, W]
        """
        B, C, H_feat, W_feat = decoder_feature.shape
        _, _, H, W = img.shape

        # Handle different channel dimensions
        if C == 768:
            decoder_feature_32 = self.feature_proj_768(decoder_feature)
        elif C == 256:
            decoder_feature_32 = self.channel_adjust(decoder_feature)
        elif C == 32:
            decoder_feature_32 = decoder_feature
        else:
            if not hasattr(self, 'dynamic_conv'):
                self.dynamic_conv = nn.Conv2d(C, 32, 1, bias=False).to(decoder_feature.device)
            decoder_feature_32 = self.dynamic_conv(decoder_feature)

        # Eq.7: Upsample decoder features to image size
        Fs_up = self.feature_upsample(decoder_feature_32)
        if H != 512 or W != 512:
            Fs_up = F.interpolate(Fs_up, size=(H, W), mode='bilinear', align_corners=False)

        # Eq.8: Extract local features + multi-scale depth-wise convs (parallel)
        Fl = self.img_in_conv(img)
        F1_e = self.dconv_3x3(Fl)
        F2_e = self.dconv_5x5(Fl)
        F3_e = self.dconv_7x7(Fl)

        # Eq.9: Edge enhancement for each scale
        F1_ee = self.edge_conv_3x3(F1_e - self.ap(F1_e)) + F1_e
        F2_ee = self.edge_conv_5x5(F2_e - self.ap(F2_e)) + F2_e
        F3_ee = self.edge_conv_7x7(F3_e - self.ap(F3_e)) + F3_e

        # Eq.10: Multi-scale edge fusion
        F_ee = self.edge_fusion(torch.cat([F1_ee, F2_ee, F3_ee], dim=1))

        # Eq.11: Se = MLP(Concat(F's, Fl + C3x3(F'ee)))
        Se = self.out_conv(torch.cat([Fs_up, Fl + self.final_edge_conv(F_ee)], dim=1))

        return Se


class FESAM(nn.Module):
    """FESAM: Frequency-Enhanced SAM with adapter-based fine-tuning.

    Applies frequency-domain adapters to SAM's image encoder and mask decoder
    for efficient parameter-efficient fine-tuning on downstream segmentation tasks.

    Args:
        sam_model: a SAM model instance (ViT-based).
        encoder_attn_adapter: adapter class for encoder attention blocks.
        decoder_mlp_adapter: adapter class for decoder MLP blocks.
        decoder_attn_adapter: adapter class for decoder attention blocks.
        use_mask_decoder_adapter: if True, insert adapters into the mask decoder;
            otherwise the mask decoder is fully fine-tuned.
        use_detail_enhancement: if True, enable the DetailEnhancement edge-refiner.
        num_classes: number of output segmentation classes.
    """

    def __init__(
        self,
        sam_model: Sam,
        encoder_attn_adapter = FLRA_Adapter,
        decoder_mlp_adapter = Adapter,
        decoder_attn_adapter = Adapter,
        use_mask_decoder_adapter: bool = True,
        use_detail_enhancement: bool = True,
        num_classes: int = 6,
    ):
        super(FESAM, self).__init__()
        self.sam = sam_model
        self.encoder_attn_adapter = encoder_attn_adapter
        self.decoder_mlp_adapter = decoder_mlp_adapter
        self.decoder_attn_adapter = decoder_attn_adapter
        self.use_mask_decoder_adapter = use_mask_decoder_adapter
        self.use_detail_enhancement = use_detail_enhancement
        self.num_classes = num_classes

        self.encoder_mlp_adapter = SW_Adapter(
            in_features=768,
            out_features=768,
            depth=4,
            act_layer=nn.GELU,
            skip_connect=True,
        )

        self.image_encoder_adapters = nn.ModuleList()
        self.mask_decoder_adapters = nn.ModuleList()
        self.final_attn_adapter = None

        if self.use_detail_enhancement:
            self.detail_enhancement = DetailEnhancement(
                img_dim=64,
                feature_dim=64,
                num_classes=num_classes,
                norm=nn.BatchNorm2d,
                act=nn.ReLU
            )

        self.coarse_mask_proj = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, 1, bias=False),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(),
        )

        self.fusion_gate = nn.Sequential(
            nn.Conv2d(num_classes * 2, num_classes, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_classes),
            nn.Sigmoid(),
        )

        self._freeze_parameters()
        self._adapt_image_encoder()

        if self.use_mask_decoder_adapter:
            self._adapt_mask_decoder_twab()
            self._adapt_mask_decoder_fab()

        self.print_model_parameters_info(True)

    def _freeze_parameters(self):
        if self.use_mask_decoder_adapter:
            for param in self.sam.parameters():
                param.requires_grad = False
        else:
            for param in self.sam.image_encoder.parameters():
                param.requires_grad = False

    def _adapt_image_encoder(self):
        for _index, block in enumerate(self.sam.image_encoder.blocks):
            attn_dim = block.attn.proj.out_features
            mlp_dim = block.mlp.lin2.out_features

            adapter_attn = self.encoder_attn_adapter(attn_dim, attn_dim)
            self.image_encoder_adapters.append(adapter_attn)

            block.attn.proj = _adapter_attn(
                block_attn_proj=block.attn.proj,
                adapter_attn=adapter_attn,
            )
            block.mlp = _adapter_mlp(
                block_mlp=block.mlp,
                adapter_mlp=self.encoder_mlp_adapter,
            )

    def _adapt_mask_decoder_twab(self):
        for _index, block in enumerate(self.sam.mask_decoder.transformer.layers):
            self_attn_dim = block.self_attn.out_proj.out_features
            cross_attn_token_to_image_dim = block.cross_attn_token_to_image.out_proj.out_features
            cross_attn_image_to_token_dim = block.cross_attn_image_to_token.out_proj.out_features
            mlp_in_dim = block.mlp.lin1.in_features
            mlp_out_dim = block.mlp.lin2.out_features

            adapter_self_attn = self.decoder_attn_adapter(self_attn_dim, self_attn_dim)
            adapter_cross_attn_token_to_image = self.decoder_attn_adapter(cross_attn_token_to_image_dim, cross_attn_token_to_image_dim)
            adapter_cross_attn_image_to_token = self.decoder_attn_adapter(cross_attn_image_to_token_dim, cross_attn_image_to_token_dim)
            adapter_mlp = self.decoder_mlp_adapter(mlp_in_dim, mlp_out_dim, skip_connect=False)

            self.mask_decoder_adapters.append(adapter_self_attn)
            self.mask_decoder_adapters.append(adapter_cross_attn_token_to_image)
            self.mask_decoder_adapters.append(adapter_cross_attn_image_to_token)
            self.mask_decoder_adapters.append(adapter_mlp)

            block.self_attn.out_proj = _adapter_attn(
                block_attn_proj=block.self_attn.out_proj,
                adapter_attn=adapter_self_attn,
            )
            block.cross_attn_token_to_image.out_proj = _adapter_attn(
                block_attn_proj=block.cross_attn_token_to_image.out_proj,
                adapter_attn=adapter_cross_attn_token_to_image,
            )
            block.cross_attn_image_to_token.out_proj = _adapter_attn(
                block_attn_proj=block.cross_attn_image_to_token.out_proj,
                adapter_attn=adapter_cross_attn_image_to_token,
            )
            block.mlp = _adapter_mlp(
                block_mlp=block.mlp,
                adapter_mlp=adapter_mlp,
            )

    def _adapt_mask_decoder_fab(self):
        final_attn = self.sam.mask_decoder.transformer.final_attn_token_to_image
        final_attn_dim = final_attn.out_proj.out_features

        self.final_attn_adapter = self.decoder_attn_adapter(final_attn_dim, final_attn_dim)
        final_attn.out_proj = _adapter_attn(
            block_attn_proj=final_attn.out_proj,
            adapter_attn=self.final_attn_adapter,
        )

    def print_model_parameters_info(self, print_more_info=False):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params_list = [p for p in self.parameters() if p.requires_grad]
        trainable_params_count = sum(p.numel() for p in trainable_params_list)

        if print_more_info:
            trainable_params_detailed = [p for p in self.named_parameters() if p[1].requires_grad]
            print("Number of trainable parameters:", len(trainable_params_detailed))
            for name, param in trainable_params_detailed:
                print(f"{name}: {param.shape}")

        print("trainable params: {} || all params: {} || trainable ratio: {:.2%}"
              .format(trainable_params_count, total_params, trainable_params_count / total_params))

    def save_adapters_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')

        adapter_tensors = {}
        prompt_encoder_tensors = {}
        mask_decoder_tensors = {}

        for i, adapter_attn in enumerate(self.image_encoder_adapters):
            adapter_attn.save_parameters(adapter_tensors, f"adapter_enc_attn_{i:03d}")
        self.encoder_mlp_adapter.save_parameters(adapter_tensors, f"adapter_enc_mlp")

        if self.use_mask_decoder_adapter:
            for i in range(len(self.mask_decoder_adapters) // 4):
                adapter_self_attn = self.mask_decoder_adapters[4 * i]
                adapter_t2img_attn = self.mask_decoder_adapters[4 * i + 1]
                adapter_img2t_attn = self.mask_decoder_adapters[4 * i + 2]
                adapter_mlp = self.mask_decoder_adapters[4 * i + 3]

                adapter_self_attn.save_parameters(adapter_tensors, f"adapter_dec_self_attn_{i:03d}")
                adapter_t2img_attn.save_parameters(adapter_tensors, f"adapter_dec_t2img_attn{i:03d}")
                adapter_img2t_attn.save_parameters(adapter_tensors, f"adapter_dec_img2t_attn{i:03d}")
                adapter_mlp.save_parameters(adapter_tensors, f"adapter_dec_mlp_{i:03d}")

            self.final_attn_adapter.save_parameters(adapter_tensors, f"adapter_dec_final_attn")
        else:
            if isinstance(self.sam, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                state_dict = self.sam.module.state_dict()
            else:
                state_dict = self.sam.state_dict()
            prompt_encoder_tensors = {k: v for k, v in state_dict.items() if 'prompt_encoder' in k}
            mask_decoder_tensors = {k: v for k, v in state_dict.items() if 'mask_decoder' in k}

        detail_enhance_tensors = {}
        if self.use_detail_enhancement:
            detail_enhance_tensors = {k: v for k, v in self.detail_enhancement.state_dict().items()}
            detail_enhance_tensors = {f"detail_enhancement.{k}": v for k, v in detail_enhance_tensors.items()}

        fusion_tensors = {f"fusion_gate.{k}": v for k, v in self.fusion_gate.state_dict().items()}
        coarse_proj_tensors = {f"coarse_mask_proj.{k}": v for k, v in self.coarse_mask_proj.state_dict().items()}

        merged_dict = {
            **adapter_tensors, **prompt_encoder_tensors, **mask_decoder_tensors,
            **detail_enhance_tensors, **fusion_tensors, **coarse_proj_tensors
        }
        torch.save(merged_dict, filename)

    def load_adapters_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
        state_dict = torch.load(filename, map_location='cpu')
        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()

        for i, adapter_attn in enumerate(self.image_encoder_adapters):
            adapter_attn.load_parameters(state_dict, f"adapter_enc_attn_{i:03d}")
        self.encoder_mlp_adapter.load_parameters(state_dict, f"adapter_enc_mlp")

        if self.use_mask_decoder_adapter:
            for i in range(len(self.mask_decoder_adapters) // 4):
                adapter_self_attn = self.mask_decoder_adapters[4 * i]
                adapter_t2img_attn = self.mask_decoder_adapters[4 * i + 1]
                adapter_img2t_attn = self.mask_decoder_adapters[4 * i + 2]
                adapter_mlp = self.mask_decoder_adapters[4 * i + 3]

                adapter_self_attn.load_parameters(state_dict, f"adapter_dec_self_attn_{i:03d}")
                adapter_t2img_attn.load_parameters(state_dict, f"adapter_dec_t2img_attn{i:03d}")
                adapter_img2t_attn.load_parameters(state_dict, f"adapter_dec_img2t_attn{i:03d}")
                adapter_mlp.load_parameters(state_dict, f"adapter_dec_mlp_{i:03d}")

            self.final_attn_adapter.load_parameters(state_dict, f"adapter_dec_final_attn")
        else:
            prompt_encoder_keys = [k for k in sam_keys if 'prompt_encoder' in k]
            prompt_encoder_values = [state_dict[k] for k in prompt_encoder_keys]
            prompt_encoder_new_state_dict = dict(zip(prompt_encoder_keys, prompt_encoder_values))
            sam_dict.update(prompt_encoder_new_state_dict)

            mask_decoder_keys = [k for k in sam_keys if 'mask_decoder' in k]
            mask_decoder_values = [state_dict[k] for k in mask_decoder_keys]
            mask_decoder_new_state_dict = dict(zip(mask_decoder_keys, mask_decoder_values))
            sam_dict.update(mask_decoder_new_state_dict)

        if self.use_detail_enhancement:
            detail_enhance_keys = [k for k in state_dict.keys() if k.startswith('detail_enhancement.')]
            if detail_enhance_keys:
                detail_enhance_state_dict = {k.replace('detail_enhancement.', ''): state_dict[k] for k in detail_enhance_keys}
                self.detail_enhancement.load_state_dict(detail_enhance_state_dict, strict=False)

        for module_name, module in [
            ('fusion_gate', self.fusion_gate),
            ('coarse_mask_proj', self.coarse_mask_proj),
        ]:
            module_keys = [k for k in state_dict.keys() if k.startswith(f'{module_name}.')]
            if module_keys:
                module_state_dict = {k.replace(f'{module_name}.', ''): state_dict[k] for k in module_keys}
                module.load_state_dict(module_state_dict, strict=False)

        self.sam.load_state_dict(sam_dict)

    def forward(self, batched_input, multimask_output, image_size):
        if self.use_detail_enhancement:
            if isinstance(batched_input, list):
                original_images = torch.stack([x["image"] for x in batched_input])
            else:
                original_images = batched_input

            batched_input = [{"image": original_images[i], "original_size": (image_size, image_size)} for i in range(original_images.shape[0])]

            preprocessed_images = torch.stack([self.sam.preprocess(x["image"]) for x in batched_input], dim=0)

            image_embeddings = self.sam.image_encoder(preprocessed_images)
            if isinstance(image_embeddings, tuple):
                image_embeddings = image_embeddings[0]

            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=None, boxes=None, masks=None,
            )

            low_res_masks, iou_predictions = self.sam.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )

            coarse_masks = self.sam.postprocess_masks(
                low_res_masks,
                input_size=preprocessed_images.shape[-2:],
                original_size=(image_size, image_size),
            )

            enhanced_masks = self.detail_enhancement(original_images, image_embeddings)

            coarse_projected = self.coarse_mask_proj(coarse_masks)
            if coarse_projected.shape[-2:] != enhanced_masks.shape[-2:]:
                coarse_projected = F.interpolate(
                    coarse_projected, size=enhanced_masks.shape[-2:],
                    mode='bilinear', align_corners=False
                )

            gate = self.fusion_gate(torch.cat([coarse_projected, enhanced_masks], dim=1))
            final_masks = gate * enhanced_masks + (1 - gate) * coarse_projected

            outputs = {
                "masks": final_masks,
                "coarse_masks": coarse_masks,
                "enhanced_masks": enhanced_masks,
                "iou_predictions": iou_predictions,
                "low_res_logits": low_res_masks,
            }

            return outputs, image_embeddings
        else:
            if isinstance(batched_input, torch.Tensor):
                batched_input = [{"image": batched_input[i], "original_size": (image_size, image_size)} for i in range(batched_input.shape[0])]
            return self.sam(batched_input, multimask_output, image_size), None
