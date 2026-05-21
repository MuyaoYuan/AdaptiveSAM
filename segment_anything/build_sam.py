# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer, ImageEncoderViTWithFuse, ImageEncoderViTWithFuseMutable, TinyViTWithFuse, TinyViT, MutableTinyViTWithFuse, MaskDecoderHQ

def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_sam = build_sam_vit_h

def build_sam_vit_h_hq(checkpoint=None):
    return _build_sam_hq(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )

# sam for efficient inference
def build_sam_efficient_inference_vit_h(checkpoint=None, adapter_type=-1, adapter_config=[-1,-1,-1,-1], mlp_config=[0.25]*20):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        efficient=True
    )

def build_sam_efficient_inference_vit_b(checkpoint=None, adapter_type=-1, adapter_config=[-1,-1,-1,-1], mlp_config=[0.25]*20):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        efficient=True,
        adapter_type=adapter_type,
    )

def build_sam_efficient_inference_vit_b_mutable(checkpoint=None, adapter_type=-1, adapter_config=[-1,-1,-1,-1], mlp_config=[0.25]*20):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        efficient=True,
        adapter_type=adapter_type,
        mutable=True,
        adapter_config=adapter_config,
        mlp_config=mlp_config,
    )

def build_sam_efficient_inference_vit_t(checkpoint=None, adapter_type=-1, adapter_config=[-1,-1,-1,-1], mlp_config=[0.25]*20):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    mobile_sam = Sam(
            image_encoder=TinyViTWithFuse(img_size=1024, in_chans=3, num_classes=1000,
                embed_dims=[64, 128, 160, 320],
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 5, 10],
                window_sizes=[7, 7, 14, 7],
                mlp_ratio=4.,
                drop_rate=0.,
                drop_path_rate=0.0,
                use_checkpoint=False,
                mbconv_expand_ratio=4.0,
                local_conv_size=3,
                layer_lr_decay=0.8,
                adapter_type=adapter_type,
            ),
            prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
            ),
            mask_decoder=MaskDecoder(
                    num_multimask_outputs=3,
                    transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
            ),
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
        )

    mobile_sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        if 'model' in state_dict.keys():
            missing_keys, unexpected_keys = mobile_sam.load_state_dict(state_dict['model'], strict=False)
            print('missing_keys: ', missing_keys)
            print('unexpected_keys: ', unexpected_keys)
        else:
            mobile_sam.load_state_dict(state_dict, strict=False)
    return mobile_sam

def build_sam_efficient_inference_vit_t_mutable(checkpoint=None, adapter_type=-1, adapter_config=[-1,-1,-1,-1], mlp_config=[0.25]*20):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    mobile_sam = Sam(
            image_encoder=MutableTinyViTWithFuse(img_size=1024, in_chans=3, num_classes=1000,
                embed_dims=[64, 128, 160, 320],
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 5, 10],
                window_sizes=[7, 7, 14, 7],
                mlp_ratio=4.,
                drop_rate=0.,
                drop_path_rate=0.0,
                use_checkpoint=False,
                mbconv_expand_ratio=4.0,
                local_conv_size=3,
                layer_lr_decay=0.8,
                adapter_type=adapter_type,
                adapter_config=adapter_config,
                mlp_config=mlp_config
            ),
            prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
            ),
            mask_decoder=MaskDecoder(
                    num_multimask_outputs=3,
                    transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
            ),
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
        )

    mobile_sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        if 'model' in state_dict.keys():
            missing_keys, unexpected_keys = mobile_sam.load_state_dict(state_dict['model'], strict=False)
            print('missing_keys: ', missing_keys)
            print('unexpected_keys: ', unexpected_keys)
        else:
            mobile_sam.load_state_dict(state_dict, strict=False)
    return mobile_sam

def build_sam_vit_t(checkpoint=None):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    mobile_sam = Sam(
            image_encoder=TinyViT(img_size=1024, in_chans=3, num_classes=1000,
                embed_dims=[64, 128, 160, 320],
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 5, 10],
                window_sizes=[7, 7, 14, 7],
                mlp_ratio=4.,
                drop_rate=0.,
                drop_path_rate=0.0,
                use_checkpoint=False,
                mbconv_expand_ratio=4.0,
                local_conv_size=3,
                layer_lr_decay=0.8
            ),
            prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
            ),
            mask_decoder=MaskDecoder(
                    num_multimask_outputs=3,
                    transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
            ),
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
        )

    mobile_sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        if 'model' in state_dict.keys():
            missing_keys, unexpected_keys = mobile_sam.load_state_dict(state_dict['model'], strict=False)
            print('missing_keys: ', missing_keys)
            print('unexpected_keys: ', unexpected_keys)
        else:
            mobile_sam.load_state_dict(state_dict, strict=False)
    return mobile_sam

def build_sam_vit_t_hq(checkpoint=None):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    mobile_sam = Sam(
            image_encoder=TinyViT(img_size=1024, in_chans=3, num_classes=1000,
                embed_dims=[64, 128, 160, 320],
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 5, 10],
                window_sizes=[7, 7, 14, 7],
                mlp_ratio=4.,
                drop_rate=0.,
                drop_path_rate=0.0,
                use_checkpoint=False,
                mbconv_expand_ratio=4.0,
                local_conv_size=3,
                layer_lr_decay=0.8,
                hq = True,
            ),
            prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
            ),
            mask_decoder=MaskDecoderHQ(
                    num_multimask_outputs=3,
                    transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
                vit_dim=160,
            ),
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
        )

    mobile_sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            # device = "cuda" if torch.cuda.is_available() else "cpu"
            device='cpu'
            state_dict = torch.load(f, map_location=device)
        info = mobile_sam.load_state_dict(state_dict, strict=False)
        print(info)
    for n, p in mobile_sam.named_parameters():
        if 'hf_token' not in n and 'hf_mlp' not in n and 'compress_vit_feat' not in n and 'embedding_encoder' not in n and 'embedding_maskfeature' not in n:
            p.requires_grad = False
    return mobile_sam

def build_sam_efficient_inference_vit_t_hq(checkpoint=None, adapter_type=-1, adapter_config=[-1,-1,-1,-1], mlp_config=[0.25]*20):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    mobile_sam = Sam(
            image_encoder=TinyViTWithFuse(img_size=1024, in_chans=3, num_classes=1000,
                embed_dims=[64, 128, 160, 320],
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 5, 10],
                window_sizes=[7, 7, 14, 7],
                mlp_ratio=4.,
                drop_rate=0.,
                drop_path_rate=0.0,
                use_checkpoint=False,
                mbconv_expand_ratio=4.0,
                local_conv_size=3,
                layer_lr_decay=0.8,
                adapter_type=adapter_type,
                hq = True,
            ),
            prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
            ),
            mask_decoder=MaskDecoderHQ(
                    num_multimask_outputs=3,
                    transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
                vit_dim=160,
            ),
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
        )

    mobile_sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            # device = "cuda" if torch.cuda.is_available() else "cpu"
            device='cpu'
            state_dict = torch.load(f, map_location=device)
            if 'model' in state_dict.keys():
                info = mobile_sam.load_state_dict(state_dict['model'], strict=False)
            else:
                info = mobile_sam.load_state_dict(state_dict, strict=False)
        print(info)
    for n, p in mobile_sam.named_parameters():
        if 'hf_token' not in n and 'hf_mlp' not in n and 'compress_vit_feat' not in n and 'embedding_encoder' not in n and 'embedding_maskfeature' not in n:
            p.requires_grad = False
    return mobile_sam

def build_sam_efficient_inference_vit_t_hq_mutable(checkpoint=None, adapter_type=-1, adapter_config=[-1,-1,-1,-1], mlp_config=[0.25]*20):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    mobile_sam = Sam(
            image_encoder=MutableTinyViTWithFuse(img_size=1024, in_chans=3, num_classes=1000,
                embed_dims=[64, 128, 160, 320],
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 5, 10],
                window_sizes=[7, 7, 14, 7],
                mlp_ratio=4.,
                drop_rate=0.,
                drop_path_rate=0.0,
                use_checkpoint=False,
                mbconv_expand_ratio=4.0,
                local_conv_size=3,
                layer_lr_decay=0.8,
                adapter_config=adapter_config,
                mlp_config=mlp_config,
                hq = True,
            ),
            prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
            ),
            mask_decoder=MaskDecoderHQ(
                    num_multimask_outputs=3,
                    transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
                vit_dim=160,
            ),
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
        )

    mobile_sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            # device = "cuda" if torch.cuda.is_available() else "cpu"
            device='cpu'
            state_dict = torch.load(f, map_location=device)
            if 'model' in state_dict.keys():
                info = mobile_sam.load_state_dict(state_dict['model'], strict=False)
            else:
                info = mobile_sam.load_state_dict(state_dict, strict=False)
        print(info)
    for n, p in mobile_sam.named_parameters():
        if 'hf_token' not in n and 'hf_mlp' not in n and 'compress_vit_feat' not in n and 'embedding_encoder' not in n and 'embedding_maskfeature' not in n:
            p.requires_grad = False
    return mobile_sam

sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
    "vit_t": build_sam_vit_t,
    "vit_h_hq": build_sam_vit_h_hq,
    "vit_t_hq": build_sam_vit_t_hq,
    "efficient_vit_h": build_sam_efficient_inference_vit_h,
    "efficient_vit_b": build_sam_efficient_inference_vit_b,
    "efficient_vit_t": build_sam_efficient_inference_vit_t,
    "efficient_vit_t_hq": build_sam_efficient_inference_vit_t_hq,
    "mutable_efficient_vit_b": build_sam_efficient_inference_vit_b_mutable,
    "mutable_efficient_vit_t": build_sam_efficient_inference_vit_t_mutable,
    "mutable_efficient_vit_t_hq": build_sam_efficient_inference_vit_t_hq_mutable,
}


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    efficient=False,
    adapter_type=-1,
    mutable=False,
    adapter_config=None,
    mlp_config=None
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    if not efficient:
        sam = Sam(
            image_encoder=ImageEncoderViT(
                depth=encoder_depth,
                embed_dim=encoder_embed_dim,
                img_size=image_size,
                mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=encoder_num_heads,
                patch_size=vit_patch_size,
                qkv_bias=True,
                use_rel_pos=True,
                global_attn_indexes=encoder_global_attn_indexes,
                window_size=14,
                out_chans=prompt_embed_dim,
            ),
            prompt_encoder=PromptEncoder(
                embed_dim=prompt_embed_dim,
                image_embedding_size=(image_embedding_size, image_embedding_size),
                input_image_size=(image_size, image_size),
                mask_in_chans=16,
            ),
            mask_decoder=MaskDecoder(
                num_multimask_outputs=3,
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
            ),
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
        )
    else:
        if not mutable:
            sam = Sam(
                image_encoder=ImageEncoderViTWithFuse(
                    depth=encoder_depth,
                    embed_dim=encoder_embed_dim,
                    img_size=image_size,
                    mlp_ratio=4,
                    norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                    num_heads=encoder_num_heads,
                    patch_size=vit_patch_size,
                    qkv_bias=True,
                    use_rel_pos=True,
                    global_attn_indexes=encoder_global_attn_indexes,
                    window_size=14,
                    out_chans=prompt_embed_dim,
                    adapter_type=adapter_type,
                ),
                prompt_encoder=PromptEncoder(
                    embed_dim=prompt_embed_dim,
                    image_embedding_size=(image_embedding_size, image_embedding_size),
                    input_image_size=(image_size, image_size),
                    mask_in_chans=16,
                ),
                mask_decoder=MaskDecoder(
                    num_multimask_outputs=3,
                    transformer=TwoWayTransformer(
                        depth=2,
                        embedding_dim=prompt_embed_dim,
                        mlp_dim=2048,
                        num_heads=8,
                    ),
                    transformer_dim=prompt_embed_dim,
                    iou_head_depth=3,
                    iou_head_hidden_dim=256,
                ),
                pixel_mean=[123.675, 116.28, 103.53],
                pixel_std=[58.395, 57.12, 57.375],
            )
        else:
            sam = Sam(
                image_encoder=ImageEncoderViTWithFuseMutable(
                    depth=encoder_depth,
                    embed_dim=encoder_embed_dim,
                    img_size=image_size,
                    mlp_ratio=4,
                    norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                    num_heads=encoder_num_heads,
                    patch_size=vit_patch_size,
                    qkv_bias=True,
                    use_rel_pos=True,
                    global_attn_indexes=encoder_global_attn_indexes,
                    window_size=14,
                    out_chans=prompt_embed_dim,
                    adapter_config=adapter_config,
                    mlp_config=mlp_config
                ),
                prompt_encoder=PromptEncoder(
                    embed_dim=prompt_embed_dim,
                    image_embedding_size=(image_embedding_size, image_embedding_size),
                    input_image_size=(image_size, image_size),
                    mask_in_chans=16,
                ),
                mask_decoder=MaskDecoder(
                    num_multimask_outputs=3,
                    transformer=TwoWayTransformer(
                        depth=2,
                        embedding_dim=prompt_embed_dim,
                        mlp_dim=2048,
                        num_heads=8,
                    ),
                    transformer_dim=prompt_embed_dim,
                    iou_head_depth=3,
                    iou_head_hidden_dim=256,
                ),
                pixel_mean=[123.675, 116.28, 103.53],
                pixel_std=[58.395, 57.12, 57.375],
            )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        if 'model' in state_dict.keys():
            sam.load_state_dict(state_dict['model'], strict=False)
        else:
            sam.load_state_dict(state_dict, strict=False)
    return sam

def _build_sam_hq(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
            hq = True,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoderHQ(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            vit_dim=encoder_embed_dim,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            # device = "cuda" if torch.cuda.is_available() else "cpu"
            device='cpu'
            state_dict = torch.load(f, map_location=device)
        info = sam.load_state_dict(state_dict, strict=False)
        print(info)
    for n, p in sam.named_parameters():
        if 'hf_token' not in n and 'hf_mlp' not in n and 'compress_vit_feat' not in n and 'embedding_encoder' not in n and 'embedding_maskfeature' not in n:
            p.requires_grad = False

    return sam
