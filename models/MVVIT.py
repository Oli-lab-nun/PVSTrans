import copy
import logging
import math
import random

from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from models.Model import Model
import models.configs as configs


def swish(x):
    return x * torch.sigmoid(x)


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    """numpy2tensor"""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query/"
ATTENTION_K = "MultiHeadDotProductAttention_1/key/"
ATTENTION_V = "MultiHeadDotProductAttention_1/value/"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out/"
FC_0 = "MlpBlock_3/Dense_0/"
FC_1 = "MlpBlock_3/Dense_1/"
ATTENTION_NORM = "LayerNorm_0/"
MLP_NORM = "LayerNorm_2/"

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)
        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # [1, 197, 768]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # [1, 197, 768]
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


# FFN
class Mlp(nn.Module):
    """
    In encoder, It's FFN
    """

    def __init__(self, config):
        super(Mlp, self).__init__()

        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config, vit_type, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.vit_type = vit_type
        self.hybrid = None
        # _pair()  img_size 448--> (448, 448)
        img_size = _pair(img_size)
        patch_size = _pair(config.patches["size"])

        if config.split == 'non-overlap':
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                           out_channels=config.hidden_size,
                                           kernel_size=patch_size,
                                           stride=patch_size)
        elif config.split == 'overlap':
            n_patches = ((img_size[0] - patch_size[0]) // config.slide_step + 1) * (
                    (img_size[1] - patch_size[1]) // config.slide_step + 1)
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                           out_channels=config.hidden_size,
                                           kernel_size=patch_size,
                                           stride=(config.slide_step, config.slide_step))

        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches + 1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x, global_token):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


# per layer
class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}/"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


# Part_selection
class Part_Attention(nn.Module):
    def __init__(self, num_views, img_tokens):
        super(Part_Attention, self).__init__()
        self.num_views = num_views
        self.img_tokens = img_tokens
        self.softmax = Softmax(dim=-1)

    def forward(self, views_attn_weights, global_attn_weights):
        length = len(views_attn_weights)
        last_map = views_attn_weights[0]
        for i in range(1, length):
            last_map = torch.matmul(views_attn_weights[i], last_map)
        last_map = last_map[:, :, 0, 1:]
        last_map = last_map.reshape(last_map.size()[0] // self.num_views, self.num_views, last_map.size()[-2],
                                    last_map.size()[-1])
        last_map = last_map.max(2).values

        global_last_map = global_attn_weights[0]
        for j in range(1, len(global_attn_weights)):
            global_last_map = torch.matmul(global_attn_weights[j], global_last_map)
        global_last_map = global_last_map[:, :, 0, 1:]
        global_last_map = global_last_map.max(1).values
        global_last_map = torch.round(self.softmax(global_last_map) * self.img_tokens).int()

        tokens_total = torch.sum(global_last_map, dim=1)
        for k in range(tokens_total.size()[0]):
            if tokens_total[k] > self.img_tokens:
                r = tokens_total[k] - self.img_tokens
                idx = random.randint(0, self.num_views - 1)
                global_last_map[k, idx] -= r
            elif tokens_total[k] < self.img_tokens:
                r = self.img_tokens - tokens_total[k]
                idx = random.randint(0, self.num_views - 1)
                global_last_map[k, idx] += r

        key_tokens_idx = []
        for i in range(global_last_map.size()[0]):
            idx = []
            for j in range(global_last_map.size()[1]):
                idx.append(last_map[i, j].argsort()[-global_last_map[i, j]:])
            key_tokens_idx.append(idx)
        return key_tokens_idx


# 自动选择patch的VIT-Encoder
class Encoder(nn.Module):
    def __init__(self, config, vit_type):
        super(Encoder, self).__init__()
        self.vit_type = vit_type
        self.layer = nn.ModuleList()
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))

        self.last_norm = LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, hidden_states):
        attn_weights = []
        for layer in self.layer:
            hidden_states, weights = layer(hidden_states)
            attn_weights.append(weights)
        encoded = self.last_norm(hidden_states)
        if self.vit_type == "part":
            return attn_weights, encoded, hidden_states
        elif self.vit_type == "global":
            return encoded


class Transformer(nn.Module):
    def __init__(self, config, img_size, vit_type, global_token=None):
        super(Transformer, self).__init__()
        self.vit_type = vit_type
        self.embeddings = Embeddings(config, vit_type=vit_type, img_size=img_size)
        self.encoder = Encoder(config, vit_type=vit_type)
        self.global_token = global_token

    def forward(self, input_ids):
        if self.vit_type == "part":
            embedding_output = self.embeddings(input_ids, self.global_token)
            attn_weights, encoded, un_last_norm_tokens = self.encoder(embedding_output)
            return attn_weights, encoded, un_last_norm_tokens
        elif self.vit_type == "global":
            embedding_output = self.embeddings(input_ids, self.global_token)
            encoded = self.encoder(embedding_output)
            return encoded

# Patch Tokens Selector
class AutoSelectEncoder(nn.Module):
    def __init__(self, config, img_size, num_classes, num_views, smoothing_value):
        super(AutoSelectEncoder, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.img_tokens = (self.img_size // config.patches.size[0]) ** 2
        self.num_views = num_views
        self.smoothing_value = smoothing_value
        self.layer = nn.ModuleList()
        for _ in range(config.transformer["other_num_layers1"]):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))
        self.part_select = Part_Attention(self.num_views, self.img_tokens)
        self.last_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.global_cls_head = Linear(config.hidden_size, num_classes)
        self.global_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

    def forward(self, un_last_norm_tokens, attn_weights, models_cls_tokens, labels=None):
        views_attn_weights = attn_weights
        B = models_cls_tokens.shape[0]
        global_tokens = self.global_token.expand(B, -1, -1)
        models_cls_tokens = torch.cat((global_tokens, models_cls_tokens), dim=1)
        global_attn_weights = []
        hidden_states = models_cls_tokens
        for layer in self.layer:
            hidden_states, weights = layer(hidden_states)
            global_attn_weights.append(weights)

        global_cls_token1 = hidden_states[:, 0]
        hidden_states = self.last_norm(hidden_states)
        global_cls_result1 = self.global_cls_head(hidden_states[:, 0])
        global_loss1 = None
        if labels is not None:
            if self.smoothing_value == 0:
                loss_fct = CrossEntropyLoss()
            else:
                loss_fct = LabelSmoothing(self.smoothing_value)
            global_loss1 = loss_fct(global_cls_result1.view(-1, self.num_classes), labels.view(-1))
        key_tokens_idx = self.part_select(views_attn_weights, global_attn_weights)
        un_last_norm_tokens = un_last_norm_tokens.reshape(un_last_norm_tokens.size()[0] // self.num_views,
                                                          self.num_views, un_last_norm_tokens.size()[-2],
                                                          un_last_norm_tokens.size()[-1])
        un_last_norm_tokens = un_last_norm_tokens[:, :, 1:, :]
        key_parts = []
        for i, data in enumerate(key_tokens_idx):
            parts = []
            for j, idx in enumerate(data):
                parts.extend(un_last_norm_tokens[i, j, idx])
            parts = torch.stack(parts)
            key_parts.append(parts)
        key_parts = torch.stack(key_parts)

        global_cls_token1 = global_cls_token1.reshape(global_cls_token1.size()[0], 1, global_cls_token1.size()[-1])
        return key_parts, global_loss1, global_cls_token1, global_cls_result1

    def load_from(self, weights):
        with torch.no_grad():
            self.global_token.copy_(np2th(weights["cls"]))
            self.last_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.last_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))
            for bname, block in self.named_children():
                if bname == "layer":
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=uname)


# Patch-View Net
class AutoSelectVIT(Model):
    def __init__(self, name, config, img_size=224, num_views=12, num_classes=21843, smoothing_value=0,
                 smoothing_value1_as_encoder=0):
        super(AutoSelectVIT, self).__init__(name)
        self.num_classes = num_classes
        self.num_views = num_views
        self.smoothing_value = smoothing_value
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vit_type="part")
        self.transEncoder = AutoSelectEncoder(config, img_size, num_classes=self.num_classes, num_views=self.num_views,
                                              smoothing_value=smoothing_value1_as_encoder)
        self.part_head = Linear(config.hidden_size, num_classes)

    def forward(self, x, labels=None):
        attn_weights, tokens, un_last_norm_tokens = self.transformer(x)
        views_cls_tokens = tokens[:, 0]
        # class_result = self.part_head(views_cls_tokens)
        new_B = views_cls_tokens.size()[0] // self.num_views
        models_cls_tokens = views_cls_tokens.reshape(new_B, self.num_views, -1)
        global_labels = labels[::self.num_views]
        key_parts, global_loss1, global_cls_token1, global_cls_result1 = self.transEncoder(un_last_norm_tokens,
                                                                                           attn_weights,
                                                                                           models_cls_tokens,
                                                                                           global_labels)
        un_norm_views_cls_tokens = un_last_norm_tokens[:, 0].reshape(new_B, self.num_views, -1)
        return un_norm_views_cls_tokens, key_parts, global_loss1, global_cls_token1, global_cls_result1

    def load_from(self, weights):
        with torch.no_grad():
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.last_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.last_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                if bname.startswith('part') == False:
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(
                    np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)


#Shape-View && Fusion
class LessVII(Model):
    def __init__(self, name, config, img_size, num_views, num_classes=21843, smoothing_value=0):
        super(LessVII, self).__init__(name)
        self.num_classes = num_classes
        self.num_views = num_views
        self.img_size = img_size
        self.smoothing_value = smoothing_value
        self.classifier = config.classifier
        self.layer = nn.ModuleList()
        for _ in range(config.transformer["other_num_layers2"]):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))
        self.class_head = Linear(config.hidden_size*self.num_views, num_classes)
        self.last_norm = LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, un_norm_views_cls_tokens, key_parts, global_cls_token1, labels=None):
        tokens = torch.cat((un_norm_views_cls_tokens, key_parts), dim=1)
        for layer in self.layer:
            tokens, weights = layer(tokens)
        tokens = self.last_norm(tokens)
        cls_tokens = tokens[:, 0: self.num_views]
        global_cls_result2 = self.class_head(cls_tokens.reshape(cls_tokens.size()[0], -1))
        global_loss2 = None
        if labels is not None:
            if self.smoothing_value == 0:
                loss_fct = CrossEntropyLoss()
            else:
                loss_fct = LabelSmoothing(self.smoothing_value)
            global_loss2 = loss_fct(global_cls_result2.view(-1, self.num_classes), labels.view(-1))
            return global_loss2, global_cls_result2
        else:
            return global_cls_result2

    def load_from(self, weights):
        with torch.no_grad():
            self.last_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.last_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))
            for bname, block in self.named_children():
                if bname == "layer":
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=uname)


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'testing': configs.get_testing(),
}
