"""
here is the mian backbone for DLF
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ...subNets import BertTextEncoder
from ...subNets.transformers_encoder.transformer import TransformerEncoder   

class DLF(nn.Module):
    def __init__(self, args):
        super(DLF, self).__init__()
        if args.use_bert:
            self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers,
                                              pretrained=args.pretrained)
        self.use_bert = args.use_bert
        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        if args.dataset_name == 'mosi':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 375
        if args.dataset_name == 'mosei':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 500
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        self.num_heads = nheads     
        self.layers = args.nlevels 
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask
        self.lfa_mode = getattr(args, 'lfa_mode', 'original')
        if self.lfa_mode not in {'original', 'selective_null', 'selective_null_gain'}:
            raise ValueError(
                f"Unknown lfa_mode={self.lfa_mode!r}; expected 'original', "
                "'selective_null' or 'selective_null_gain'."
            )
        self.fusion_gate_mode = getattr(args, 'fusion_gate_mode', 'none')
        if self.fusion_gate_mode not in {'none', 'fixed', 'global', 'sample'}:
            raise ValueError(
                f"Unknown fusion_gate_mode={self.fusion_gate_mode!r}; expected "
                "'none', 'fixed', 'global' or 'sample'."
            )
        self.fusion_gate_init = float(getattr(args, 'fusion_gate_init', 0.9))
        if not 0.0 < self.fusion_gate_init < 1.0:
            raise ValueError(
                f"fusion_gate_init must be in (0, 1), got {self.fusion_gate_init}."
            )
        gate_bias = math.log(self.fusion_gate_init / (1.0 - self.fusion_gate_init))

        # LFA sequence pooling: replaces the [-1] summarize on the private LFA
        # paths. 'last' reproduces the original model byte-for-byte.
        self.lfa_pool_mode = getattr(args, 'lfa_pool_mode', 'last')
        if self.lfa_pool_mode not in {'last', 'mean', 'attn'}:
            raise ValueError(
                f"Unknown lfa_pool_mode={self.lfa_pool_mode!r}; expected "
                "'last', 'mean' or 'attn'."
            )
        raw_paths = getattr(args, 'lfa_pool_paths', []) or []
        invalid_paths = set(raw_paths) - {'text', 'audio', 'video'}
        if invalid_paths:
            raise ValueError(
                f"Unknown lfa_pool_paths={sorted(invalid_paths)!r}; expected a "
                "subset of {'text', 'audio', 'video'}."
            )
        self.lfa_pool_paths = set(raw_paths)
        self.conv1d_kernel_size_l = args.conv1d_kernel_size_l
        combined_dim_low = self.d_a
        combined_dim_high = self.d_a 
        combined_dim = (self.d_l + self.d_a + self.d_v ) + self.d_l * 3  
        
        output_dim = 1

        # 1. Temporal convolutional layers for initial feature
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)

        # 2. Modality-specific encoder
        self.encoder_s_l = self.get_network(self_type='l', layers = self.layers)       
        self.encoder_s_v = self.get_network(self_type='v', layers = self.layers)
        self.encoder_s_a = self.get_network(self_type='a', layers = self.layers)

        #   Modality-shared encoder 
        self.encoder_c = self.get_network(self_type='l', layers = self.layers)        
        

        # 3. Decoder for reconstruct three modalities
        self.decoder_l = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0, bias=False)     
        self.decoder_v = nn.Conv1d(self.d_v * 2, self.d_v, kernel_size=1, padding=0, bias=False)
        self.decoder_a = nn.Conv1d(self.d_a * 2, self.d_a, kernel_size=1, padding=0, bias=False)

        # for calculate cosine sim between s_x
        self.proj_cosine_l = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)     
        self.proj_cosine_v = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.proj_cosine_a = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)

        # for align c_l, c_v, c_a
        self.align_c_l = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
        self.align_c_v = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.align_c_a = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)

        self.self_attentions_c_l = self.get_network(self_type='l')
        self.self_attentions_c_v = self.get_network(self_type='v')
        self.self_attentions_c_a = self.get_network(self_type='a')

        self.proj1_c = nn.Linear(self.d_l * 3, self.d_l * 3)
        self.proj2_c = nn.Linear(self.d_l * 3, self.d_l * 3)
        self.out_layer_c = nn.Linear(self.d_l * 3, output_dim)


        # 4 Multimodal Crossmodal Attentions
        use_selective_null = self.lfa_mode in {'selective_null', 'selective_null_gain'}
        self.trans_l_with_a = self.get_network(
            self_type='la', layers=self.layers, selective_null=use_selective_null
        )
        self.trans_l_with_v = self.get_network(
            self_type='lv', layers=self.layers, selective_null=use_selective_null
        )
        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')
        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=self.layers)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)


        # 5. fc layers for shared features
        self.proj1_l_low = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
        self.proj2_l_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1))
        self.out_layer_l_low = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), output_dim)
        self.proj1_v_low = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.proj2_v_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1))
        self.out_layer_v_low = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), output_dim)
        self.proj1_a_low = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)
        self.proj2_a_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1))
        self.out_layer_a_low = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), output_dim)

        
        # 6. fc layers for specific features
        self.proj1_l_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_l_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_l_high = nn.Linear(combined_dim_high, output_dim)
        self.proj1_v_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_v_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_v_high = nn.Linear(combined_dim_high, output_dim)
        self.proj1_a_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_a_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_a_high = nn.Linear(combined_dim_high, output_dim)
        
        # 7. project for fusion
        self.projector_l = nn.Linear(self.d_l, self.d_l)
        self.projector_v = nn.Linear(self.d_v, self.d_v)
        self.projector_a = nn.Linear(self.d_a, self.d_a)
        self.projector_c = nn.Linear(3 * self.d_l, 3 * self.d_l)

        # 7b. per-sample scalar gates on the audio / video fusion branches
        self.gate_a = None
        self.gate_v = None
        self.gate_a_global = None
        self.gate_v_global = None
        if self.fusion_gate_mode == 'sample':
            self.gate_a = nn.Linear(2 * self.d_l, 1)
            self.gate_v = nn.Linear(2 * self.d_l, 1)
            for gate in (self.gate_a, self.gate_v):
                nn.init.zeros_(gate.weight)
                nn.init.constant_(gate.bias, gate_bias)
        elif self.fusion_gate_mode == 'global':
            self.gate_a_global = nn.Parameter(torch.tensor(gate_bias))
            self.gate_v_global = nn.Parameter(torch.tensor(gate_bias))

        # 7c. attention-pool scorers for the selected LFA private paths
        self.lfa_attn = nn.ModuleDict()
        if self.lfa_pool_mode == 'attn':
            if 'text' in self.lfa_pool_paths:
                self.lfa_attn['text'] = self._make_attn_pool(self.d_l)
            if 'audio' in self.lfa_pool_paths:
                self.lfa_attn['audio'] = self._make_attn_pool(self.d_a)
            if 'video' in self.lfa_pool_paths:
                self.lfa_attn['video'] = self._make_attn_pool(self.d_v)

        # 8. final project
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
        
    def get_network(self, self_type='l', layers=-1, selective_null=False):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v        
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask,
                                  selective_null_attn=selective_null)

    def set_null_gate_open(self, is_open, closed_bias=-1e4):
        """Open/close the selective-null gate on the A->L and V->L encoders."""
        if self.lfa_mode not in {'selective_null', 'selective_null_gain'}:
            return
        self.trans_l_with_a.set_null_gate_open(is_open, closed_bias)
        self.trans_l_with_v.set_null_gate_open(is_open, closed_bias)


    def _high_head(self, last_h, proj1, proj2, out_layer):
        """Shared high-level head: returns (logits, hs_proj) before fusion."""
        hs_proj = proj2(
            F.dropout(F.relu(proj1(last_h), inplace=True),
                      p=self.output_dropout, training=self.training))
        hs_proj += last_h
        return out_layer(hs_proj), hs_proj

    @staticmethod
    def _make_attn_pool(dim):
        """Additive attention scorer: e_t = v^T tanh(W h_t + b)."""
        return nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1))

    def _build_lfa_mask(self, text):
        """Build a [B, T'] validity mask on the private LFA time axis.

        All three LFA paths are indexed by text positions (query = s_l, length
        len_l - kernel + 1). Valid tokens are taken from the BERT input mask
        when present, otherwise inferred from non-zero rows, then aligned to the
        conv1d output length with a stride-1 max-pool (any real token in the
        receptive field marks the position valid).
        """
        if self.use_bert and text.dim() == 3 and text.size(1) >= 2:
            token_mask = text[:, 1, :].float()  # (B, len_l)
        else:
            token_mask = (text.abs().sum(dim=-1) > 0).float()
        if token_mask.dim() == 1:
            token_mask = token_mask.unsqueeze(0)
        pooled = F.max_pool1d(
            token_mask.unsqueeze(1),
            kernel_size=self.conv1d_kernel_size_l,
            stride=1,
        ).squeeze(1)
        return pooled > 0.5  # (B, T') bool

    def _lfa_pool(self, h, path_key, mask):
        """Summarize an LFA path over time: last (original) / mean / attn.

        `h` is (T, B, D); `mask` is (B, T) or None. Only the selected paths
        deviate from the original last-position read.
        """
        if self.lfa_pool_mode == 'last' or path_key not in self.lfa_pool_paths:
            return h[-1]

        hm = h.transpose(0, 1)  # (B, T, D)
        if mask is None:
            valid = torch.ones(hm.size(0), hm.size(1), dtype=torch.bool, device=hm.device)
        else:
            valid = mask
            if valid.size(1) != hm.size(1):
                raise ValueError(
                    f"LFA mask length {valid.size(1)} != sequence length {hm.size(1)} "
                    f"for path {path_key!r}."
                )
        valid_f = valid.unsqueeze(-1).to(hm.dtype)  # (B, T, 1)

        if self.lfa_pool_mode == 'mean':
            denom = valid_f.sum(dim=1).clamp_min(1.0)  # (B, 1)
            pooled = (hm * valid_f).sum(dim=1) / denom
            has_valid = valid.any(dim=1, keepdim=True)
            return torch.where(has_valid, pooled, h[-1])

        # attn
        scores = self.lfa_attn[path_key](hm).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(~valid, float('-inf'))
        has_valid = valid.any(dim=1, keepdim=True)
        # If a row has no valid position, fall back to the last position.
        scores = torch.where(has_valid, scores, torch.zeros_like(scores))
        weights = torch.softmax(scores, dim=1)  # (B, T)
        pooled = (weights.unsqueeze(-1) * hm).sum(dim=1)
        return torch.where(has_valid, pooled, h[-1])

    def _gate_audio(self, h_l, h_a):
        """Scalar audio gate with shape [B, 1], or None when gates are disabled."""
        if self.fusion_gate_mode == 'none':
            return None
        if self.fusion_gate_mode == 'fixed':
            return torch.full((h_l.size(0), 1), self.fusion_gate_init,
                              device=h_l.device, dtype=h_l.dtype)
        if self.fusion_gate_mode == 'global':
            return torch.sigmoid(self.gate_a_global).expand(h_l.size(0), 1)
        return torch.sigmoid(self.gate_a(torch.cat([h_l, h_a], dim=1)))

    def _gate_video(self, h_l, h_v):
        """Scalar video gate with shape [B, 1], or None when gates are disabled."""
        if self.fusion_gate_mode == 'none':
            return None
        if self.fusion_gate_mode == 'fixed':
            return torch.full((h_l.size(0), 1), self.fusion_gate_init,
                              device=h_l.device, dtype=h_l.dtype)
        if self.fusion_gate_mode == 'global':
            return torch.sigmoid(self.gate_v_global).expand(h_l.size(0), 1)
        return torch.sigmoid(self.gate_v(torch.cat([h_l, h_v], dim=1)))

    def _fuse_and_predict(self, feat_l, feat_v, feat_a, feat_c):
        """Fuse the four sigmoid-projected branches and predict sentiment."""
        last_hs = torch.cat([feat_l, feat_v, feat_a, feat_c], dim=1)
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs), inplace=True),
                      p=self.output_dropout, training=self.training))
        last_hs_proj += last_hs
        return self.out_layer(last_hs_proj)

    def forward(self, text, audio, video, return_counterfactual=False):
        #extraction
        lfa_mask = None
        if self.lfa_pool_mode != 'last':
            lfa_mask = self._build_lfa_mask(text)
        if self.use_bert:
            text = self.text_model(text)
        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        x_a = audio.transpose(1, 2)
        x_v = video.transpose(1, 2)
        

        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l) 
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a) 
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        
        proj_x_l = proj_x_l.permute(2, 0, 1)   
        proj_x_v = proj_x_v .permute(2, 0, 1)  
        proj_x_a = proj_x_a.permute(2, 0, 1)

        #disentanglement
        s_l = self.encoder_s_l(proj_x_l)    
        s_v = self.encoder_s_v(proj_x_v)
        s_a = self.encoder_s_a(proj_x_a)

        c_l = self.encoder_c(proj_x_l)
        c_v = self.encoder_c(proj_x_v)
        c_a = self.encoder_c(proj_x_a)


        s_l = s_l.permute(1, 2, 0)   
        s_v = s_v.permute(1, 2, 0)
        s_a = s_a.permute(1, 2, 0)

        c_l = c_l.permute(1, 2, 0)
        c_v = c_v.permute(1, 2, 0)
        c_a = c_a.permute(1, 2, 0)
        c_list = [c_l, c_v, c_a]


        c_l_sim = self.align_c_l(c_l.contiguous().view(x_l.size(0), -1))
        c_v_sim = self.align_c_v(c_v.contiguous().view(x_l.size(0), -1))
        c_a_sim = self.align_c_a(c_a.contiguous().view(x_l.size(0), -1))
        
        recon_l = self.decoder_l(torch.cat([s_l, c_list[0]], dim=1))
        recon_v = self.decoder_v(torch.cat([s_v, c_list[1]], dim=1))
        recon_a = self.decoder_a(torch.cat([s_a, c_list[2]], dim=1))

        recon_l = recon_l.permute(2, 0, 1)  
        recon_v = recon_v.permute(2, 0, 1)   
        recon_a = recon_a.permute(2, 0, 1)

        s_l_r = self.encoder_s_l(recon_l).permute(1, 2, 0)                                                                                             
        s_v_r = self.encoder_s_v(recon_v).permute(1, 2, 0)
        s_a_r = self.encoder_s_a(recon_a).permute(1, 2, 0)
        
        s_l = s_l.permute(2, 0, 1)  
        s_v = s_v.permute(2, 0, 1)   
        s_a = s_a.permute(2, 0, 1)

        c_l = c_l.permute(2, 0, 1)
        c_v = c_v.permute(2, 0, 1)
        c_a = c_a.permute(2, 0, 1)
       
       #enhancement
        hs_l_low = c_l.transpose(0, 1).contiguous().view(x_l.size(0), -1)  
        repr_l_low = self.proj1_l_low(hs_l_low)                            
        hs_proj_l_low = self.proj2_l_low(
            F.dropout(F.relu(repr_l_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_l_low += hs_l_low         
        logits_l_low = self.out_layer_l_low(hs_proj_l_low)

        hs_v_low = c_v.transpose(0, 1).contiguous().view(x_v.size(0), -1)
        repr_v_low = self.proj1_v_low(hs_v_low)
        hs_proj_v_low = self.proj2_v_low(
            F.dropout(F.relu(repr_v_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_v_low += hs_v_low
        logits_v_low = self.out_layer_v_low(hs_proj_v_low)

        hs_a_low = c_a.transpose(0, 1).contiguous().view(x_a.size(0), -1)
        repr_a_low = self.proj1_a_low(hs_a_low)
        hs_proj_a_low = self.proj2_a_low(
            F.dropout(F.relu(repr_a_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_a_low += hs_a_low
        logits_a_low = self.out_layer_a_low(hs_proj_a_low)

        
        c_l_att = self.self_attentions_c_l(c_l)  
        if type(c_l_att) == tuple:
            c_l_att = c_l_att[0]
        c_l_att = c_l_att[-1]

        c_v_att = self.self_attentions_c_v(c_v)
        if type(c_v_att) == tuple:
            c_v_att = c_v_att[0]
        c_v_att = c_v_att[-1]

        c_a_att = self.self_attentions_c_a(c_a)
        if type(c_a_att) == tuple:
            c_a_att = c_a_att[0]
        c_a_att = c_a_att[-1]

        c_fusion = torch.cat([c_l_att, c_v_att, c_a_att], dim=1)   

        c_proj = self.proj2_c(
            F.dropout(F.relu(self.proj1_c(c_fusion), inplace=True), p=self.output_dropout,
                      training=self.training))
        c_proj += c_fusion                       
        logits_c = self.out_layer_c(c_proj)     
        
        # LFA
        # L --> L
        h_ls = s_l
        h_ls = self.trans_l_mem(h_ls)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = last_hs = self._lfa_pool(h_ls, 'text', lfa_mask)

        # A --> L
        h_l_with_as = self.trans_l_with_a(s_l, s_a, s_a)
        null_attn_a = None
        if self.lfa_mode in {'selective_null', 'selective_null_gain'}:
            null_attn_a = self.trans_l_with_a.layers[-1].self_attn.last_null_attention
        h_as = h_l_with_as
        h_as = self.trans_a_mem(h_as)
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = last_hs = self._lfa_pool(h_as, 'audio', lfa_mask)

        # V --> L
        h_l_with_vs = self.trans_l_with_v(s_l, s_v, s_v)
        null_attn_v = None
        if self.lfa_mode in {'selective_null', 'selective_null_gain'}:
            null_attn_v = self.trans_l_with_v.layers[-1].self_attn.last_null_attention
        h_vs = h_l_with_vs
        h_vs = self.trans_v_mem(h_vs)
        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        last_h_v = last_hs = self._lfa_pool(h_vs, 'video', lfa_mask)


        logits_l_high, hs_proj_l_high = self._high_head(
            last_h_l, self.proj1_l_high, self.proj2_l_high, self.out_layer_l_high
        )
        logits_v_high, hs_proj_v_high = self._high_head(
            last_h_v, self.proj1_v_high, self.proj2_v_high, self.out_layer_v_high
        )
        logits_a_high, hs_proj_a_high = self._high_head(
            last_h_a, self.proj1_a_high, self.proj2_a_high, self.out_layer_a_high
        )

        #fusion
        fuse_l = torch.sigmoid(self.projector_l(hs_proj_l_high))
        fuse_v = torch.sigmoid(self.projector_v(hs_proj_v_high))
        fuse_a = torch.sigmoid(self.projector_a(hs_proj_a_high))
        fuse_c = torch.sigmoid(self.projector_c(c_fusion))

        gate_a = self._gate_audio(hs_proj_l_high, hs_proj_a_high)
        gate_v = self._gate_video(hs_proj_l_high, hs_proj_v_high)
        if gate_a is not None:
            fuse_v = gate_v * fuse_v
            fuse_a = gate_a * fuse_a

        output = self._fuse_and_predict(fuse_l, fuse_v, fuse_a, fuse_c)

        #counterfactual branches (training-only, no extra inference cost)
        pred_a_null = None
        pred_v_null = None
        if return_counterfactual:
            if self.lfa_mode != 'selective_null_gain':
                raise ValueError(
                    "return_counterfactual requires lfa_mode='selective_null_gain'"
                )
            # Force only the A->L path to read Null; shared branch unchanged.
            h_as_null = self.trans_l_with_a(s_l, s_a, s_a, force_null=True)
            h_as_null = self.trans_a_mem(h_as_null)
            if type(h_as_null) == tuple:
                h_as_null = h_as_null[0]
            last_h_a_null = self._lfa_pool(h_as_null, 'audio', lfa_mask)
            _, hs_proj_a_null = self._high_head(
                last_h_a_null, self.proj1_a_high, self.proj2_a_high, self.out_layer_a_high
            )
            fuse_a_null = torch.sigmoid(self.projector_a(hs_proj_a_null))
            gate_a_null = self._gate_audio(hs_proj_l_high, hs_proj_a_null)
            if gate_a_null is not None:
                fuse_a_null = gate_a_null * fuse_a_null
            pred_a_null = self._fuse_and_predict(fuse_l, fuse_v, fuse_a_null, fuse_c)

            # Force only the V->L path to read Null; shared branch unchanged.
            h_vs_null = self.trans_l_with_v(s_l, s_v, s_v, force_null=True)
            h_vs_null = self.trans_v_mem(h_vs_null)
            if type(h_vs_null) == tuple:
                h_vs_null = h_vs_null[0]
            last_h_v_null = self._lfa_pool(h_vs_null, 'video', lfa_mask)
            _, hs_proj_v_null = self._high_head(
                last_h_v_null, self.proj1_v_high, self.proj2_v_high, self.out_layer_v_high
            )
            fuse_v_null = torch.sigmoid(self.projector_v(hs_proj_v_null))
            gate_v_null = self._gate_video(hs_proj_l_high, hs_proj_v_null)
            if gate_v_null is not None:
                fuse_v_null = gate_v_null * fuse_v_null
            pred_v_null = self._fuse_and_predict(fuse_l, fuse_v_null, fuse_a, fuse_c)

        res = {
            'origin_l': proj_x_l,
            'origin_v': proj_x_v,
            'origin_a': proj_x_a,
            's_l': s_l,        
            's_v': s_v,
            's_a': s_a,
            'c_l': c_l,
            'c_v': c_v,
            'c_a': c_a,
            's_l_r': s_l_r,
            's_v_r': s_v_r,
            's_a_r': s_a_r,
            'recon_l': recon_l,
            'recon_v': recon_v,
            'recon_a': recon_a,
            'c_l_sim': c_l_sim,
            'c_v_sim': c_v_sim,
            'c_a_sim': c_a_sim,
            'logits_l_hetero': logits_l_high,           
            'logits_v_hetero': logits_v_high, 
            'logits_a_hetero': logits_a_high,
            'logits_c': logits_c,
            'output_logit': output,
            'null_attn_a': null_attn_a,
            'null_attn_v': null_attn_v,
            'output_logit_a_null': pred_a_null,
            'output_logit_v_null': pred_v_null,
            'fusion_gate_a': gate_a.detach() if gate_a is not None else None,
            'fusion_gate_v': gate_v.detach() if gate_v is not None else None,
        }
        return res
