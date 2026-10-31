import logging
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from ..utils import MetricsTop, dict_to_str
from .shared_alignment import SharedAlignmentLoss
from .gain_supervision import (
    EMATeacher, GainSupervision, gain_weight_for_epoch, per_sample_smooth_l1,
)


logger = logging.getLogger('MMSA')

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse

class DLF():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss()           
        self.cosine = nn.CosineEmbeddingLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.MSE = MSE()
        self.shared_alignment = SharedAlignmentLoss(
            mode=getattr(args, 'shared_loss_mode', 'original'),
            atp_weight=getattr(args, 'atp_weight', 0.01),
            sent_weight=getattr(args, 'sent_weight', 0.01),
            sent_sigma=getattr(args, 'sent_sigma', 1.0),
            sent_tau=getattr(args, 'sent_tau', 1.0),
        )
        logger.info(
            "Shared loss: mode=%s, atp_weight=%g, sent_weight=%g, sigma=%g, tau=%g",
            self.shared_alignment.mode, self.shared_alignment.atp_weight,
            self.shared_alignment.sent_weight, self.shared_alignment.sent_sigma,
            self.shared_alignment.sent_tau,
        )
        self.lfa_mode = getattr(args, 'lfa_mode', 'original')
        self.use_gain = self.lfa_mode == 'selective_null_gain'
        self.use_null = self.lfa_mode in {'selective_null', 'selective_null_gain'}
        # Null gate is held closed for this many epochs, then released.
        self.null_warmup_epochs = getattr(args, 'null_warmup_epochs', 5)
        self._null_gate_open = None
        self.gain_criterion = GainSupervision(
            margin=getattr(args, 'gain_margin', 0.01),
            temperature=getattr(args, 'gain_temperature', 0.1),
            confidence_scale=getattr(args, 'gain_confidence_scale', 0.1),
        )
        self.gain_max_weight = getattr(args, 'gain_weight', 0.05)
        self.gain_warmup = getattr(args, 'gain_warmup_epochs', 5)
        self.gain_ramp = getattr(args, 'gain_ramp_epochs', 5)
        self.gain_ema_decay = getattr(args, 'gain_ema_decay', 0.995)
        if self.use_gain:
            logger.info(
                "Gain supervision: margin=%g, tau=%g, scale=%g, lambda_max=%g, "
                "warmup=%d, ramp=%d, ema=%g",
                self.gain_criterion.margin, self.gain_criterion.temperature,
                self.gain_criterion.confidence_scale, self.gain_max_weight,
                self.gain_warmup, self.gain_ramp, self.gain_ema_decay,
            )

    def do_train(self, model, dataloader, return_epoch_results=False):

        # 0: DLF model
        params = model[0].parameters() 

        optimizer = optim.Adam(params, lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True, patience=self.args.patience)

        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        net = []
        net_DLF = model[0]
        net.append(net_DLF)
        model = net

        # EMA teacher for stable per-sample gain labels (training-only).
        teacher = None
        if self.use_gain:
            teacher = EMATeacher(model[0], decay=self.gain_ema_decay).to(self.args.device)

        gain_totals = {'gain_a': 0.0, 'gain_v': 0.0, 'loss_gain': 0.0}

        while True:
            epochs += 1
            y_pred, y_true = [], []
            for mod in model:
                mod.train()

            gain_weight = (
                gain_weight_for_epoch(
                    epochs, self.gain_warmup, self.gain_ramp, self.gain_max_weight
                ) if self.use_gain else 0.0
            )

            # Null gate warm-up: closed for the first `null_warmup_epochs`,
            # then released so it can learn normally.
            if self.use_null:
                gate_open = epochs > self.null_warmup_epochs
                if gate_open != self._null_gate_open:
                    model[0].set_null_gate_open(gate_open)
                    self._null_gate_open = gate_open
                    logger.info(
                        "Null gate %s at epoch %d (warmup=%d)",
                        "opened" if gate_open else "closed", epochs,
                        self.null_warmup_epochs,
                    )

            train_loss = 0.0
            shared_loss_totals = dict.fromkeys(('atp', 'sent', 'triplet', 'total'), 0.0)
            left_epochs = self.args.update_epochs
            with tqdm(dataloader['train']) as td:
                for batch_data in td:

                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)


                   
                    output = model[0](text, audio, vision) 

                    # task loss
                    loss_task_all = self.criterion(output['output_logit'], labels)
                    
                    loss_task_l_hetero = self.criterion(output['logits_l_hetero'], labels)  
                    loss_task_v_hetero = self.criterion(output['logits_v_hetero'], labels)
                    loss_task_a_hetero = self.criterion(output['logits_a_hetero'], labels)
                    loss_task_c = self.criterion(output['logits_c'], labels)
                    
                    # total MSA loss L_msa
                    loss_task = 1* (1 * loss_task_all + 1*loss_task_c  + 3 * loss_task_l_hetero + 1*loss_task_v_hetero + 1*loss_task_a_hetero)
                    
                    # reconstruction loss L_r
                    loss_recon_l = self.MSE(output['recon_l'], output['origin_l'])
                    loss_recon_v = self.MSE(output['recon_v'], output['origin_v'])
                    loss_recon_a = self.MSE(output['recon_a'], output['origin_a'])
                    loss_recon = loss_recon_l + loss_recon_v + loss_recon_a

                    # specific loss L_s 
                    loss_sl_slr = self.MSE(output['s_l'].permute(1, 2, 0), output['s_l_r'])
                    loss_sv_slv = self.MSE(output['s_v'].permute(1, 2, 0), output['s_v_r'])
                    loss_sa_sla = self.MSE(output['s_a'].permute(1, 2, 0), output['s_a_r'])
                    loss_s_sr = loss_sl_slr + loss_sv_slv + loss_sa_sla

                    # ort loss L_o
                    if self.args.dataset_name == 'mosi':
                        num = 50
                    elif self.args.dataset_name == 'mosei':
                        num = 10

                    cosine_similarity_s_c_l = self.cosine(output['s_l'].reshape(-1, num), output['c_l'].reshape(-1, num), torch.tensor([-1]).cuda())
                    cosine_similarity_s_c_v = self.cosine(output['s_v'].reshape(-1, num), output['c_v'].reshape(-1, num), torch.tensor([-1]).cuda())
                    cosine_similarity_s_c_a = self.cosine(output['s_a'].reshape(-1, num), output['c_a'].reshape(-1, num), torch.tensor([-1]).cuda())
                    
                    loss_ort = cosine_similarity_s_c_l + cosine_similarity_s_c_v + cosine_similarity_s_c_a

                    # Shared-space constraints on the existing projected vectors.
                    shared_losses = self.shared_alignment(
                        output['c_l_sim'], output['c_a_sim'], output['c_v_sim'], labels
                    )

                    # New weights are final coefficients, without an outer 0.1.
                    combined_loss = (
                        loss_task + 0.1 * loss_s_sr + 0.1 * loss_recon
                        + 0.01 * loss_ort + shared_losses['total']
                    )

                    # Gain supervision: EMA teacher labels how much each
                    # cross-modal path helps; student's gate is trained toward it.
                    loss_gain = None
                    if self.use_gain:
                        with torch.no_grad():
                            teacher_out = teacher.module(
                                text, audio, vision, return_counterfactual=True
                            )
                            e_full = per_sample_smooth_l1(
                                teacher_out['output_logit'], labels
                            )
                            e_a_null = per_sample_smooth_l1(
                                teacher_out['output_logit_a_null'], labels
                            )
                            e_v_null = per_sample_smooth_l1(
                                teacher_out['output_logit_v_null'], labels
                            )
                            gain_a = (e_a_null - e_full).squeeze(-1)
                            gain_v = (e_v_null - e_full).squeeze(-1)
                        gain_losses = self.gain_criterion(
                            output['null_attn_a'].mean(dim=-1),
                            output['null_attn_v'].mean(dim=-1),
                            gain_a, gain_v,
                        )
                        loss_gain = gain_losses['total']
                        if gain_weight > 0:
                            combined_loss = combined_loss + gain_weight * loss_gain

                    combined_loss.backward()


                    if self.args.grad_clip != -1.0:
                        params = list(model[0].parameters())  
                                 
                        nn.utils.clip_grad_value_(params, self.args.grad_clip)

                    train_loss += combined_loss.item()
                    for name in shared_loss_totals:
                        shared_loss_totals[name] += shared_losses[name].detach().item()

                    if self.use_gain:
                        with torch.no_grad():
                            gain_totals['gain_a'] += gain_a.mean().item()
                            gain_totals['gain_v'] += gain_v.mean().item()
                            gain_totals['loss_gain'] += loss_gain.item()

                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())
                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                        if teacher is not None:
                            teacher.update(model[0])
                if not left_epochs:
                    # update
                    optimizer.step()
                    if teacher is not None:
                        teacher.update(model[0])


            train_loss = train_loss / len(dataloader['train'])
            logger.info(
                "Shared losses (batch means; total is weighted): %s",
                dict_to_str({name: value / len(dataloader['train'])
                             for name, value in shared_loss_totals.items()}),
            )
            if self.use_gain:
                num_batches = len(dataloader['train'])
                logger.info(
                    "Gain supervision (batch means): lambda=%.4f %s",
                    gain_weight,
                    dict_to_str({name: value / num_batches
                                 for name, value in gain_totals.items()}),
                )
                gain_totals = {'gain_a': 0.0, 'gain_v': 0.0, 'loss_gain': 0.0}
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f">> Epoch: {epochs} "
                f"TRAIN -({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> total_loss: {round(train_loss, 4)} "
                f"{dict_to_str(train_results)}"
            )
            # validation
            val_results = self.do_test(model[0], dataloader['valid'], mode="VAL")
            test_results = self.do_test(model[0], dataloader['test'], mode="TEST")
            cur_valid = val_results[self.args.KeyEval]
            scheduler.step(val_results['Loss'])
            # save each epoch model
            epoch_name = (
                f"{self.args.dataset_name}{self.args.variant_suffix}_{epochs}.pth"
            )
            torch.save(model[0].state_dict(), self.args.model_save_dir / epoch_name)
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                torch.save(model[0].state_dict(), self.args.checkpoint_path)

            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):

        model.eval()
        y_pred, y_true = [], []

        eval_loss = 0.0
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)
                    output = model(text, audio, vision)
                    loss = self.criterion(output['output_logit'], labels)
                    eval_loss += loss.item()
                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())

        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)

        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results
