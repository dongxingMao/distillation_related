import torch
import torch.nn as nn
import torch.nn.functional as F
from .module.model_diffusion import Diffusion
from ._base import Distiller

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def kd_loss(logits_student_in, logits_teacher_in, temperature, logit_stand):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


class DiffKD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(DiffKD, self).__init__(student, teacher)
        # self.temperature = cfg.DiffKD.TEMPERATURE
        # import pdb
        # pdb.set_trace()
        self.ce_loss_weight = cfg.DiffKD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.DiffKD.LOSS.KD_WEIGHT
        self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND 
        # import pdbjJDYavjI6OHI
        # pdb.set_trace()
        self.diffusion_model = Diffusion(n_class=10, fp_dim=256, feature_dim=256,
                                device='cuda', ddim_num_steps=10)       # 先hard code
    def forward_train(self, image, target, **kwargs):
        # import pdb
        # pdb.set_trace()
        device = self.diffusion_model.device
        logits_student, feats_student = self.student(image) # [64,1], [64, 256]
        with torch.no_grad():
            logits_teacher, feats_teacher = self.teacher(image)

        # losses
        n = logits_student.shape[0]
        t = torch.randint(low=0, high=self.diffusion_model.num_timesteps, size=(n // 2 + 1,)).to(device)
        t = torch.cat([t, self.diffusion_model.num_timesteps - 1 - t], dim=0)[:n]
        
        # pdb.set_trace()
        output, e = self.diffusion_model.forward_t(feats_teacher["pooled_feat"], feats_student["pooled_feat"], t)
        
        
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        
        # for error loss        
        loss_kd = self.kd_loss_weight * F.mse_loss(
            output, e)
        
        # loss_kd = self.kd_loss_weight * kd_loss(
        #     logits_student, logits_teacher, self.temperature, self.logit_stand
        # )
        
        ## according form PKT
        # loss_feat = self.feat_loss_weight * F.mse_loss(
        #     f_s, feature_teacher["feats"][self.hint_layer]
        # )
        
        # loss_kd = self.kd_loss_weight * F.mse_loss(
        #     feats_student["pooled_feat"], feats_teacher["pooled_feat"])
        
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
