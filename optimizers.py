import torch

def create_optimizers(models, config):
    lr = config['lr']
    weight_decay = config['weight_decay']
    gradient_accumulation_steps = config['gradient_accumulation_steps']
    train_batch_size = config['train_batch_size']


    if self.cfg.optim.scale_lr:
        self.cfg.optim.learning_rate = (self.cfg.optim.learning_rate *
                                        self.cfg.optim.gradient_accumulation_steps *
                                        self.cfg.optim.train_batch_size *
                                        self.accelerator.num_processes)
    optimizer = torch.optim.AdamW(
        self.text_encoder.text_model.embeddings.mapper.parameters(),  # only optimize the embeddings
        lr=self.cfg.optim.learning_rate,
        betas=(self.cfg.optim.adam_beta1, self.cfg.optim.adam_beta2),
        weight_decay=self.cfg.optim.adam_weight_decay,
        eps=self.cfg.optim.adam_epsilon,
    )
    return optimizer


def _init_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler:
    lr_scheduler = get_scheduler(
        self.cfg.optim.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=self.cfg.optim.lr_warmup_steps * self.cfg.optim.gradient_accumulation_steps,
        num_training_steps=self.cfg.optim.max_train_steps * self.cfg.optim.gradient_accumulation_steps,
    )
    return lr_scheduler
