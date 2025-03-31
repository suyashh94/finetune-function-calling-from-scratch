import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch


from dataset_function_calling import get_datasets
from encoder import Encoder
from model import GPT, GPTConfig
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class Config:
    def __init__(self, pretrained_dir):
        self.init_from = "pretrained"
        self.out_dir = "./outputs/fc_fully_finetuned"
        self.pretrained_dir = pretrained_dir
        self.device = "cuda"
        self.eval_interval = 500
        self.sample_interval = 50
        self.log_interval = 1
        self.eval_iters = 2
        self.eval_only = False
        self.always_save_checkpoint = True
        self.gradient_accumulation_steps = 8
        self.batch_size = 64
        self.learning_rate = 5e-6
        self.max_iters = 60000
        self.weight_decay = 1e-1
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.grad_clip = 1.0
        self.decay_lr = True
        self.warmup_iters = 2000
        self.lr_decay_iters = int(0.9 * 60000)
        self.min_lr = 1e-6
        self.backend = "nccl"
        self.dtype = (
            "bfloat16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "float16"
        )
        self.compile = True
        self.block_size = None
        self.finetuning_dropout = 0.2
        self.loss_on_prompt = False


class CustomDataset(Dataset):
    def __init__(self, dst):
        self.dataset = dst

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        if "output_ids" not in data:
            raise ValueError(f"Missing 'output_ids' in dataset at index {idx} with data: {data}")
        ids = torch.tensor(data["output_ids"], dtype=torch.int64)
        return {
            "input_ids": ids,
            "padding_mask": torch.tensor(data["pad_mask"], dtype=torch.int64),
            "prompt_mask": torch.tensor(data["prompt_mask"], dtype=torch.int64),
        }


class GPTTrainer:
    def __init__(self, config, data_dir=None):
        self.config = config
        self.data_dir = data_dir
        print("data dir: ", self.data_dir)
        self.enc = Encoder().encoder
        self.setup_distributed()
        self.setup_model()
        self.setup_data()

        self.iter_num = 0
        self.best_val_loss = float("inf")

        self.tokens_per_iter = (
            self.config.gradient_accumulation_steps
            * self.ddp_world_size
            * self.config.batch_size
            * self.config.block_size
        )
        print(f"tokens per iteration will be: {self.tokens_per_iter:,}")

    def setup_distributed(self):
        self.ddp = int(os.environ.get("RANK", -1)) != -1
        if self.ddp:
            init_process_group(backend=self.config.backend)
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            self.device = f"cuda:{self.ddp_local_rank}"
            torch.cuda.set_device(self.device)
            self.master_process = self.ddp_rank == 0
            self.seed_offset = self.ddp_rank
            assert self.config.gradient_accumulation_steps % self.ddp_world_size == 0
            self.config.gradient_accumulation_steps //= self.ddp_world_size
        else:
            self.master_process = True
            self.seed_offset = 0
            self.ddp_world_size = 1
            self.device = self.config.device

        if self.master_process:
            os.makedirs(self.config.out_dir, exist_ok=True)

        torch.manual_seed(1337 + self.seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.device_type = "cuda" if "cuda" in self.device else "cpu"
        self.ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[self.config.dtype]
        self.ctx = (
            nullcontext()
            if self.device_type == "cpu"
            else torch.amp.autocast(device_type=self.device_type, dtype=self.ptdtype)  # type: ignore
        )

    def setup_model(self):
        if self.config.init_from == "scratch":
            print("Initializing a new model from scratch")
            gptconf = GPTConfig(**self.config.model_args)
            self.model = GPT(gptconf)

        elif self.config.init_from == "resume":
            print(f"Resuming training from {self.config.out_dir}")
            ckpt_path = os.path.join(self.config.out_dir, "ckpt.pt")
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            try:
                gptconf = GPTConfig(**checkpoint["model_args"])
            except:
                gptconf = GPTConfig(
                    block_size=1024, vocab_size=50304, n_layer = 12, n_head = 12, n_embd = 768, bias=False
                )
            self.model = GPT(gptconf)
            self.model.load_state_dict(checkpoint["model"])
            self.iter_num = checkpoint["iter_num"]
            self.best_val_loss = checkpoint["best_val_loss"]
        elif self.config.init_from == "pretrained":
            print(f"Initializing from pretrained model in {self.config.pretrained_dir}")
            ckpt_path = self.config.pretrained_dir
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            gptconf = GPTConfig(**checkpoint["model_args"])
            gptconf.dropout = self.config.finetuning_dropout  # adjust as needed
            self.model = GPT(gptconf)
            self.gptconf = gptconf
            state_dict = checkpoint["model"]
            unwanted_prefix = "_orig_mod."
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
            self.model.load_state_dict(state_dict)

        if self.config.block_size is None:
            self.config.block_size = gptconf.block_size

        if self.config.block_size < self.model.config.block_size:
            self.model.crop_block_size(self.config.block_size)
            gptconf.block_size = self.config.block_size

        self.model.to(self.device)

        self.optimizer = self.model.configure_optimizers(
            self.config.weight_decay,
            self.config.learning_rate,
            (self.config.beta1, self.config.beta2),
            self.device_type,
        )

        if self.config.compile:
            print("compiling the model... (takes a ~minute)")
            self.model = torch.compile(self.model)

        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.ddp_local_rank])

        if self.config.init_from == "resume":
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.config.dtype == "float16"))

    def setup_data(self):
        train_dataset, val_dataset = get_datasets(self.data_dir, self.enc)
        # train_dataset, val_dataset = get_datasets()
        self.train_dataset = CustomDataset(train_dataset)
        self.val_dataset = CustomDataset(val_dataset)

        if self.master_process:
            print(f"Train dataset length: {len(self.train_dataset)}")
            print(f"Val dataset length: {len(self.val_dataset)}")

        self.train_loader = self.get_dataloader(self.train_dataset, is_train=True)
        self.val_loader = self.get_dataloader(self.val_dataset, is_train=False)

    def get_dataloader(self, dataset, is_train=True):
        if self.ddp:
            sampler = DistributedSampler(
                dataset, num_replicas=self.ddp_world_size, rank=self.ddp_rank, shuffle=is_train
            )
        else:
            sampler = None

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=(sampler is None) and is_train,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=is_train,
        )

    def get_batch(self, split):
        loader = self.train_loader if split == "train" else self.val_loader
        data = next(iter(loader))
        x = data["input_ids"][:, :-1].to(self.device)
        y = data["input_ids"][:, 1:].to(self.device)
        y[data["padding_mask"] == 1] = -1
        if not self.config.loss_on_prompt:
            y[data["prompt_mask"] == 1] = -1
        else:
            y[data["prompt_mask"] == 1] = 1

        return x, y

    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split, loader in [("train", self.train_loader), ("val", self.val_loader)]:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                batch = next(iter(loader))
                input_ids = batch["input_ids"]
                padding_mask = batch["padding_mask"]
                prompt_mask = batch["prompt_mask"]
                X = input_ids[:, :-1].to(self.device)
                Y = input_ids[:, 1:].to(self.device)
                Y[padding_mask == 1] = -1
                Y[prompt_mask == 1] = -1
                with self.ctx:
                    logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def get_lr(self, it):
        if it < self.config.warmup_iters:
            return self.config.learning_rate * it / self.config.warmup_iters
        if it > self.config.lr_decay_iters:
            return self.config.min_lr
        decay_ratio = (it - self.config.warmup_iters) / (
            self.config.lr_decay_iters - self.config.warmup_iters
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)

    def sample_inference(self, iter_num):
        if not self.master_process:
            return

        self.model.eval()
        with torch.no_grad():
            # np.random.seed(1337 + self.seed_offset)
            for split in ["train", "val"]:
                dataset = self.train_dataset if split == "train" else self.val_dataset
                sample = dataset[np.random.randint(0, len(dataset))]
                ids = sample["input_ids"]
                x = torch.clone(ids[:-1])
                y = torch.clone(ids[1:])

                x_eop_idx = torch.where(x == self.enc.eop_token)[0].item()  # type: ignore
                y_first_pad_idx = torch.where(y == self.enc.pad_token)[0][0].item()  # type: ignore
                y = y[x_eop_idx:y_first_pad_idx]

                x = x[None, ...].to(self.device)
                y = y[None, ...].to(self.device)

                with self.ctx:
                    input_string = self.enc.decode(x[0, : x_eop_idx + 1].tolist())
                    if self.ddp:
                        res = self.model.module.generate_answer_for_question(
                            x, 800, 0.8, 200, isTraining=True, sampleMax=False
                        )
                    else:
                        res = self.model.generate_answer_for_question(
                            x, 800, 0.8, 200, isTraining=True, sampleMax=False
                        )
                    dec_string = self.enc.decode(res[0].tolist())
                    actual_string = self.enc.decode(y[0, :].tolist())

                if self.master_process:
                    fname = f"./outputs/samples/{split}_sample.txt"
                    base_dir = os.path.dirname(fname)
                    os.makedirs(base_dir, exist_ok=True)
                    mode = "w" if iter_num == 0 and not os.path.exists(fname) else "a"
                    with open(fname, mode) as f:
                        f.write(f" Iteration: {iter_num} \n")
                        f.write("---------------------------------- \n")
                        f.write(f"Input: {input_string}\n")
                        f.write("---------------------------------- \n")
                        f.write(f"Actual: {actual_string}\n")
                        f.write("---------------------------------- \n")
                        f.write(f"Predicted: {dec_string}\n")
                        f.write("---------------------------------- \n" * 3)
        self.model.train()

    def train(self):
        t0 = time.time()

        raw_model = self.model.module if self.ddp else self.model
        running_mfu = -1.0

        while True:
            if self.iter_num >= self.config.max_iters:
                break

            # Set epoch for samplers
            if self.ddp:
                self.train_loader.sampler.set_epoch(self.iter_num // len(self.train_loader))  # type: ignore
                self.val_loader.sampler.set_epoch(self.iter_num // len(self.train_loader))  # type: ignore

            for batch in self.train_loader:
                # Correctly unpack the batch
                input_ids = batch["input_ids"]
                padding_mask = batch["padding_mask"]
                prompt_mask = batch["prompt_mask"]

                X = input_ids[:, :-1].to(self.device)
                Y = input_ids[:, 1:].to(self.device)
                Y[padding_mask == 1] = -1
                if hasattr(self.config, "loss_on_prompt") and self.config.loss_on_prompt:
                    Y[prompt_mask == 1] = 1
                else:
                    Y[prompt_mask == 1] = -1

                lr = (
                    self.get_lr(self.iter_num)
                    if self.config.decay_lr
                    else self.config.learning_rate
                )
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

                if self.iter_num % self.config.eval_interval == 0 and self.master_process:
                    losses = self.estimate_loss()
                    print(
                        f"step {self.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                    )

                    if losses["val"] < self.best_val_loss or self.config.always_save_checkpoint:
                        self.best_val_loss = losses["val"]
                        if self.iter_num > 0:
                            checkpoint = {
                                "model": raw_model.state_dict(),
                                "optimizer": self.optimizer.state_dict(),
                                "model_args": self.gptconf.__dict__,
                                "iter_num": self.iter_num,
                                "best_val_loss": self.best_val_loss,
                            }
                            print(f"saving checkpoint to {self.config.out_dir}")
                            torch.save(checkpoint, os.path.join(self.config.out_dir, "ckpt.pt"))

                if self.config.eval_only:
                    return

                if self.iter_num % self.config.sample_interval == 0 and self.master_process:
                    self.sample_inference(self.iter_num)

                for micro_step in range(self.config.gradient_accumulation_steps):
                    if self.ddp:
                        self.model.require_backward_grad_sync = (
                            micro_step == self.config.gradient_accumulation_steps - 1
                        )
                    with self.ctx:
                        logits, loss = self.model(X, Y)
                        loss = loss / self.config.gradient_accumulation_steps
                    self.scaler.scale(loss).backward()

                if self.config.grad_clip != 0.0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                if self.iter_num % self.config.log_interval == 0 and self.master_process:
                    lossf = loss.item() * self.config.gradient_accumulation_steps
                    if self.iter_num % self.config.log_interval == 0:
                        mfu = raw_model.estimate_mfu(
                            self.config.batch_size * self.config.gradient_accumulation_steps, dt
                        )
                        running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                    print(
                        f"iter {self.iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
                    )
                    

                self.iter_num += 1

                if self.iter_num >= self.config.max_iters:
                    break

        if self.ddp:
            destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_dir", type=str, default=None)
    parser.add_argument("--fc_data_dir", type=str, default=None)

    args = parser.parse_args()
    if args.pretrained_dir is None:
        raise ValueError("Please provide the path to the pretrained model directory")

    # Create configuration
    config = Config(pretrained_dir=args.pretrained_dir)
    os.makedirs(config.out_dir, exist_ok=True)
    # Initialize trainer
    trainer = GPTTrainer(config=config, data_dir=args.fc_data_dir)

    # Start training
    trainer.train()
