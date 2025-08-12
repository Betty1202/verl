# Tested with 2 & 4 GPUs

set -x

# if [ "$#" -lt 2 ]; then
#     echo "Usage: run_qwen_05_peft.sh <nproc_per_node> <save_path> [other_configs...]"
#     exit 1
# fi

# nproc_per_node=$1
# save_path=$2

# # Shift the arguments so $@ refers to the rest
# shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=data/swe-perf \
    data.val_files=data/swe-perf \
    data.prompt_key='text' \
    data.response_key='code' \
    optim.lr=1e-4 \
    data.prompt_dict_keys=['text'] \
    +data.response_dict_keys=['code'] \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=Qwen/Qwen2.5-0.5B-Instruct \
    trainer.default_local_dir=save \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-qwen-2.5-0.5b-instruct \
    trainer.logger=console \
    trainer.total_epochs=3 \
    model.lora_rank=32 \
    model.lora_alpha=16 \
    model.target_modules=all-linear

    # Or you can do this:
    # model.target_modules=[q_proj,v_proj] \
