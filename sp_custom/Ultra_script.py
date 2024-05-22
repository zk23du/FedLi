import subprocess

def run_in_tmux(gpu_id, beta):
    session_name = f"session_gpu_{gpu_id}"
    commands = [
        f"tmux new-session -d -s {session_name}",
        f"tmux send-keys -t {session_name} 'conda activate cycp' C-m",
        f"tmux send-keys -t {session_name} 'cd Neurips2024/sp_decentralized_mnist_lr_example/' C-m",
        f"tmux send-keys -t {session_name} 'python VaryParamsAndRun.py --gpu_id {gpu_id} --beta {beta}' C-m"
    ]

    for cmd in commands:
        subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    gpu_ids = [0, 1, 2, 3]
    betas = [1.0, 100.0, 0.5, 0.1]
    for gpu_id, beta in zip(gpu_ids, betas):
        run_in_tmux(gpu_id, beta)
