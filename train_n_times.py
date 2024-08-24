import subprocess
import re
import shutil
import os
import argparse

def main():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='Process some parameters.')

    # 添加参数
    parser.add_argument('--total_times', type=int, default=15, help='The first parameter')
    parser.add_argument('--sh_script', type=str, default='./train_one_time.sh', help='The second parameter')
    parser.add_argument('--epochs', type=int, default=500, help='')
    parser.add_argument('--eval_start', type=int, default=100, help='')
    parser.add_argument('--dataset_file', type=str, default='SHA', help='')
    parser.add_argument('--output_dir', type=str, default='pet_model_ntimes', help='')
    parser.add_argument('--result_dir', type=str, default='pet_model_ntimes_save', help='')

    # 解析参数
    args = parser.parse_args()

    # 配置参数
    N = args.total_times  # 循环次数
    sh_script = args.sh_script  # 你的sh脚本路径
    epochs = args.epochs
    eval_start = args.eval_start
    dataset_file = args.dataset_file
    output_dir = f'outputs/{dataset_file}/{args.output_dir}' # 输出目录
    result_dir = f'outputs/{dataset_file}/{args.result_dir}' # 结果目录

    # 确保结果目录存在
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 正则表达式用于提取“best mae”和“best mse”
    mae_pattern = re.compile(r'best mae: (\d+\.\d+)')
    mse_pattern = re.compile(r'best mse: (\d+\.\d+)')

    for i in range(N):
        print(f'Executing iteration {i + 1}...')

        # 执行sh脚本并重定向输出到当前控制台
        process = subprocess.Popen(['bash', sh_script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        # 读取输出并提取“best mae”和“best mse”
        mae = 0
        mse = 0
        for line in process.stdout:
            print(line, end='')
            mae_match = mae_pattern.search(line)
            mse_match = mse_pattern.search(line)
            if mae_match:
                mae = float(mae_match.group(1))
            if mse_match:
                mse = float(mse_match.group(1))

        # 等待脚本执行完毕
        process.wait()

        # 检查输出目录是否存在
        if os.path.exists(output_dir):
            # 复制输出目录到结果目录并重命名
            new_output_dir = os.path.join(result_dir, f'outputs_{i + 1}_{mae:.2f}_{mse:.2f}')
            # if os.path.exists(new_output_dir):
            #     shutil.rmtree(new_output_dir)
            shutil.copytree(output_dir, new_output_dir)
            print(f'Copied {output_dir} to {new_output_dir}')
        else:
            print(f'Output directory {output_dir} does not exist')

        # 输出提取的“best mae”和“best mse”
        if mae and mse:
            print(f'======Iteration {i + 1} - best mae: {mae:.2f}, best mse: {mse:.2f}=========')
        else:
            print(f'======Iteration {i + 1} - best mae or best mse not found in output======')

        print('\n\n')

    print('All iterations completed.')

if __name__ == "__main__":
    main()