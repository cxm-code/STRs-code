import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def calculate_4mer_frequency(sequence):
    kmer_count = {}
    total_kmers = 0

    for i in range(len(sequence) - 3):
        kmer = sequence[i:i+4]
        if kmer in kmer_count:
            kmer_count[kmer] += 1
        else:
            kmer_count[kmer] = 1
        total_kmers += 1

    kmer_frequency = {}

    for kmer, count in kmer_count.items():
        frequency = count / total_kmers
        kmer_frequency[kmer] = frequency

    return kmer_frequency

def calculate_5mer_frequency(sequence):
    kmer_count = {}
    total_kmers = 0

    for i in range(len(sequence) - 4):
        kmer = sequence[i:i+5]
        if kmer in kmer_count:
            kmer_count[kmer] += 1
        else:
            kmer_count[kmer] = 1
        total_kmers += 1

    kmer_frequency = {}

    for kmer, count in kmer_count.items():
        frequency = count / total_kmers
        kmer_frequency[kmer] = frequency

    return kmer_frequency

def calculate_6mer_frequency(sequence):
    kmer_count = {}
    total_kmers = 0

    for i in range(len(sequence) - 5):
        kmer = sequence[i:i+6]
        if kmer in kmer_count:
            kmer_count[kmer] += 1
        else:
            kmer_count[kmer] = 1
        total_kmers += 1

    kmer_frequency = {}

    for kmer, count in kmer_count.items():
        frequency = count / total_kmers
        kmer_frequency[kmer] = frequency

    return kmer_frequency

def compare_kmer_frequencies(generated_sequence, natural_sequence):
    generated_freq = calculate_6mer_frequency(generated_sequence)
    natural_freq = calculate_6mer_frequency(natural_sequence)

    x = []
    y = []

    for kmer, freq in generated_freq.items():
        x.append(freq)
        y.append(natural_freq.get(kmer, 0))  # 如果自然序列中没有该4-mer，设频率为0


    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line = slope * np.array(x) + intercept

    # plt.scatter(x, y, color='olive')  # 散点图近端
    # plt.scatter(x, y, color='maroon')  # 散点图远端
    plt.scatter(x, y)  # 散点图全部
    # plt.plot(x, line, color='lightcoral')  # 趋势线

    # plt.scatter(x, y, color='maroon')
    plt.xlabel('Generated',fontsize=20)
    plt.ylabel('Natural',fontsize=20)
    plt.title('All region:5-mer',fontsize=20)
    plt.tick_params(axis='both', labelsize=18)  # 增加坐标轴数值字体大小
    # 设置 y 轴刻度
    # yticks = np.arange(0, 0.03, 0.01)  # 根据需要调整范围和间隔
    # plt.yticks(yticks)
    #  # 设置 x 轴刻度
    # xticks = np.arange(0, 0.02, 0.005)  # 根据需要调整范围和间隔
    # plt.xticks(xticks)
    plt.show()

def calculate_pearson_correlation(seq1, seq2):
    freq1 = calculate_5mer_frequency(seq1)
    freq2 = calculate_5mer_frequency(seq2)

    common_kmers = set(freq1.keys()) & set(freq2.keys())

    x = np.array([freq1[kmer] for kmer in common_kmers])
    y = np.array([freq2[kmer] for kmer in common_kmers])

    correlation = np.corrcoef(x, y)[0, 1]

    return correlation

# 从文件中读取生成的序列和自然序列
with open("/home/cxm/train/seq-exp/rannatger/tiquGenStr.txt", 'r') as gen_file, open("/home/cxm/train/seq-exp/rannatger/tiquNat.txt", 'r') as natural_file:
    gen_sequence = gen_file.read().strip()
    natural_sequence = natural_file.read().strip()

compare_kmer_frequencies(gen_sequence, natural_sequence)
# 计算相似性
correlation = calculate_pearson_correlation(gen_sequence, natural_sequence)
print(f"Pearson Correlation: {correlation}")



# # 提取生成序列-10区近端启动子
# with open("/home/cxm/train/seq-exp/Generator/cache2/inducible_1fillseq_20240307150102_tiqu_2024-03-07-18-24-16_results.csv", 'r') as input_file:
#     # 跳过首行
#     next(input_file)
#     # 打开输出文件
#     with open('/home/cxm/train/seq-exp/rannatger/GenProximal.txt', 'w') as output_file:
#         for line in input_file:
#             # 按逗号分割每行数据
#             data = line.strip().split(',')
#             if len(data) != 4:
#                 continue  # 确保数据行格式正确
#
#             # realA = data[0]
#             # realB = data[1]
#             # expr = data[2]
#
#             fakeB = data[0]
#             predict = data[1]
#             realA = data[2]
#             realB = data[3]
#
#
#             # 找到 realA 列中最后一个未被 'M' 掩盖的基序的起始位置
#             last_motif_start = -1
#             for i in range(len(realA) - 1, -1, -1):
#                 if realA[i] != 'M':
#                     last_motif_start = i
#                     while last_motif_start > 0 and realA[last_motif_start - 1] != 'M':
#                         last_motif_start -= 1
#                     break
#
#             # 如果找到了基序
#             if last_motif_start != -1:
#                 # 根据位置在 realB 列中找到对应的序列，并提取其前 xx 个碱基
#                 if last_motif_start >= 30:
#                     upstream_seq = fakeB[last_motif_start - 30:last_motif_start]
#                 else:
#                     upstream_seq = fakeB[:last_motif_start]  # 如果基序前不足 50 个碱基
#
#                 # 将提取的序列写入输出文件
#                 output_file.write(upstream_seq + '\n')
#
# # 提取自然序列-10区近端启动子
# with open("/home/cxm/train/seq-exp/data/ecoli_mpra_3_laco.csv", 'r') as input_file:
#     # 跳过首行
#     next(input_file)
#     # 打开输出文件
#     with open('/home/cxm/train/seq-exp/rannatger/NatProximal.txt', 'w') as output_file:
#         for line in input_file:
#             # 按逗号分割每行数据
#             data = line.strip().split(',')
#             if len(data) != 3:
#                 continue  # 确保数据行格式正确
#
#             realA = data[0]
#             realB = data[1]
#             expr = data[2]
#
#
#             # 找到 realA 列中最后一个未被 'M' 掩盖的基序的起始位置
#             last_motif_start = -1
#             for i in range(len(realA) - 1, -1, -1):
#                 if realA[i] != 'M':
#                     last_motif_start = i
#                     while last_motif_start > 0 and realA[last_motif_start - 1] != 'M':
#                         last_motif_start -= 1
#                     break
#
#             # 如果找到了基序
#             if last_motif_start != -1:
#                 # 根据位置在 realB 列中找到对应的序列，并提取其前 xx 个碱基
#                 if last_motif_start >= 30:
#                     upstream_seq = realB[last_motif_start - 30:last_motif_start]
#                 else:
#                     upstream_seq = realB[:last_motif_start]  # 如果基序前不足 50 个碱基
#
#                 # 将提取的序列写入输出文件
#                 output_file.write(upstream_seq + '\n')
#
# # 提取自然序列-10区远端启动子
# # 定义输入和输出文件名
# input_filename = "/home/cxm/train/seq-exp/data/ecoli_mpra_3_laco.csv"
# output_filename = '/home/cxm/train/seq-exp/rannatger/NatDistal.txt'
#
# # 打开输入文件
# with open(input_filename, 'r') as input_file:
#     # 跳过首行
#     next(input_file)
#     # 打开输出文件
#     with open(output_filename, 'w') as output_file:
#         # 逐行读取输入文件
#         for line in input_file:
#             # 按逗号分割每行数据
#             data = line.strip().split(',')
#             if len(data) != 3:
#                 continue  # 确保数据行格式正确
#
#             realB = data[1]
#             # 提取 realA 列的前 50 个碱基
#             upstream_seq = realB[:30]
#
#             # 将提取的序列写入输出文件
#             output_file.write(upstream_seq + '\n')
#
#
# # 提取生成序列-10区远端启动子
# # 定义输入和输出文件名
# input_filename = "/home/cxm/train/seq-exp/Generator/cache2/inducible_1fillseq_20240307150102_tiqu_2024-03-07-18-24-16_results.csv"
# output_filename = '/home/cxm/train/seq-exp/rannatger/GenDistal.txt'
#
# # 打开输入文件
# with open(input_filename, 'r') as input_file:
#     # 跳过首行
#     next(input_file)
#     # 打开输出文件
#     with open(output_filename, 'w') as output_file:
#         # 逐行读取输入文件
#         for line in input_file:
#             # 按逗号分割每行数据
#             data = line.strip().split(',')
#             if len(data) != 4:
#                 continue  # 确保数据行格式正确
#
#             fakeB = data[0]
#             # 提取 realA 列的前 50 个碱基
#             upstream_seq = fakeB[:30]
#
#             # 将提取的序列写入输出文件
#             output_file.write(upstream_seq + '\n')
