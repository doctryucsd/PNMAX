import os
import csv

def find_csv_files(directory):
    csv_files = []
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            # csv_files.append(os.path.join(directory, file))
            csv_files.append(file)
    csv_files.sort()
    return csv_files

"""
As we did not consider the operator fusion in the paper, the latency of entire network is obtained by **summing up the latency** of each operator considering **how many times** it is used in the inference.
For llama2, operators are repeated 32 times for 7B model and 40 times for 13B model, except the `final_proj` and `final_rms_norm`.
For CNNs, we use the batch size of 32, but it will cause a low compilation time for conv layers. So we compile the conv layers with batch size 1, and the rest of the layers with batch size 32. As a result, **the latency of conv layers should be multiplied by 32 when calculating the total latency**.
"""

if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    csv_files = find_csv_files(current_directory)
    result_csv = 'combined_results.csv'
    with open(result_csv, mode='w') as result_file:
        csv_writer = csv.writer(result_file)
        csv_writer.writerow(['workload_architecture', 'lat_optimal', 'cmd_optimal', 'pu_dram_access_optimal', 'host_dram_access_optimal', 'row_change_optimal',
                              'lat_baseline', 'cmd_baseline', 'pu_dram_access_baseline', 'host_dram_access_baseline', 'row_change_baseline'])
        for csv_file in csv_files:
            csv_data = [0 for _ in range(10)] # 10 columns
            csv_file_path = os.path.join(current_directory, csv_file)
            csv_name = csv_file.split('.')[0]
            with open(csv_file_path, mode='r') as file:
                csv_reader = csv.reader(file)
                _ = next(csv_reader)
                # csv_name, csv_data
                if 'llama2' in csv_file:
                    if '7B' in csv_file:
                        for row in csv_reader:
                            layer_name = row[0]
                            layer_data = row[2:]
                            assert len(layer_data) == 10
                            if 'final' in layer_name:
                                csv_data = [int(float(layer_data[i])) + csv_data[i] for i in range(10)]
                            else:
                                csv_data = [int(float(layer_data[i])) * 32 + csv_data[i] for i in range(10)]
                    # prfloat(f"Processing 7B file: {csv_file}")
                    elif '13B' in csv_file:
                        for row in csv_reader:
                            layer_name = row[0]
                            layer_data = row[2:]
                            assert len(layer_data) == 10
                            if 'final' in layer_name:
                                csv_data = [int(float(layer_data[i])) + csv_data[i] for i in range(10)]
                            else:
                                csv_data = [int(float(layer_data[i])) * 40 + csv_data[i] for i in range(10)]
                    else:
                        continue
                elif 'resnet' in csv_file:
                    for row in csv_reader:
                        layer_name = row[0]
                        layer_type = row[1]
                        layer_data = row[2:]
                        assert len(layer_data) == 10
                        if layer_type == 'mm' and layer_name != 'fc':
                            csv_data = [int(float(layer_data[i])) * 32 + csv_data[i] for i in range(10)]
                        else:
                            csv_data = [int(float(layer_data[i])) + csv_data[i] for i in range(10)]
                elif 'vgg' in csv_file:
                    for row in csv_reader:
                        layer_name = row[0]
                        layer_type = row[1]
                        layer_data = row[2:]
                        assert len(layer_data) == 10
                        if layer_type == 'mm':
                            csv_data = [int(float(layer_data[i])) * 32 + csv_data[i] for i in range(10)]
                        else:
                            csv_data = [int(float(layer_data[i])) + csv_data[i] for i in range(10)]
                else:
                    continue
            csv_data = [csv_name] + csv_data
            csv_writer.writerow(csv_data)
