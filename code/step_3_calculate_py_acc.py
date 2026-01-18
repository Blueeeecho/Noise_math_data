import json
import sys
import os

def execute_python_code(code, sample_id):
    try:
        # 创建一个局部命名空间来执行代码
        local_namespace = {}
        exec(code, {}, local_namespace)
        # 调用solve函数获取结果
        if 'solve' in local_namespace:
            result = local_namespace['solve']()
            return result
        else:
            print(f"Sample {sample_id}: Code does not contain 'solve' function")
            return None
    except Exception as e:
        print(f"Sample {sample_id}: Error executing code: {e}")
        return None

def extract_plan_code(sample):
    pred_full = sample.get('pred_full')
    if not isinstance(pred_full, str):
        return None
    start_tag = "<plan>"
    end_tag = "</plan>"
    start = pred_full.find(start_tag)
    end = pred_full.find(end_tag)
    if start == -1 or end == -1:
        return None
    start += len(start_tag)
    if start >= end:
        return None
    return pred_full[start:end].strip()

def get_true_answer(sample):
    if 'true_answer' in sample:
        return sample.get('true_answer')
    gold = sample.get('gold')
    if gold is None:
        return None
    if isinstance(gold, (int, float)):
        return gold
    if isinstance(gold, str):
        s = gold.strip()
        try:
            return float(s)
        except:
            return None
    return None

def calculate_accuracy(jsonl_file_path):
    total_samples = 0
    correct_predictions = 0
    execution_errors = 0
    
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            try:
                sample = json.loads(line)
                total_samples += 1
                
                sample_id = sample.get('sample_id', sample.get('id'))
                code = sample.get('plan_code_excerpt')
                if not code:
                    code = extract_plan_code(sample)
                true_answer = get_true_answer(sample)
                
                if not code:
                    print(f"Sample {sample_id}: No Python code found")
                    continue
                    
                if true_answer is None:
                    print(f"Sample {sample_id}: No true_answer found")
                    continue
                    
                # 执行Python代码
                predicted_answer = execute_python_code(code, sample_id)
                
                if predicted_answer is None:
                    execution_errors += 1
                    continue
                    
                # 比较预测结果和正确答案
                # 考虑到浮点数精度问题，使用近似比较
                if isinstance(predicted_answer, (int, float)) and isinstance(true_answer, (int, float)):
                    if abs(predicted_answer - true_answer) < 1e-6:
                        correct_predictions += 1
                        print(f"Sample {sample_id}: Correct (Predicted: {predicted_answer}, True: {true_answer})")
                    else:
                        print(f"Sample {sample_id}: Incorrect (Predicted: {predicted_answer}, True: {true_answer})")
                else:
                    print(f"Sample {sample_id}: Cannot compare answers (Predicted: {type(predicted_answer).__name__}, True: {type(true_answer).__name__})")
                    
            except json.JSONDecodeError:
                print(f"Error parsing JSON line: {line}")
                continue
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue
    
    # 计算准确率
    if total_samples > 0:
        accuracy = (correct_predictions / total_samples) * 100
        print(f"\n=== Accuracy Report ===")
        print(f"Total Samples: {total_samples}")
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Execution Errors: {execution_errors}")
        print(f"Accuracy: {accuracy:.2f}%")
        return accuracy
    else:
        print("No valid samples found.")
        return 0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python calculate_py_acc.py <jsonl_file_path>")
        sys.exit(1)
    
    jsonl_file_path = sys.argv[1]
    if not os.path.exists(jsonl_file_path):
        print(f"Error: File {jsonl_file_path} does not exist.")
        sys.exit(1)
    
    calculate_accuracy(jsonl_file_path)
